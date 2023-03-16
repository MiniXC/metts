import math

import torch
from torch import nn
import torch.nn.functional as F
import lco
import numpy as np

from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer

lco.init("config/config.yaml")

class DiffusionSampler():

    # from https://github.com/Rongjiehuang/FastDiff
    noise_schedules = {
        "original": torch.linspace(
            lco["diffusion"]["beta_0"],
            lco["diffusion"]["beta_T"],
            lco["diffusion"]["num_steps"]
        ),
        1000: torch.linspace(0.000001, 0.01, 1000),
        200: torch.linspace(0.0001, 0.02, 200),
        8: [
            6.689325005027058e-07,
            1.0033881153503899e-05,
            0.00015496854030061513,
            0.002387222135439515,
            0.035597629845142365,
            0.3681158423423767,
            0.4735414385795593,
            0.5,
        ],
        6: [
            1.7838445955931093e-06,
            2.7984189728158526e-05,
            0.00043231004383414984,
            0.006634317338466644,
            0.09357017278671265,
            0.6000000238418579
        ],
        4: [
            3.2176e-04,
            2.5743e-03,
            2.5376e-02,
            7.0414e-01
        ],
        3: [
            9.0000e-05,
            9.0000e-03,
            6.0000e-01
        ]
    }

    def __init__(self, model):
        self.model = model
        self.diff_params = model.diff_params

    def __call__(self, c, N=4, bs=1):
        if N not in self.noise_schedules:
            raise ValueError(f"Invalid noise schedule length {N}")

        noise_schedule = self.noise_schedules[N]

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule)

        noise_schedule = noise_schedule.to(c.device)
        noise_schedule = noise_schedule.to(torch.float32)

        frame_pred, sequence_pred = self.sampling_given_noise_schedule(
            noise_schedule,
            c,
        )

        return frame_pred, sequence_pred

    def sampling_given_noise_schedule(
        self,
        noise_schedule,
        c
    ):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
        Parameters:
        net (torch network):            the wavenet models
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors
        condition (torch.tensor):       ground truth mel spectrogram read from disk
                                        None if used for unconditional generation
        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        batch_size = c.shape[0]
        sequence_length = c.shape[1]

        _dh = self.diff_params
        T, alpha = _dh["T"], _dh["alpha"]
        assert len(alpha) == T

        N = len(noise_schedule)
        beta_infer = noise_schedule
        alpha_infer = [1 - float(x) for x in beta_infer] 
        sigma_infer = beta_infer + 0

        for n in range(1, N):
            alpha_infer[n] *= alpha_infer[n - 1]
            sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
        alpha_infer = torch.FloatTensor([np.sqrt(x) for x in alpha_infer])
        sigma_infer = torch.sqrt(sigma_infer)

        # Mapping noise scales to time steps
        steps_infer = []
        for n in range(N):
            step = self.map_noise_scale_to_time_step(alpha_infer[n], alpha)
            if step >= 0:
                steps_infer.append(step)
        steps_infer = torch.FloatTensor(steps_infer)

        N = len(steps_infer)

        xs = [
                torch.normal(0, 1, size=(batch_size, sequence_length, self.model.frame_level_outputs)).to(c.device),
            ]

        if self.model.sequence_level_outputs > 0:
            xs.append(torch.normal(0, 1, size=(batch_size, self.model.sequence_level_outputs)).to(c.device))
 
        with torch.no_grad():
            for n in range(N - 1, -1, -1):
                step = (steps_infer[n] * torch.ones((batch_size, 1))).to(c.device)
                if self.model.sequence_level_outputs > 0:
                    es = self.model._forward(c, step, xs[0], xs[1])
                else:
                    es = self.model._forward(c, step, xs[0])
                for i in range(len(xs)):
                    xs[i] = xs[i] - (beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * es[i])
                    xs[i] = xs[i] / torch.sqrt(1 - beta_infer[n])
                if n > 0:
                    if i in range(len(xs)):
                        z = torch.normal(0, 1, size=xs[i].shape).to(c.device)
                        xs[i] = xs[i] + sigma_infer[n] * z

        frame_pred = xs[0]
        
        if self.model.sequence_level_outputs > 0:
            sequence_pred = xs[1]
        else:
            sequence_pred = None

        return frame_pred, sequence_pred

    def map_noise_scale_to_time_step(self, alpha_infer, alpha):
        if alpha_infer < alpha[-1]:
            return len(alpha) - 1
        if alpha_infer > alpha[0]:
            return 0
        for t in range(len(alpha) - 1):
            if alpha[t+1] <= alpha_infer <= alpha[t]:
                step_diff = alpha[t] - alpha_infer
                step_diff /= alpha[t] - alpha[t+1]
                return t + step_diff.item()
        return -1

def compute_diffusion_params(beta):
    """
    Compute diffusion parameters from beta
    source: https://github.com/tencent-ailab/bddm/blob/2cebe0e6b7fd4ce8121a45d1194e2eb708597936/bddm/utils/diffusion_utils.py#L16
    """
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, len(beta)):
        alpha[t] *= alpha[t-1]
        sigma[t] *= (1-alpha[t-1]) / (1-alpha[t])
    alpha = torch.sqrt(alpha)
    sigma = torch.sqrt(sigma)
    diff_params = {"T": len(beta), "beta": beta, "alpha": alpha, "sigma": sigma}
    return diff_params

class StepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_t1 = nn.Linear(
            lco["diffusion"]["step_embed_dim_in"],
            lco["diffusion"]["step_embed_dim_hidden"]
        )
        self.fc_t2 = nn.Linear(
            lco["diffusion"]["step_embed_dim_hidden"],
            lco["diffusion"]["step_embed_dim_out"]
        )

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, steps):
        half_dim = lco["diffusion"]["step_embed_dim_in"] // 2
        _embed = np.log(10000) / (half_dim - 1)
        _embed = torch.exp(torch.arange(half_dim) * -_embed)
        _embed = steps * _embed
        diff_embed = torch.cat(
            (torch.sin(_embed), torch.cos(_embed)),
            1
        ).to(steps.device)
        diff_embed = StepEmbedding.swish(self.fc_t1(diff_embed))
        diff_embed = StepEmbedding.swish(self.fc_t2(diff_embed))
        return diff_embed

class DiffusionConformer(nn.Module):
    def __init__(self, in_channels, frame_level_outputs, sequence_level_outputs):
        super().__init__()
        noise_schedule = torch.linspace(
            lco["diffusion"]["beta_0"],
            lco["diffusion"]["beta_T"],
            lco["diffusion"]["num_steps"]
        )
        self.diff_params = compute_diffusion_params(noise_schedule)

        self.conditional_in = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.step_in = nn.Sequential(
            nn.Linear(lco["diffusion"]["step_embed_dim_out"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.x_frame_in = nn.Sequential(
            nn.Linear(frame_level_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.positional_encoding = PositionalEncoding(256)
        self.frame_level_outputs = frame_level_outputs
        self.sequence_level_outputs = sequence_level_outputs
        self.step_embed = StepEmbedding()
        self.conformer = TransformerEncoder(
            ConformerLayer(
                256,
                2,
                conv_in=256,
                conv_filter_size=1024,
                conv_kernel=(9, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=True,
            ),
            num_layers=2,
        )
        self.out_layer_frame = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, frame_level_outputs),
        )

        if self.sequence_level_outputs > 0:
            self.out_layer_sequence = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, sequence_level_outputs),
            )

    def _forward(self, c, step, x_frame, x_sequence=None):
        step_embed = self.step_embed(step)

        step_embed = self.step_in(step_embed)
        c = self.conditional_in(c)
        x_frame = self.x_frame_in(x_frame)
        x = x_frame + step_embed.unsqueeze(1) + c
        x = self.positional_encoding(x)
        out = self.conformer(x)
        frame_out = self.out_layer_frame(x)

        # crashing somewhere before this

        # use max + avg pooling to get sequence level conditional embedding
        sequence_out = torch.cat(
            (
                torch.max(out, 1)[0],
                torch.mean(out, 1),
            ),
            1
        )
        if self.sequence_level_outputs > 0:
            sequence_out = self.out_layer_sequence(sequence_out)
        else:
            sequence_out = None
        return frame_out, sequence_out

        # TODO: add support for sequence level outputs

    def forward(self, c, x_frame, x_sequence=None):
        batch_size = x_frame.shape[0]
        step = torch.randint(low=0, high=self.diff_params["T"], size=(batch_size,1,1))
        
        noise_scale = self.diff_params["alpha"].to(c.device)[step]
        delta = (1 - noise_scale**2).sqrt()
        
        z = torch.normal(0, 1, size=x_frame.shape).to(c.device)

        x_frame = (x_frame * noise_scale) + (z * delta)
        step = step.view(batch_size, 1).to(c.device)

        return z, *self._forward(c, step, x_frame, x_sequence)