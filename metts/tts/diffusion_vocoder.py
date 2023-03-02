import math

import torch
from torch import nn
import torch.nn.functional as F
import lco
import numpy as np
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PretrainedConfig
from nnAudio.features.mel import MelSpectrogram

from .convolutions import ConvolutionLayer, Transpose, DepthwiseConv1d


lco.init("config/config.yaml")

class StepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_t1 = nn.Linear(
            lco["diffusion_vocoder"]["step_embed_dim_in"],
            lco["diffusion_vocoder"]["step_embed_dim_hidden"]
        )
        self.fc_t2 = nn.Linear(
            lco["diffusion_vocoder"]["step_embed_dim_hidden"],
            lco["diffusion_vocoder"]["step_embed_dim_out"]
        )

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, steps):
        half_dim = lco["diffusion_vocoder"]["step_embed_dim_in"] // 2
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

class SpectrogramUpsampler(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = nn.ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = DepthwiseConv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.conditioner_projection = DepthwiseConv1d(n_mels, 2 * residual_channels, 1)

        self.output_projection = DepthwiseConv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWave(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_projection = DepthwiseConv1d(1, lco["diffusion_vocoder"]["residual_channels"], 1)
        self.diffusion_embedding = StepEmbedding()
        self.spectrogram_upsampler = SpectrogramUpsampler()

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                lco["audio"]["n_mels"],
                lco["diffusion_vocoder"]["residual_channels"],
                2**(i % lco["diffusion_vocoder"]["dilation_cycle_length"]),
            )
            for i in range(lco["diffusion_vocoder"]["residual_layers"])
        ])
        self.skip_projection = DepthwiseConv1d(lco["diffusion_vocoder"]["residual_channels"], lco["diffusion_vocoder"]["residual_channels"], 1)
        self.output_projection = DepthwiseConv1d(lco["diffusion_vocoder"]["residual_channels"], 1, 1)
        nn.init.zeros_(self.output_projection.depthwise_conv.weight)
        nn.init.zeros_(self.output_projection.pointwise_conv.weight)

    @staticmethod
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

    def forward(self, audio, spectrogram, diffusion_step):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x).squeeze(1)
        return x
        
class DiffWaveSampler():

    # from https://github.com/Rongjiehuang/FastDiff
    noise_schedules = {
        "original": torch.linspace(
            lco["diffusion_vocoder"]["beta_0"],
            lco["diffusion_vocoder"]["beta_T"],
            lco["diffusion_vocoder"]["T"]
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

    def __init__(self, model, diff_params):
        self.model = model
        self.diff_params = diff_params

    def __call__(self, c, N=4, bs=1):
        if N not in self.noise_schedules:
            raise ValueError(f"Invalid noise schedule length {N}")

        noise_schedule = self.noise_schedules[N]

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule)

        noise_schedule = noise_schedule.to(c.device)
        noise_schedule = noise_schedule.to(torch.float32)

        audio_length = c.shape[-1] * lco["audio"]["hop_length"]

        pred_wav = self.sampling_given_noise_schedule(
            (bs, audio_length),
            noise_schedule,
            c,
        )

        pred_wav = pred_wav / max(pred_wav.abs().max(), 1e-5)
        #pred_wav = pred_wav.view(-1)
        return pred_wav

    def sampling_given_noise_schedule(
        self,
        size,
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

        _dh = self.diff_params
        T, alpha = _dh["T"], _dh["alpha"]
        assert len(alpha) == T
        assert len(size) == 3 or len(size) == 2

        N = len(noise_schedule)
        beta_infer = noise_schedule
        alpha_infer = [1 - float(x) for x in beta_infer] 
        sigma_infer = beta_infer + 0

        for n in range(1, N):
            alpha_infer[n] *= alpha_infer[n - 1]
            sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
        alpha_infer = torch.FloatTensor([np.sqrt(x) for x in alpha_infer])
        sigma_infer = torch.sqrt(sigma_infer)

        torch.set_printoptions(precision=10)

        # Mapping noise scales to time steps
        steps_infer = []
        for n in range(N):
            step = self.map_noise_scale_to_time_step(alpha_infer[n], alpha)
            if step >= 0:
                steps_infer.append(step)
        steps_infer = torch.FloatTensor(steps_infer)

        N = len(steps_infer)

        x = torch.normal(0, 1, size=size).to(c.device)
        with torch.no_grad():
            for n in range(N - 1, -1, -1):
                ts = (steps_infer[n] * torch.ones((size[0], 1))).to(c.device)
                e = self.model(x, c, ts)
                x -= beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * e
                x /= torch.sqrt(1 - beta_infer[n])
                if n > 0:
                    z = torch.normal(0, 1, size=size).to(c.device)
                    x = x + sigma_infer[n] * z

        return x

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

class VocoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Vocoder(PreTrainedModel):
    config_class = VocoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.diffusion_vocoder = DiffWave()
        noise_schedule = torch.linspace(
            lco["diffusion_vocoder"]["beta_0"],
            lco["diffusion_vocoder"]["beta_T"],
            lco["diffusion_vocoder"]["T"]
        )
        self.mel = MelSpectrogram(
            sr=lco["audio"]["sampling_rate"],
            n_fft=lco["audio"]["n_fft"],
            win_length=lco["audio"]["win_length"],
            hop_length=lco["audio"]["hop_length"],
            n_mels=lco["audio"]["n_mels"],
            pad_mode="constant",
            power=2,
            htk=True,
            fmin=0,
            fmax=8000,
            trainable_mel=lco["diffusion_vocoder"]["learnable_mel"],
            trainable_STFT=lco["diffusion_vocoder"]["learnable_mel"],
        )
        self.diff_params = DiffWave.compute_diffusion_params(noise_schedule)
        self.sampler = DiffWaveSampler(self.diffusion_vocoder, self.diff_params)

    @staticmethod
    def drc(x, C=1, clip_val=1e-5, log10=True):
        """Dynamic Range Compression"""
        if log10:
            return torch.log10(torch.clamp(x, min=clip_val) * C)
        else:
            return torch.log(torch.clamp(x, min=clip_val) * C)

    def forward(self, vocoder_mel, vocoder_audio, **kwargs):
        audio = vocoder_audio
        batch_size = audio.shape[0]
        c = vocoder_mel
        ts = torch.randint(low=0, high=lco["diffusion_vocoder"]["T"], size=(batch_size, 1))
        noise_scales = self.diff_params["alpha"].to(c.device)[ts]
        z = torch.normal(0, 1, size=audio.shape).to(c.device)
        noisy_audios = noise_scales * audio + (1 - noise_scales**2.).sqrt() * z
        e = self.diffusion_vocoder(noisy_audios, c, ts.to(c.device))
        loss = nn.MSELoss()(e, z)

        return {
            "loss": loss,
            "logits": e,
        }

    def generate(self, mel, n, bs=1):
        return self.sampler(mel, n, bs)