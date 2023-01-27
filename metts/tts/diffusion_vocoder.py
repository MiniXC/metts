import math

import torch
from torch import nn
import torch.nn.functional as F
import lco
import numpy as np

from .convolutions import ConvolutionLayer, Transpose, DepthwiseConv1d

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
  def __init__(self, n_mels):
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
        assert (conditioner is None and self.conditioner_projection is None) or \
                (conditioner is not None and self.conditioner_projection is not None)

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
        self.spectrogram_upsampler = SpectrogramUpsampler(lco["audio"]["n_mels"])

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
    
# class MelUpsampler(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mel_upsample = nn.ModuleList(
#             [
#                 nn.utils.weight_norm(
#                     nn.nn.ConvTranspose2d(
#                         1,
#                         1,
#                         (3, 2 * s),
#                         padding=(1, s // 2),
#                         stride=(1, s)
#                     )
#                 ) for s in [16, 16]
#             ]
#         )
#         self.mel_projection = Transpose(nn.nn.Linear(
#             lco["diffusion_vocoder"]["conv_channels"],
#             lco["diffusion_vocoder"]["conv_channels"]
#         ))

#     def forward(self, mel):
#         mel = mel.unsqueeze(1)
#         for upsample in self.mel_upsample:
#             mel = F.leaky_relu(upsample(mel), 0.4, inplace=False)
#         mel = mel.squeeze(1)
#         mel = self.mel_projection(mel)
#         return mel

# class DiffusionVocoder(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.step_embedding = StepEmbedding()
#         self.init_conv = nn.Sequential(
#             Conv(1, lco["diffusion_vocoder"]["conv_channels"], kernel_size=1),
#             nn.ReLU(inplace=False)
#         )
#         self.mel_upsampler = MelUpsampler()
#         kernel_size = lco["diffusion_vocoder"]["conv_kernel_size"]
#         self.layers = nn.ModuleList([
#             ConvolutionLayer(
#                 in_channels=lco["diffusion_vocoder"]["conv_channels"],
#                 filter_size=lco["diffusion_vocoder"]["conv_filter_size"],
#                 out_channels=lco["diffusion_vocoder"]["conv_channels"],
#                 kernel_size=kernel_size,
#                 padding=(kernel_size - 1) // 2
#             ) for _ in range(lco["diffusion_vocoder"]["conv_layers"])
#         ])
#         self.nn.Linear = Transpose(nn.nn.Linear(lco["diffusion_vocoder"]["conv_channels"], 1))
    
#     @staticmethod
#     def compute_diffusion_params(beta):
#         """
#         Compute diffusion parameters from beta
#         source: https://github.com/tencent-ailab/bddm/blob/2cebe0e6b7fd4ce8121a45d1194e2eb708597936/bddm/utils/diffusion_utils.py#L16
#         """
#         alpha = 1 - beta
#         sigma = beta + 0
#         for t in range(1, len(beta)):
#             alpha[t] *= alpha[t-1]
#             sigma[t] *= (1-alpha[t-1]) / (1-alpha[t])
#         alpha = torch.sqrt(alpha)
#         sigma = torch.sqrt(sigma)
#         diff_params = {"T": len(beta), "beta": beta, "alpha": alpha, "sigma": sigma}
#         return diff_params

#     def forward(self, x, mel, steps):
#         diff_embed = self.step_embedding(steps)
#         diff_embed = diff_embed.view(x.shape[0], lco["diffusion_vocoder"]["conv_channels"], -1)
#         x = x.unsqueeze(1)
#         x = self.init_conv(x)
#         mel = self.mel_upsampler(mel)
#         x = x + diff_embed + mel
#         for layer in self.layers:
#             x = layer(x)
#         x = self.nn.Linear(x).squeeze(1)
#         return x
    