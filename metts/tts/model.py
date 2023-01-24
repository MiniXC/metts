import multiprocessing
from dataclasses import dataclass

import lco
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
from transformers import PreTrainedModel, PretrainedConfig

from nnAudio.features.mel import MelSpectrogram

from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .length_regulator import LengthRegulator
from .diffusion_vocoder import DiffWave

num_cpus = multiprocessing.cpu_count()

class MeTTSConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MeTTS(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.positional_encoding = PositionalEncoding(256)

        # processing
        self.mel = MelSpectrogram(
            sr=lco["audio"]["sampling_rate"],
            n_fft=lco["audio"]["n_fft"],
            win_length=lco["audio"]["win_length"],
            hop_length=lco["audio"]["hop_length"],
            n_mels=lco["audio"]["n_mels"],
            pad_mode="constant",
            # trainable_mel=True,
            # trainable_STFT=True,
        )

        # layers
        self.embedding = nn.Embedding(
                100, 256, padding_idx=0
        )
        self.encoder = TransformerEncoder(
            # ConformerLayer(
            #     256,
            #     2,
            #     conv_in=256,
            #     conv_filter_size=1024,
            #     conv_kernel=(9, 1),
            #     batch_first=True,
            #     dropout=0.1,
            #     conv_depthwise=False,
            # ),
            TransformerEncoderLayer(256, 2, 256),
            num_layers=4,
        )
        self.lr = LengthRegulator()
        self.decoder = TransformerEncoder(
            # ConformerLayer(
            #     256,
            #     2,
            #     conv_in=256,
            #     conv_filter_size=1024,
            #     conv_kernel=(9, 1),
            #     batch_first=True,
            #     dropout=0.1,
            #     conv_depthwise=False,
            # ),
            TransformerEncoderLayer(256, 2, 256),
            num_layers=4,
            return_additional_layer=2,
        )
        self.hidden_linear = nn.Linear(256, 80)
        self.linear = nn.Linear(256, 80)

        self.diffusion_vocoder = DiffWave()
        noise_schedule = torch.linspace(
            lco["diffusion_vocoder"]["beta_0"],
            lco["diffusion_vocoder"]["beta_T"],
            lco["diffusion_vocoder"]["T"]
        )
        self.diff_params = DiffWave.compute_diffusion_params(noise_schedule)

    def forward(self, phones, phone_durations, audio, val_ind, **measures):
        ### Transformer TTS
        x = phones
        batch_size = x.shape[0]
        durations = phone_durations
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x, mask = self.lr(x, durations, val_ind)
        x = self.positional_encoding(x)
        x, hidden = self.decoder(x)
        x = self.linear(x)
        mel = self.mel(audio).view(batch_size, lco["audio"]["n_mels"], -1)
        mel = nn.ConstantPad2d((0, x.shape[-2] - mel.shape[-1], 0, 0), 0)(mel)
        mask = mask.unsqueeze(-1)
        disc_loss = nn.MSELoss()(x, mel.view(batch_size, -1, lco["audio"]["n_mels"]))

        ### Diffusion Vocoder
        hidden = self.hidden_linear(hidden)
        hidden = hidden[:, :lco["max_lengths"]["vocoder"], :]
        hidden = hidden.view(batch_size, 80, -1)
        audio = audio[:, :lco["max_lengths"]["vocoder"]*256]
        ts = torch.randint(low=0, high=lco["diffusion_vocoder"]["T"], size=(batch_size, 1)).to(x.device)
        noise_scales = self.diff_params["alpha"].to(x.device)[ts]
        z = torch.normal(0, 1, size=audio.shape).to(x.device)
        noisy_audios = noise_scales * audio + (1 - noise_scales**2.).sqrt() * z
        e = self.diffusion_vocoder(noisy_audios, hidden, ts)
        theta_loss = nn.L1Loss()(e, z)

        return {
            "logits": x,
            "loss": disc_loss,
        }