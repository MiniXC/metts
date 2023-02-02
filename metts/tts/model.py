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
from .diffusion_vocoder import DiffWave, DiffWaveSampler

num_cpus = multiprocessing.cpu_count()

class MeTTSConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MeTTS(PreTrainedModel):
    config_class = MeTTSConfig

    def __init__(self, config):
        super().__init__(config)

        self.positional_encoding = PositionalEncoding(256)

        # processing
        # self.mel = MelSpectrogram(
        #     sr=lco["audio"]["sampling_rate"],
        #     n_fft=lco["audio"]["n_fft"],
        #     win_length=lco["audio"]["win_length"],
        #     hop_length=lco["audio"]["hop_length"],
        #     n_mels=lco["audio"]["n_mels"],
        #     pad_mode="constant",
        #     power=2,
        #     htk=True,
        #     fmin=0,
        #     fmax=8000,
        #     # trainable_mel=True,
        #     # trainable_STFT=True,
        # )

        # layers
        self.speaker_embedding = nn.Embedding(2500, 256)
        self.embedding = nn.Embedding(
            100, 256, padding_idx=0
        )
        self.encoder = TransformerEncoder(
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
            num_layers=4,
        )

        # attributes
        self.attribute_transformer = TransformerEncoder(
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
        self.attribute_linear = nn.Linear(512, 256 + (2 * 5)) # 4 measures

        self.lr = LengthRegulator()

        self.decoder = TransformerEncoder(
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

    @staticmethod
    def drc(x, C=1, clip_val=1e-5, log10=True):
        """Dynamic Range Compression"""
        if log10:
            return torch.log10(torch.clamp(x, min=clip_val) * C)
        else:
            return torch.log(torch.clamp(x, min=clip_val) * C)

    def forward(self, phones, phone_durations, mel, val_ind, speaker, vocoder_mask, vocoder_audio, **kwargs):
        ### Transformer TTS
        tts = self.tts_fwd(phones, phone_durations, mel, val_ind, speaker)

        ### Diffusion Vocoder
        vocoder = self.vocoder_fwd(tts["mel"][:, :, vocoder_mask], tts["hidden"][:, vocoder_mask, :], vocoder_audio)

        return {
            "loss": tts["loss"] + vocoder["loss"],
        }

    def attribute_fwd(self, x, speaker, dvecs):
        x = x + speaker.unsqueeze(1)
        x = self.attribute_transformer(x)
        # average + max pooling
        hidden = torch.cat(
            (torch.mean(x, dim=1), torch.max(x, dim=1)[0]),
            dim=-1,
        )
        attr = self.attribute_linear(x)

        return attr

    def tts_fwd(self, x, durations, mel, val_ind, speaker):
        batch_size = x.shape[0]
        x = self.embedding(x)
        speaker = self.speaker_embedding(speaker)
        x = x + speaker.unsqueeze(1)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        #self.attribute_fwd(x, speaker)

        x, mask = self.lr(x, durations, val_ind)
        x = self.positional_encoding(x)
        x, hidden = self.decoder(x)
        x = self.linear(x)

        if x.shape[1] > mel.shape[1]:
            mel = nn.ConstantPad2d((0, 0, 0, x.shape[1] - mel.shape[1]), 0)(mel)
        else:
            mel = mel[:, :x.shape[1]]
        mask = mask.unsqueeze(-1)
        mel = mel.transpose(1, 2)

        x = x.transpose(1, 2)

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])
        loss = nn.MSELoss()(x, mel) * mask_scale

        return {
            "loss": loss,
            "hidden": hidden,
            "mel": mel,
        }

    def vocoder_fwd(self, mel, hidden, audio):
        batch_size = mel.shape[0]
        hidden = self.hidden_linear(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = mel + hidden * 0.1
        ts = torch.randint(low=0, high=lco["diffusion_vocoder"]["T"], size=(batch_size, 1))
        noise_scales = self.diff_params["alpha"].to(mel.device)[ts]
        z = torch.normal(0, 1, size=audio.shape).to(mel.device)
        noisy_audios = noise_scales * audio + (1 - noise_scales**2.).sqrt() * z
        e = self.diffusion_vocoder(noisy_audios, hidden, ts.to(mel.device))
        loss = nn.L1Loss()(e, z)

        return {
            "loss": loss,
            "logits": e,
        }

    def generate(self, phones, durations, val_ind, audio):
        sampler = DiffWaveSampler(self.diffusion_vocoder, self.diff_params)

        x = phones
        batch_size = x.shape[0]
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x, mask = self.lr(x, durations, val_ind)
        x = self.positional_encoding(x)
        x, hidden = self.decoder(x)
        x = self.linear(x)

        mel = self.mel(audio).transpose(1, 2)
        if x.shape[-2] > mel.shape[1]:
            mel = nn.ConstantPad2d((0, 0, 0, x.shape[1] - mel.shape[1]), 0)(mel)
        else:
            mel = mel[:, :x.shape[1]]
        mask = mask.unsqueeze(-1)
        mel = mel.transpose(1, 2)
        mel = MeTTS.drc(mel)

        x = x.transpose(1, 2)

        hidden = self.hidden_linear(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = hidden + x

        audio = sampler(hidden, "original")
        

        return audio #x.transpose(1, 2)