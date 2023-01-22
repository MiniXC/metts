import multiprocessing
from dataclasses import dataclass

import lco
import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from transformers import PreTrainedModel
from nnAudio.features.mel import MelSpectrogram

from .transformer import TransformerEncoder
from .conformer_layer import ConformerLayer
from .length_regulator import LengthRegulator
from .diffusion_vocoder import DiffusionVocoder

num_cpus = multiprocessing.cpu_count()

class MeTTS(PreTrainedModel):
    def __init__(self, config):
        super().__init__()

        self.positional_encoding = Summer(PositionalEncoding1D(256))

        # processing
        self.mel = MelSpectrogram(
            sr=lco["audio"]["sampling_rate"],
            n_fft=lco["audio"]["n_fft"],
            win_length=lco["audio"]["win_length"],
            hop_length=lco["audio"]["hop_length"],
            n_mels=lco["audio"]["n_mels"],
            # trainable_mel=True,
            # trainable_STFT=True,
        )

        # layers
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
        )
        self.linear = nn.Linear(256, 80)

        self.diffusion_vocoder = DiffusionVocoder()

    def forward(self, phones, phone_durations, audio, **measures):
        x = phones
        batch_size = x.shape[0]
        durations = phone_durations
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.lr(x, durations)
        x = self.positional_encoding(x)
        x = self.decoder(x)
        x = self.linear(x)
        mel = self.mel(audio).view(batch_size, lco["audio"]["n_mels"], -1)
        mel = nn.ConstantPad2d((batch_size, x.shape[-2] - mel.shape[-1], 0, 0), 0)(mel)
        mel = mel.view(batch_size, -1, lco["audio"]["n_mels"])


        loss = nn.MSELoss()(x, mel)

        return {
            "logits": x,
            "loss": loss
        }