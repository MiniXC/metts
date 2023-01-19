import multiprocessing
from dataclasses import dataclass

import lco
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import PreTrainedModel
from nnAudio.features.mel import MelSpectrogram

num_cpus = multiprocessing.cpu_count()

class ConformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        if "conv_depthwise" in kwargs and kwargs["conv_depthwise"]:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_in"],
                    kwargs["conv_in"],
                    kernel_size=kwargs["conv_kernel"][0],
                    padding="same",
                    groups=kwargs["conv_in"],
                ),
                nn.Conv1d(kwargs["conv_in"], kwargs["conv_filter_size"], 1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_filter_size"],
                    kwargs["conv_filter_size"],
                    kernel_size=kwargs["conv_kernel"][1],
                    padding="same",
                    groups=kwargs["conv_in"],
                ),
                nn.Conv1d(kwargs["conv_filter_size"], kwargs["conv_in"], 1),
            )
        else:
            self.conv1 = nn.Conv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding="same",
            )
            self.conv2 = nn.Conv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding="same",
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_length = lco["max_lengths"]["frame"]

    def forward(self, x, durations):
        MAX_FRAMES = self.target_length
        MAX_PHONES = x.shape[1]
        BATCH_SIZE = x.shape[0]
        EMB_DIM = x.shape[-1]

        val_ind = (torch.zeros((MAX_FRAMES, BATCH_SIZE), dtype=torch.int64).to(x.device)
            .scatter(
                0,
                durations.cumsum(-1).T,
                torch.ones(MAX_FRAMES, BATCH_SIZE, dtype=torch.int64).to(x.device)
            )
            .T.cumsum(-1)
        )

        ind = val_ind + (MAX_PHONES * torch.arange(BATCH_SIZE)).unsqueeze(1)
        val = x.reshape((-1, EMB_DIM))

        x = torch.nn.functional.embedding(ind.to(x.device), val)
        tgt_mask = ~(val_ind.view(x.shape[0], -1) == durations.shape[1]-1)
        
        return x, ~tgt_mask

class MeTTS(nn.Module):
    def __init__(self):
        super().__init__()

        # processing
        self.mel = MelSpectrogram(
            sr=lco["audio"]["sampling_rate"],
            n_fft=lco["audio"]["n_fft"],
            win_length=lco["audio"]["win_length"],
            hop_length=lco["audio"]["hop_length"],
            n_mels=lco["audio"]["n_mels"],
            trainable_mel=True,
            trainable_STFT=True,
        )

        # layers
        self.embedding = nn.Embedding(
                100, 256, padding_idx=0
        )
        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
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
            ConformerEncoderLayer(
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

    def forward(self, phones, durations, audio, **measures):
        x = phones
        print(x)
        durations = durations
        x = self.embedding(x)
        x = self.encoder(x)
        x, mask = self.lr(x, durations)
        x = self.decoder(x)
        x = self.linear(x)
        mel = self.mel(audio).view(x.shape[0], lco["audio"]["n_mels"], -1)
        mel = nn.ConstantPad2d((0, x.shape[-2] - mel.shape[-1], 0, 0), 0)(mel)
        mel = mel.view(x.shape[0], -1, lco["audio"]["n_mels"])

        loss = nn.MSELoss()(x, mel)

        return {
            "logits": x,
            "loss": loss
        }