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
# from .diffusion_vocoder import DiffWave, DiffWaveSampler
from .refinement_models import DiffusionConformer, DiffusionSampler
from .scaler import GaussianMinMaxScaler

num_cpus = multiprocessing.cpu_count()

class MeTTSConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FastSpeechWithConsistency(PreTrainedModel):
    config_class = MeTTSConfig

    def __init__(self, config, consistency_net):
        super().__init__(config)

        self.loss_compounds = [
            "mel",
            "mel_diff",
            "duration_predictor",
            "duration_predictor_diff",
            "measure_predictor",
            "measure_predictor_diff",
            "measure_consistency",
            "dvector_predictor",
            "dvector_predictor_diff",
            "dvector_consistency",
        ]

        # consistency
        self.con = consistency_net
        for param in self.con.parameters():
            param.requires_grad = False

        self.positional_encoding = PositionalEncoding(256)

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
                conv_depthwise=lco["conformer"]["depthwise"],
            ),
            num_layers=4,
        )

        # durations
        self.duration_transformer = TransformerEncoder(
            ConformerLayer(
                256,
                2,
                conv_in=256,
                conv_filter_size=256,
                conv_kernel=(3, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=lco["conformer"]["depthwise"],
            ),
            num_layers=2,
        )
        self.durations_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.positional_encoding_durations = PositionalEncoding(257)
        self.durations_diffuser = DiffusionConformer(
            in_channels=257, # hidden dim + predicted duration
            frame_level_outputs=1,
            sequence_level_outputs=0,
        )
        self.durations_sampler = DiffusionSampler(
            self.durations_diffuser,
        )
        self.duration_scaler = GaussianMinMaxScaler(10)

        self.lr = LengthRegulator()

        # measures & dvector
        self.measure_transformer = TransformerEncoder(
            ConformerLayer(
                256,
                2,
                conv_in=256,
                conv_filter_size=1024,
                conv_kernel=(9, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=lco["conformer"]["depthwise"],
            ),
            num_layers=2,
        )
        self.measures_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        ) # 4 measures
        self.measures_dvector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.measures_dvector_in = nn.Linear(256, 256 + 4)
        self.positional_encoding_measures = PositionalEncoding(256 + 4)
        self.measures_diffuser = DiffusionConformer(
            in_channels=256 + 4, # hidden dim + 4 measures
            frame_level_outputs=4,
            sequence_level_outputs=256, # dvector
        )
        self.measures_sampler = DiffusionSampler(
            self.measures_diffuser,
        )

        self.dvector_to_encoder = nn.Linear(256, 256)
        # measure quantization
        nbins = 256
        self.bins = nn.Parameter(torch.linspace(-5, 5, nbins), requires_grad=False)
        for measure in self.con.measures:
            setattr(self, f"{measure}_embed", nn.Embedding(nbins, 256))

        self.decoder = TransformerEncoder(
            ConformerLayer(
                256,
                2,
                conv_in=256,
                conv_filter_size=1024,
                conv_kernel=(9, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=lco["conformer"]["depthwise"],
            ),
            num_layers=6,
            return_additional_layer=4,
        )
        self.final = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 80),
        )
        self.mel_dvector_in = nn.Linear(256, 256 + 80)
        self.positional_encoding_mel = PositionalEncoding(256)
        self.hidden_to_mel = nn.Linear(256, 80)
        self.mel_diffuser = DiffusionConformer(
            in_channels=80, # hidden dim + predicted mel
            frame_level_outputs=80,
            sequence_level_outputs=0,
        )
        self.mel_sampler = DiffusionSampler(
            self.mel_diffuser,
        )

        self.diffusion_steps_per_forward = 1

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, phones, phone_durations, durations, mel, val_ind, speaker, inference=False, force_tf=False):
        tf = True # force_tf or (torch.rand(1).item() < 0.8)
        inference = False

        loss_dict = {}

        norm_mel = self.con.scalers["mel"].transform(mel)

        if self.duration_scaler._n <= 1_000_000:
            self.duration_scaler.partial_fit(durations)
        norm_durations = self.duration_scaler.transform(durations)

        ### Encoder
        batch_size = phones.shape[0]
        x = self.embedding(phones)
        speaker = self.speaker_embedding(speaker)
        x = x + speaker.unsqueeze(1)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        ### Duration Prediction
        pred_durations_disc = self.duration_transformer(x)
        pred_durations_disc = self.durations_linear(pred_durations_disc).squeeze(-1)
        loss_dict["duration_predictor"] = nn.MSELoss()(pred_durations_disc, norm_durations)

        #### Duration Diffusion
        duration_diffuser_input = torch.cat([x, pred_durations_disc.unsqueeze(-1)], dim=-1)
        duration_diffuser_input = self.positional_encoding_durations(duration_diffuser_input)

        norm_durations = norm_durations.to(duration_diffuser_input.dtype)
        duration_diff = norm_durations.unsqueeze(-1) - pred_durations_disc.unsqueeze(-1)
        loss_dict["duration_predictor_diff"] = 0.0
        for _ in range(self.diffusion_steps_per_forward):
            true_duration_noise, pred_duration_noise, _, _ = self.durations_diffuser(duration_diffuser_input, duration_diff)
            loss_dict["duration_predictor_diff"] += nn.MSELoss()(true_duration_noise, pred_duration_noise) / self.diffusion_steps_per_forward

        ### Length Regulator
        if not inference:
            x, mask = self.lr(x, phone_durations, val_ind)
            if x.shape[1] > norm_mel.shape[1]:
                x = x[:, :norm_mel.shape[1]]
        else:
            duration_diff, _ = self.durations_sampler(
                duration_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_durations = pred_durations_disc.squeeze(-1) + duration_diff.squeeze(-1)
            pred_durations = self.duration_scaler.inverse_transform(pred_durations)
            pred_durations = torch.round(pred_durations).long()
            if (pred_durations < 0).any():
                print("negative duration, setting to 0")
                pred_durations[pred_durations < 0] = 0
            pred_durations[(phones == 0) | (phones == 46)] = 0
            x, mask = self.lr(x, pred_durations)

        ### Consistency
        consistency_result = self.con(norm_mel, inference=True)
        true_measures = consistency_result["logits"]
        true_dvector = consistency_result["logits_dvector"]

        ### Measure Prediction
        pred_measures = self.measure_transformer(x)
        pred_measures = self.measures_linear(pred_measures)
        pred_measures = pred_measures.transpose(1, 2)
        # max pool and avg pool for dvector        
        pred_dvector = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        pred_dvector = self.measures_dvector(pred_dvector)
        #### Measure Loss
        pred_measures = pred_measures * mask.transpose(1, 2)
        true_measures = true_measures * mask.transpose(1, 2)
        loss_dict["measure_predictor"] = nn.MSELoss()(pred_measures, true_measures)
        loss_dict["dvector_predictor"] = nn.MSELoss()(pred_dvector, true_dvector)
        ### Measure Diffusion
        measure_diffuser_input = torch.cat([x, pred_measures.transpose(1, 2)], dim=-1)
        measure_diffuser_dvec = self.measures_dvector_in(pred_dvector)
        measure_diffuser_input = measure_diffuser_input + measure_diffuser_dvec.unsqueeze(1)
        measure_diffuser_input = self.positional_encoding_measures(measure_diffuser_input)
        measures_diff = true_measures.transpose(1, 2) - pred_measures.transpose(1, 2)
        dvector_diff = true_dvector - pred_dvector
        loss_dict["measure_predictor_diff"] = 0.0
        loss_dict["dvector_predictor_diff"] = 0.0
        for _ in range(self.diffusion_steps_per_forward):
            true_measure_noise, pred_measure_noise, true_dvector_noise, pred_dvector_noise = self.measures_diffuser(measure_diffuser_input, measures_diff, dvector_diff)
            loss_dict["measure_predictor_diff"] += nn.MSELoss()(pred_measure_noise, true_measure_noise) / self.diffusion_steps_per_forward
            loss_dict["dvector_predictor_diff"] += nn.MSELoss()(pred_dvector_noise, true_dvector_noise) / self.diffusion_steps_per_forward

        if inference or not tf:
            ### Measure & Dvector Sampling
            pred_measures_diff, pred_dvector_diff = self.measures_sampler(
                measure_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_measures = pred_measures + pred_measures_diff.transpose(1, 2)
            pred_dvector = pred_dvector + pred_dvector_diff
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(pred_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            for i, measures in enumerate(self.con.measures):
                measure_input = getattr(self, f"{measures}_embed")(
                    torch.bucketize(pred_measures[:, i], self.bins)
                )
                x = x + measure_input
        else:
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(true_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            for i, measures in enumerate(self.con.measures):
                measure_input = getattr(self, f"{measures}_embed")(
                    torch.bucketize(true_measures[:, i], self.bins)
                )
                x = x + measure_input

        ### Decoder
        x = self.positional_encoding(x)
        x, hidden = self.decoder(x)
        x = self.final(x)

        ### Consistency Loss
        synthetic_result = self.con(x, inference=True)
        cons_measures = synthetic_result["logits"]
        cons_dvector = synthetic_result["logits_dvector"]
        loss_dict["measure_consistency"] = nn.MSELoss()(cons_measures, true_measures)
        loss_dict["dvector_consistency"] = nn.MSELoss()(cons_dvector, true_dvector)

        ### Mel Loss
        x = x * mask
        norm_mel = norm_mel * mask
        norm_mel = norm_mel.transpose(1, 2)
        x = x.transpose(1, 2)
        loss_dict["mel"] = nn.MSELoss()(x, norm_mel)
        norm_mel = norm_mel.transpose(1, 2)
        x = x.transpose(1, 2)

        ### Mel Diffusion
        mel_diffuser_input = self.hidden_to_mel(hidden) + x
        mel_diffuser_input = self.positional_encoding_mel(mel_diffuser_input)
        if not inference:
            mel_diffuser_dvec = self.mel_dvector_in(true_dvector)
        else:
            mel_diffuser_dvec = self.mel_dvector_in(pred_dvector)
        mel_diffuser_input = mel_diffuser_input + mel_diffuser_dvec.unsqueeze(1)
        mel_diffuser_input = self.positional_encoding_diffuser(mel_diffuser_input)

        mel_diff = norm_mel - x
        loss_dict["mel_diff"] = 0.0
        for _ in range(self.diffusion_steps_per_forward):
            true_mel_noise, pred_mel_noise, _, _ = self.mel_diffuser(mel_diffuser_input, mel_diff)
            pred_mel_noise = pred_mel_noise * mask
            true_mel_noise = true_mel_noise * mask
            loss_dict["mel_diff"] += nn.MSELoss()(pred_mel_noise, true_mel_noise) / self.diffusion_steps_per_forward

        ### Mel Sampling
        if True:
            mel_add, _ = self.mel_sampler(mel_diffuser_input, lco["evaluation"]["num_steps"], batch_size)
            x = x + mel_add

        # denormalize mel
        x = self.con.scalers["mel"].inverse_transform(x.detach())
        x = x * mask

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])
        
        loss = sum(loss_dict.values()) / len(loss_dict) * mask_scale

        return {
            "loss": loss,
            "mel": x,
            "compound_losses": loss_dict,
            "mask": mask,
        }