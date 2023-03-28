import multiprocessing
from dataclasses import dataclass

import lco
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig

from nnAudio.features.mel import MelSpectrogram

from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .length_regulator import LengthRegulator
# from .diffusion_vocoder import DiffWave, DiffWaveSampler
from .refinement_models import DiffusionConformer, DiffusionSampler, DiffusionLinear
from .scaler import GaussianMinMaxScaler

# wasserstein distance
from scipy.stats import wasserstein_distance

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

        self._n = 0

        self.diffusion_only = lco["training"]["diffusion_only"]

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
            num_layers=6,
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
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.durations_in = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.durations_diffuser = DiffusionConformer(
            in_channels=256,
            frame_level_outputs=1,
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
                conv_kernel=(3, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=lco["conformer"]["depthwise"],
            ),
            num_layers=2,
        )
        self.measures_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        ) # 4 measures
        self.dvector_scaler = GaussianMinMaxScaler(10)
        self.measures_dvector = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.measures_dvector_in = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.measures_in = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.measures_diffuser = DiffusionConformer(
            in_channels=256,
            frame_level_outputs=4,
        )
        self.dvector_diffuser = DiffusionLinear(
            in_dim=256,
            out_dim=256,
            hidden_dim=1024,
            layers=8,
        )
        self.measures_dvector_diff = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.measures_sampler = DiffusionSampler(
            self.measures_diffuser,
        )
        self.dvector_sampler = DiffusionSampler(
            self.dvector_diffuser,
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
            nn.GELU(),
            nn.Linear(256, 80),
        )
        self.mel_dvector_in = nn.Linear(256, 256)
        self.mel_in = nn.Linear(80, 256)
        self.mel_diffuser_hidden_in = nn.Linear(256, 256)
        self.mel_diffuser = DiffusionConformer(
            in_channels=256, # hidden dim + predicted mel
            frame_level_outputs=80,
            layers=4,
            hidden_dim=512,
        )
        self.mel_sampler = DiffusionSampler(
            self.mel_diffuser,
        )

        self.diffusion_steps_per_forward = lco["diffusion"]["steps_per_forward"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, phones, phone_durations, durations, mel, val_ind, speaker, inference=False, return_loss=True, return_mel=False):
        loss_dict = {}

        self._n += 1

        norm_mel = self.con.scalers["mel"].transform(mel)

        if inference:
            steps_per_forward = 1
        else:
            steps_per_forward = self.diffusion_steps_per_forward

        if self.duration_scaler._n <= 1_000:
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
        if not self.diffusion_only:
            loss_dict["duration_predictor"] = nn.MSELoss()(pred_durations_disc, norm_durations)

        #### Duration Diffusion
        duration_diffuser_input = x + self.durations_in(pred_durations_disc.unsqueeze(-1))
        duration_diffuser_input = F.gelu(duration_diffuser_input)

        norm_durations = norm_durations.to(duration_diffuser_input.dtype)
        duration_diff = norm_durations.unsqueeze(-1)
        if not inference:
            loss_dict["duration_predictor_diff"] = 0.0
            for _ in range(steps_per_forward):
                true_duration_noise, pred_duration_noise, _, _ = self.durations_diffuser(duration_diffuser_input, duration_diff)
                loss_dict["duration_predictor_diff"] += nn.MSELoss()(true_duration_noise, pred_duration_noise) / self.diffusion_steps_per_forward

        ### Length Regulator
        if inference:
            pred_duration_diff, _ = self.durations_sampler(
                duration_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_durations = pred_duration_diff.squeeze(-1)
            pred_durations = self.duration_scaler.inverse_transform(pred_durations)
            pred_durations = torch.round(pred_durations).long()

            if (pred_durations < 0).any():
                print("negative duration, setting to 0")
                pred_durations[pred_durations < 0] = 0
            pred_durations[(phones == 0) | (phones == 46)] = 0

            x, mask = self.lr(x, pred_durations)
        else:
            x, mask = self.lr(x, phone_durations, val_ind)
            if x.shape[1] > norm_mel.shape[1]:
                x = x[:, :norm_mel.shape[1]]
        ### Consistency
        consistency_result = self.con(norm_mel, inference=True)
        true_measures = consistency_result["logits"]
        true_dvector = consistency_result["logits_dvector"]

        if self.dvector_scaler._n <= 1_000:
            self.dvector_scaler.partial_fit(true_dvector)
        true_dvector = self.dvector_scaler.transform(true_dvector)

        ### Measure Prediction
        pred_measures = self.measure_transformer(x + speaker.unsqueeze(1))
        pred_measures = self.measures_linear(pred_measures)
        pred_measures = pred_measures.transpose(1, 2)
        # max pool and avg pool for dvector        
        pred_dvector_disc = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        pred_dvector_disc = self.measures_dvector(pred_dvector_disc)
        #### Measure Loss
        pred_measures_disc = pred_measures * mask.transpose(1, 2)
        true_measures_disc = true_measures * mask.transpose(1, 2)
        if not self.diffusion_only:
            loss_dict["measure_predictor"] = nn.MSELoss()(pred_measures_disc, true_measures_disc)
            loss_dict["dvector_predictor"] = nn.MSELoss()(pred_dvector_disc, true_dvector)

        ### Measure Diffusion
        measure_diffuser_input = (
            x +
            self.measures_in(pred_measures_disc.transpose(1, 2)) +
            speaker.unsqueeze(1)
        )
        measure_diffuser_input = F.gelu(measure_diffuser_input)
        measures_diff = true_measures.transpose(1, 2) * mask

        ### Dvector Diffusion
        dvector_diffuser_input = (
            # self.measures_dvector_in(pred_dvector_disc) +
            # speaker +
            pred_dvector_disc
        )
        #dvector_diffuser_input = F.gelu(dvector_diffuser_input)
        dvector_diff = true_dvector

        loss_dict["measure_predictor_diff"] = 0.0
        loss_dict["dvector_predictor_diff"] = 0.0
        if not inference:
            for _ in range(steps_per_forward):
                true_measure_noise, pred_measure_noise, _, _ = self.measures_diffuser(measure_diffuser_input, measures_diff)
                # multiply with mask
                true_measure_noise = true_measure_noise * mask
                pred_measure_noise = pred_measure_noise * mask
                loss_dict["measure_predictor_diff"] += nn.MSELoss()(pred_measure_noise, true_measure_noise) / self.diffusion_steps_per_forward
                true_dvector_noise, pred_dvector_noise, _, _ = self.dvector_diffuser(dvector_diffuser_input, dvector_diff)
                loss_dict["dvector_predictor_diff"] += nn.MSELoss()(pred_dvector_noise.squeeze(1), true_dvector_noise) / self.diffusion_steps_per_forward

        if inference:
            ### Measure & Dvector Sampling
            pred_measures_diff, _ = self.measures_sampler(
                measure_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_measures_diff = pred_measures_diff.transpose(1, 2)     

            pred_measures = pred_measures_diff * mask.transpose(1, 2)

            pred_dvector_diff, _ = self.dvector_sampler(
                dvector_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size,
                single_element=True,
            )
            pred_dvector = pred_dvector_diff.squeeze(1)

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
        pred_mel_disc = self.final(x)

        ### Consistency Loss
        synthetic_result = self.con(pred_mel_disc, inference=True)
        cons_measures = synthetic_result["logits"]
        cons_dvector = self.dvector_scaler.transform(synthetic_result["logits_dvector"])
        if not self.diffusion_only:
            loss_dict["measure_consistency"] = nn.MSELoss()(cons_measures, true_measures)
            loss_dict["dvector_consistency"] = nn.MSELoss()(cons_dvector, true_dvector)

        ### Mel Loss
        pred_mel_disc = pred_mel_disc * mask
        norm_mel = norm_mel * mask
        if not self.diffusion_only:
            loss_dict["mel"] = nn.MSELoss()(pred_mel_disc, norm_mel)

        ### Mel Diffusion
        mel_diffuser_input = self.mel_in(pred_mel_disc) + self.mel_diffuser_hidden_in(hidden)
        if inference:
            mel_diffuser_dvec = self.mel_dvector_in(pred_dvector)
        else:
            mel_diffuser_dvec = self.mel_dvector_in(true_dvector)
        mel_diffuser_input = mel_diffuser_input + mel_diffuser_dvec.unsqueeze(1)
        mel_diffuser_input = F.gelu(mel_diffuser_input)

        mel_diff = norm_mel
        loss_dict["mel_diff"] = 0.0
        if not inference:
            for _ in range(steps_per_forward):
                true_mel_noise, pred_mel_noise, _, _ = self.mel_diffuser(mel_diffuser_input, mel_diff)
                pred_mel_noise = pred_mel_noise * mask
                true_mel_noise = true_mel_noise * mask
                loss_dict["mel_diff"] += nn.MSELoss()(pred_mel_noise, true_mel_noise) / self.diffusion_steps_per_forward

        ### Mel Sampling
        if inference:
            pred_mel_diff, _ = self.mel_sampler(mel_diffuser_input, lco["evaluation"]["num_steps"], batch_size)
            pred_mel = pred_mel_diff
        else:
            pred_mel = pred_mel_disc

        results = {
            "compound_losses": loss_dict,
            "mask": mask,
        }
        

        if self._n % 100 == 0:
            print({
                k: v.item() for k, v in loss_dict.items()
            })

        # denormalize mel
        #if inference:
        
        if inference or return_mel:
            x = self.con.scalers["mel"].inverse_transform(pred_mel)
            x = x * mask
            results["mel"] = x

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])
        
        loss = sum(loss_dict.values()) / len(loss_dict) * mask_scale

        if return_loss:
            results["loss"] = loss

        return results