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
            "measures_predictor",
            "measures_predictor_diff",
            "measures_consistency",
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
                conv_depthwise=True,
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
                conv_depthwise=True,
            ),
            num_layers=2,
        )
        self.durations_linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
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
                conv_filter_size=256,
                conv_kernel=(9, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=True,
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
                conv_depthwise=True,
            ),
            num_layers=6,
            return_additional_layer=2,
        )
        self.final = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 80),
        )
        self.mel_diffuser = DiffusionConformer(
            in_channels=256 + 80, # hidden dim + predicted mel
            frame_level_outputs=80,
            sequence_level_outputs=0,
        )
        self.mel_sampler = DiffusionSampler(
            self.mel_diffuser,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, phones, phone_durations, durations, mel, val_ind, speaker, inference=False, force_tf=False):
        tf = True # force_tf or (torch.rand(1).item() < 0.8)

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
        norm_durations = norm_durations.to(duration_diffuser_input.dtype)
        true_duration_noise, pred_duration_noise, _, _ = self.durations_diffuser(duration_diffuser_input, norm_durations.unsqueeze(-1))
        loss_dict["duration_predictor_diff"] = nn.MSELoss()(pred_duration_noise, true_duration_noise)

        ### Length Regulator
        if not inference:
            x, mask = self.lr(x, phone_durations, val_ind)
            if x.shape[1] > norm_mel.shape[1]:
                x = x[:, :norm_mel.shape[1]]
        else:
            pred_durations, _ = self.durations_sampler(
                duration_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_durations = pred_durations.squeeze(-1)
            pred_durations = self.duration_scaler.inverse_transform(pred_durations) * 4.7797 + 4.9389
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
        pred_measures = pred_measures * mask
        true_measures = true_measures * mask
        loss_dict["measure_predictor"] = nn.MSELoss()(pred_measures, true_measures)
        loss_dict["dvector_predictor"] = nn.MSELoss()(pred_dvector, true_dvector)
        ### Measure Diffusion
        measure_diffuser_input = torch.cat([x, pred_measures.transpose(1, 2)], dim=-1)
        measure_diffuser_dvec = self.measures_dvector_in(pred_dvector)
        measure_diffuser_input = measure_diffuser_input + measure_diffuser_dvec.unsqueeze(1)
        true_measures_diff = true_measures.transpose(1, 2)
        true_measure_noise, pred_measure_noise, true_dvector_noise, pred_dvector_noise = self.measures_diffuser(measure_diffuser_input, true_measures_diff, true_dvector)
        pred_measure_noise = pred_measure_noise * mask
        true_measure_noise = true_measure_noise * mask
        loss_dict["measure_predictor_diff"] = nn.MSELoss()(pred_measure_noise, true_measure_noise)
        loss_dict["dvector_predictor_diff"] = nn.MSELoss()(pred_dvector_noise, true_dvector_noise)

        if inference or not tf:
            ### Measure & Dvector Sampling
            pred_measures, pred_dvector = self.measures_sampler(
                measure_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(true_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            for i, measures in enumerate(self.con.measures):
                measure_input = getattr(self, f"{measures}_embed")(
                    torch.bucketize(pred_measures.transpose(1, 2)[:, i], self.bins)
                )
                x = x + measure_input
        else:
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(pred_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            for i, measures in enumerate(self.con.measures):
                measure_input = getattr(self, f"{measures}_embed")(
                    torch.bucketize(pred_measures[:, i], self.bins)
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
        mel_diffuser_input = torch.cat([hidden, x], dim=-1)
        mel_diffuser_input = mel_diffuser_input + measure_diffuser_dvec.unsqueeze(1)

        true_mel_noise, pred_mel_noise, _, _ = self.mel_diffuser(mel_diffuser_input, norm_mel)
        pred_mel_noise = pred_mel_noise * mask
        true_mel_noise = true_mel_noise * mask
        loss_dict["mel_diff"] = nn.MSELoss()(pred_mel_noise, true_mel_noise)

        ### Mel Samplingq
        if inference:
            # 200
            x, _ = self.mel_sampler(mel_diffuser_input, 200, batch_size)

        # denormalize mel
        x = self.con.scalers["mel"].inverse_transform(x)
        x = x * mask

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])
        
        loss = sum(loss_dict.values()) / len(loss_dict) * mask_scale

        return {
            "loss": loss,
            "mel": x,
            "compound_losses": loss_dict,
            "mask": mask,
        }