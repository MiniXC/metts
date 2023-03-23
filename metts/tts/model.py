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
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.positional_encoding_durations = PositionalEncoding(256)
        self.durations_in = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.durations_diffuser = DiffusionConformer(
            in_channels=256, # hidden dim + predicted duration
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
        self.measures_dvector_in = nn.Linear(256, 256)
        self.positional_encoding_measures = PositionalEncoding(256)
        self.measures_in = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.measures_diffuser = DiffusionConformer(
            in_channels=256, # hidden dim + 4 measures
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
        self.mel_dvector_in = nn.Linear(256, 256)
        self.positional_encoding_mel = PositionalEncoding(256)
        self.mel_in = nn.Linear(80, 256)
        self.mel_diffuser = DiffusionConformer(
            in_channels=256, # hidden dim + predicted mel
            frame_level_outputs=80,
            sequence_level_outputs=0,
            layers=4,
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

    def forward(self, phones, phone_durations, durations, mel, val_ind, speaker, inference=False):
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
        duration_diffuser_input = x + self.durations_in(pred_durations_disc.unsqueeze(-1))
        duration_diffuser_input = self.positional_encoding_durations(duration_diffuser_input)

        norm_durations = norm_durations.to(duration_diffuser_input.dtype)
        duration_diff = norm_durations.unsqueeze(-1) - pred_durations_disc.unsqueeze(-1)
        print("duration diff", duration_diffuser_input.shape, duration_diff.shape)
        loss_dict["duration_predictor_diff"] = 0.0
        for _ in range(self.diffusion_steps_per_forward):
            true_duration_noise, pred_duration_noise, _, _ = self.durations_diffuser(duration_diffuser_input, duration_diff)
            loss_dict["duration_predictor_diff"] += nn.MSELoss()(true_duration_noise, pred_duration_noise) / self.diffusion_steps_per_forward

        ### Length Regulator
        if inference:
            pred_duration_diff, _ = self.durations_sampler(
                duration_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_durations = pred_durations_disc.squeeze(-1) #+ pred_duration_diff.squeeze(-1)
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

        ### Measure Prediction
        pred_measures = self.measure_transformer(x)
        pred_measures = self.measures_linear(pred_measures)
        pred_measures = pred_measures.transpose(1, 2)
        # max pool and avg pool for dvector        
        pred_dvector_disc = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        pred_dvector_disc = self.measures_dvector(pred_dvector_disc)
        #### Measure Loss
        pred_measures_disc = pred_measures * mask.transpose(1, 2)
        true_measures_disc = true_measures * mask.transpose(1, 2)
        loss_dict["measure_predictor"] = nn.MSELoss()(pred_measures_disc, true_measures)
        loss_dict["dvector_predictor"] = nn.MSELoss()(pred_dvector_disc, true_dvector)
        ### Measure Diffusion
        measure_diffuser_input = x + self.measures_in(pred_measures_disc.transpose(1, 2))
        measure_diffuser_dvec = self.measures_dvector_in(pred_dvector_disc)
        measure_diffuser_input = measure_diffuser_input + measure_diffuser_dvec.unsqueeze(1)
        measure_diffuser_input = self.positional_encoding_measures(measure_diffuser_input)
        measures_diff = true_measures.transpose(1, 2) - pred_measures_disc.transpose(1, 2)
        print("measures diff", measure_diffuser_input.shape, measures_diff.shape)
        dvector_diff = true_dvector - pred_dvector_disc
        loss_dict["measure_predictor_diff"] = 0.0
        loss_dict["dvector_predictor_diff"] = 0.0

        for _ in range(self.diffusion_steps_per_forward):
            true_measure_noise, pred_measure_noise, true_dvector_noise, pred_dvector_noise = self.measures_diffuser(measure_diffuser_input, measures_diff, dvector_diff)
            loss_dict["measure_predictor_diff"] += nn.MSELoss()(pred_measure_noise, true_measure_noise) / self.diffusion_steps_per_forward
            loss_dict["dvector_predictor_diff"] += nn.MSELoss()(pred_dvector_noise, true_dvector_noise) / self.diffusion_steps_per_forward

        if inference:
            ### Measure & Dvector Sampling
            pred_measures_diff, pred_dvector_diff = self.measures_sampler(
                measure_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_measures = pred_measures_disc #+ pred_measures_diff.transpose(1, 2)
            pred_dvector = pred_dvector_disc #+ pred_dvector_diff

            print("measures diff pred", pred_measures_diff.min(), pred_measures_diff.max(), pred_measures_diff.mean(), pred_measures_diff.std())
            print("measures diff true", measures_diff.min(), measures_diff.max(), measures_diff.mean(), measures_diff.std())

            print("dvector diff pred", pred_dvector_diff.min(), pred_dvector_diff.max(), pred_dvector_diff.mean(), pred_dvector_diff.std())
            print("dvector diff true", dvector_diff.min(), dvector_diff.max(), dvector_diff.mean(), dvector_diff.std())

            for i, measure in enumerate(self.con.measures):
                print()
                print("Measure", measure)
                w_diff = wasserstein_distance(pred_measures[:, i].flatten().detach().numpy(), true_measures[:, i].flatten().detach().numpy())
                w_disc = wasserstein_distance(pred_measures_disc[:, i].flatten().detach().numpy(), true_measures[:, i].flatten().detach().numpy())
                print(f"Wasserstein distance diff {w_diff:.3f}")
                print(f"Wasserstein distance disc {w_disc:.3f}")
                mse_diff = nn.MSELoss()(pred_measures[:, i], true_measures[:, i])
                mse_disc = nn.MSELoss()(pred_measures_disc[:, i], true_measures[:, i])
                print(f"MSE diff {mse_diff:.3f}")
                print(f"MSE disc {mse_disc:.3f}")
                print()

            #raise

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
        cons_dvector = synthetic_result["logits_dvector"]
        loss_dict["measure_consistency"] = nn.MSELoss()(cons_measures, true_measures)
        loss_dict["dvector_consistency"] = nn.MSELoss()(cons_dvector, true_dvector)

        ### Mel Loss
        pred_mel_disc = pred_mel_disc * mask
        norm_mel = norm_mel * mask
        loss_dict["mel"] = nn.MSELoss()(pred_mel_disc, norm_mel)

        ### Mel Diffusion
        mel_diffuser_input = self.mel_in(pred_mel_disc) + hidden
        if inference:
            mel_diffuser_dvec = self.mel_dvector_in(pred_dvector)
        else:
            mel_diffuser_dvec = self.mel_dvector_in(true_dvector)
        mel_diffuser_input = mel_diffuser_input + mel_diffuser_dvec.unsqueeze(1)
        mel_diffuser_input = self.positional_encoding_mel(mel_diffuser_input)

        mel_diff = norm_mel - pred_mel_disc
        loss_dict["mel_diff"] = 0.0
        for _ in range(self.diffusion_steps_per_forward):
            true_mel_noise, pred_mel_noise, _, _ = self.mel_diffuser(mel_diffuser_input, mel_diff)
            pred_mel_noise = pred_mel_noise * mask
            true_mel_noise = true_mel_noise * mask
            loss_dict["mel_diff"] += nn.MSELoss()(pred_mel_noise, true_mel_noise) / self.diffusion_steps_per_forward

        ### Mel Sampling
        if True:
            print("mel diff input", mel_diffuser_input.shape, mel_diff.shape)
            pred_mel_diff, _ = self.mel_sampler(mel_diffuser_input, lco["evaluation"]["num_steps"], batch_size)
            print("mel diff pred", pred_mel_diff.min(), pred_mel_diff.max(), pred_mel_diff.mean(), pred_mel_diff.std())
            print("mel diff true", mel_diff.min(), mel_diff.max(), mel_diff.mean(), mel_diff.std())
            print(pred_mel_diff.shape, pred_mel_disc.shape)
            pred_mel = pred_mel_disc #+ pred_mel_diff
            print("mel wasserstein diff", wasserstein_distance(pred_mel.flatten().detach().numpy(), norm_mel.flatten().detach().numpy()))
            print("mel wasserstein disc", wasserstein_distance(pred_mel_disc.flatten().detach().numpy(), norm_mel.flatten().detach().numpy()))
            mel_mse_diff = nn.MSELoss()(pred_mel, norm_mel)
            mel_mse_disc = nn.MSELoss()(pred_mel_disc, norm_mel)
            print("mel mse diff", mel_mse_diff)
            print("mel mse disc", mel_mse_disc)
        else:
            pred_mel = pred_mel_disc

        results = {
            "compound_losses": loss_dict,
            "mask": mask,
        }

        # denormalize mel
        if inference:
            x = self.con.scalers["mel"].inverse_transform(pred_mel)
            x = x * mask
            results["mel"] = x

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])
        
        loss = sum(loss_dict.values()) / len(loss_dict) * mask_scale

        results["loss"] = loss

        return results