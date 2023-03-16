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

num_cpus = multiprocessing.cpu_count()

class MeTTSConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MeTTS(PreTrainedModel):
    config_class = MeTTSConfig

    def __init__(self, config):
        super().__init__(config)

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


class FastSpeechWithConsistency(PreTrainedModel):
    config_class = MeTTSConfig

    def __init__(self, config, consistency_net):
        super().__init__(config)

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

        # measures
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
        
        self.dvector_to_encoder = nn.Linear(256, 256)
        self.measures_to_encoder = nn.Linear(4, 256)

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
            num_layers=6,
            return_additional_layer=2,
        )
        self.final = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 80),
        )

        self.con = consistency_net

        for param in self.con.parameters():
            param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, phones, phone_durations, durations, mel, val_ind, speaker, inference=False, force_tf=False):
        tf = True # force_tf or (torch.rand(1).item() < 0.8)

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
        duration_loss_disc = nn.L1Loss()(pred_durations_disc, durations)

        #### Duration Diffusion
        duration_diffuser_input = torch.cat([x, pred_durations_disc.unsqueeze(-1) / 0.42], dim=-1)
        # todo: test without detaching
        true_duration_noise, pred_duration_noise, _ = self.durations_diffuser(duration_diffuser_input, pred_durations_disc.unsqueeze(-1))
        duration_loss_gen = nn.MSELoss()(pred_duration_noise, true_duration_noise)
        if inference:
            pred_durations, _ = self.durations_sampler(
                duration_diffuser_input,
                lco["evaluation"]["num_steps"],
                batch_size
            )
            pred_durations = pred_durations.squeeze(-1)

        duration_loss = duration_loss_disc + duration_loss_gen

        ### Length Regulator
        if not inference:
            x, mask = self.lr(x, phone_durations, val_ind)
            if x.shape[1] > mel.shape[1]:
                x = x[:, :mel.shape[1]]
        else:
            # denoramalize durations
            # "mean": 4.9033311291969826,
            # "std": 4.793017975164098
            print()
            print((pred_durations_disc / 0.42).mean(), (pred_durations_disc / 0.42).std())
            print()
            pred_durations = pred_durations * 4.793017975164098 + 4.9033311291969826
            # round
            pred_durations = torch.round(pred_durations).long()
            if (pred_durations < 0).any():
                print("negative duration, setting to 0")
                pred_durations[pred_durations < 0] = 0
            pred_durations[phones == 0] = 0
            x, mask = self.lr(x, pred_durations)

        mask_scale = mask.sum() / (mask.shape[0] * mask.shape[1])

        ### Measure Prediction
        pred_measures = self.measure_transformer(x)
        pred_measures = self.measures_linear(pred_measures)
        pred_measures = pred_measures.transpose(1, 2)
        # max pool and avg pool for dvector
        pred_dvector = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        pred_dvector = self.measures_dvector(pred_dvector)
        consistency_result = self.con(mel)
        true_measures = consistency_result["logits"]
        true_dvector = consistency_result["dvector"]
        #### Measure Loss
        measure_loss = nn.MSELoss()(pred_measures, true_measures) / 4
        measure_loss = measure_loss * mask_scale
        #### Dvector Loss
        dvector_loss = nn.MSELoss()(pred_dvector, true_dvector)

        if inference or not tf:
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(true_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            measures_input = self.measures_to_encoder(pred_measures.transpose(1, 2))
            x = x + measures_input
        else:
            ### Add Dvector to Decoder
            dvector_input = self.dvector_to_encoder(pred_dvector)
            x = x + dvector_input.unsqueeze(1)
            ### Add Measures to Decoder
            measures_input = self.measures_to_encoder(true_measures.transpose(1, 2))
            x = x + measures_input

        ### Decoder
        x = self.positional_encoding(x)
        x, hidden = self.decoder(x)
        x = self.final(x)

        ### Consistency Loss
        cons_measures = self.con(mel)["logits"]
        cons_dvector = self.con(x)["dvector"]
        if inference or not tf:
            consistency_loss = nn.MSELoss()(cons_measures, pred_measures)
            consistency_loss = consistency_loss + nn.MSELoss()(cons_dvector, pred_dvector)
        else:
            consistency_loss = nn.MSELoss()(cons_measures, true_measures)
            consistency_loss = consistency_loss + nn.MSELoss()(cons_dvector, true_dvector)

        consistency_loss = consistency_loss * mask_scale

        ### Mel Loss
        x = x * mask
        mel = mel * mask
        mel = mel.transpose(1, 2)
        x = x.transpose(1, 2)
        mel_loss = nn.L1Loss()(x, mel)

        # denormalize mel
        x = x * 1.0958075523376465 + -0.19927863776683807
        
        loss = (mel_loss + duration_loss + measure_loss + consistency_loss + dvector_loss) / 5

        return {
            "loss": loss,
            "mel": x,
            "loss_dict": {
                "mel_loss": mel_loss,
                "duration_loss": duration_loss,
                "measure_loss": measure_loss,
                "consistency_loss": consistency_loss,
                "dvector_loss": dvector_loss,
            },
            "mask": mask,
        }