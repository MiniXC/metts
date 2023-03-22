import lco
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .scaler import GaussianMinMaxScaler

class Transpose(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.transpose(1, 2)).transpose(1, 2)

class ConsistencyPredictorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ConformerConsistencyPredictorWithDVector(PreTrainedModel):
    config_class = ConsistencyPredictorConfig

    def __init__(self, config, layers=8):
        super().__init__(config)
        self.measures = lco["consistency"]["measures"]
        in_channels = lco["consistency"]["in_channels"]
        filter_size = lco["consistency"]["filter_size"]
        kernel_size = lco["consistency"]["kernel_size"]
        dropout = lco["consistency"]["dropout"]
        depthwise = lco["consistency"]["depthwise"]
        num_outputs = len(self.measures)

        self.loss_compounds = self.measures + ["dvector"]
        
        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=True,
            ),
            num_layers=lco["consistency"]["measure_nlayers"],
        )

        self.linear = nn.Sequential(
            nn.Linear(filter_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        dvector_dim = lco["consistency"]["dvector_dim"]
        
        self.dvector_layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=True,
            ),
            num_layers=lco["consistency"]["dvector_nlayers"],
        )

        dvector_input_dim = filter_size * 2
        
        self.dvector_linear = nn.Sequential(
            nn.Linear(dvector_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, dvector_dim),
        )

        self.scaler_dict = {
            k: GaussianMinMaxScaler(10) for k in self.measures
        }
        self.scaler_dict["mel"] = GaussianMinMaxScaler(10)
        self.scaler_dict["dvector"] = GaussianMinMaxScaler(10, sqrt=False)
        self.scaler_dict = nn.ModuleDict(self.scaler_dict)

        self.has_teacher = False

        self.apply(self._init_weights)

    def set_teacher(self, teacher):
        self._external_teacher = teacher
        self.scaler_dict = teacher.scaler_dict
        self.has_teacher = True
        # freeze teacher
        for param in self._external_teacher.parameters():
            param.requires_grad = False

    @property
    def scalers(self):
        return self.scaler_dict

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, mel, dvector=None, measures=None, inference=False):
        if self.scalers["mel"]._n <= 10_000_000:
            self.scalers["mel"].partial_fit(mel)
        x = self.scalers["mel"].transform(mel)
        x = x + torch.randn_like(x) * lco["consistency"]["noise_factor"]
        x = self.in_layer(x)
        x = self.positional_encoding(x)
        out_conv = self.layers(x)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        measure_true = {}
        loss = 0
        loss_dict = {}
        assert not (self.has_teacher and (measures is not None))
        assert (self.has_teacher or (measures is not None) or inference)
        ### teacher distillation
        if self.has_teacher:
            measures_loss = 0
            teacher_results = self._external_teacher(mel)["logits"]
            for i, measure in enumerate(self.measures):
                m_loss = nn.MSELoss()(out[:, i], teacher_results[:, i])
                measure_results[measure] = out[:, i]
                loss_dict[measure] = m_loss
                measures_loss = measures_loss + m_loss
            loss = loss + measures_loss / len(self.measures)
        ### measures (without teacher)
        if measures is not None:
            loss_dict = {}
            for i, measure in enumerate(self.measures):
                measure_out = out[:, i]
                if self.scalers[measure]._n <= 1_000_000:
                    self.scalers[measure].partial_fit(measures[measure])
                measure_results[measure] = measure_out #self.scalers[measure].transform(measure_out)
                measure_true[measure] = self.scalers[measure].transform(measures[measure])
            measures_loss = 0
            for measure in self.measures:
                m_loss = nn.MSELoss()(measure_results[measure], measure_true[measure])
                loss_dict[measure] = m_loss
                measures_loss += m_loss
            loss = measures_loss / len(self.measures)
            loss = loss + measures_loss / len(self.measures)
        ### d-vector
        # predict d-vector using global average and max pooling as input
        out_dvec = self.dvector_layers(x)
        dvector_input = torch.cat(
            [
                torch.mean(out_dvec, dim=1),
                torch.max(out_dvec, dim=1)[0],
            ],
            dim=1,
        )
        dvector_pred = self.dvector_linear(dvector_input)
        if dvector is not None:
            if self.scalers["dvector"]._n <= 1_000_000:
                self.scalers["dvector"].partial_fit(dvector)
            dvector_pred = dvector_pred # self.scalers["dvector"].transform(
            true_dvector = self.scalers["dvector"].transform(dvector)
            dvector_loss = nn.MSELoss()(dvector_pred, true_dvector)
            loss_dict["dvector"] = dvector_loss
            loss = loss + dvector_loss
        return {
            "loss": loss,
            "compound_losses": loss_dict,
            "logits": out,
            "logits_dvector": dvector_pred,
        }