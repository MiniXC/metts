import lco
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer

class Transpose(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.transpose(1, 2)).transpose(1, 2)

class ConsistencyPredictorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ConsistencyPredictor(PreTrainedModel):
    config_class = ConsistencyPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.measures = lco["consistency"]["measures"]
        nlayers = lco["consistency"]["nlayers"]
        in_channels = lco["consistency"]["in_channels"]
        filter_size = lco["consistency"]["filter_size"]
        kernel_size = lco["consistency"]["kernel_size"]
        dropout = lco["consistency"]["dropout"]
        depthwise = lco["consistency"]["depthwise"]
        num_outputs = len(self.measures)

        self.in_layer = nn.Linear(in_channels, filter_size)

        self.layers = nn.Sequential(
            *[
                VarianceConvolutionLayer(
                    filter_size, filter_size, kernel_size, dropout, depthwise
                )
                for _ in range(nlayers)
            ]
        )
        self.linear = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, num_outputs),
        )

    def forward(self, mel, measures=None):
        mel = self.in_layer(mel)
        out_conv = self.layers(mel)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        loss = None
        loss_dict = None
        if measures is not None:
            loss_dict = {}
            for i, measure in enumerate(self.measures):
                measure_results[measure] = out[:, i]
            measures_loss = 0
            for measure in self.measures:
                m_loss = nn.MSELoss()(measure_results[measure], measures[measure])
                loss_dict[measure] = m_loss
                measures_loss += m_loss
            loss = measures_loss / len(self.measures)
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "logits": out,
        }

class ConformerConsistencyPredictor(PreTrainedModel):
    config_class = ConsistencyPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.measures = lco["consistency"]["measures"]
        nlayers = lco["consistency"]["nlayers"]
        in_channels = lco["consistency"]["in_channels"]
        filter_size = lco["consistency"]["filter_size"]
        kernel_size = lco["consistency"]["kernel_size"]
        dropout = lco["consistency"]["dropout"]
        depthwise = lco["consistency"]["depthwise"]
        num_outputs = len(self.measures)

        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(3, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=True,
            ),
            num_layers=lco["consistency"]["transformer_layers"],
        )

        self.linear = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, num_outputs),
        )

    def forward(self, mel, measures=None):
        mel = self.in_layer(mel)
        mel = self.positional_encoding(mel)
        out_conv = self.layers(mel)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        loss = None
        loss_dict = None
        if measures is not None:
            loss_dict = {}
            for i, measure in enumerate(self.measures):
                measure_results[measure] = out[:, i]
            measures_loss = 0
            for measure in self.measures:
                m_loss = nn.MSELoss()(measure_results[measure], measures[measure])
                loss_dict[measure] = m_loss
                measures_loss += m_loss
            loss = measures_loss / len(self.measures)
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "logits": out,
        }

class ConformerConsistencyPredictorWithDVector(PreTrainedModel):
    config_class = ConsistencyPredictorConfig

    def __init__(self, config):
        super().__init__(config)
        self.measures = lco["consistency"]["measures"]
        nlayers = lco["consistency"]["nlayers"]
        in_channels = lco["consistency"]["in_channels"]
        filter_size = lco["consistency"]["filter_size"]
        kernel_size = lco["consistency"]["kernel_size"]
        dropout = lco["consistency"]["dropout"]
        depthwise = lco["consistency"]["depthwise"]
        num_outputs = len(self.measures)

        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(3, 1),
                batch_first=True,
                dropout=0.1,
                conv_depthwise=True,
            ),
            num_layers=2 # lco["consistency"]["transformer_layers"],
        )

        self.linear = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, num_outputs),
        )

        dvector_dim = lco["consistency"]["dvector_dim"]
        dvector_input_dim = filter_size * 2
        self.dvector = nn.Sequential(
            nn.Linear(dvector_input_dim, dvector_dim),
            nn.ReLU(),
            nn.Linear(dvector_dim, dvector_dim),
        )

    def forward(self, mel, dvector=None):
        x = self.in_layer(mel)
        x = self.positional_encoding(x)
        out_conv = self.layers(x)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        loss = None
        loss_dict = None
        if hasattr(self, "teacher"):
            loss_dict = {}
            measures_loss = 0
            teacher_results = self.teacher(mel)["logits"]
            for i, measure in enumerate(self.measures):
                measure_results[measure] = out[:, i]
                m_loss = nn.MSELoss()(measure_results[measure], teacher_results[:, i])
                loss_dict[measure] = m_loss
                measures_loss += m_loss
            loss = measures_loss / len(self.measures)
        ### d-vector
        # predict d-vector using global average and max pooling as input
        dvector_input = torch.cat(
            [
                torch.mean(out_conv, dim=1),
                torch.max(out_conv, dim=1)[0],
            ],
            dim=1,
        )
        dvector_pred = self.dvector(dvector_input)
        if dvector is not None:
            dvector_loss = nn.MSELoss()(dvector_pred, dvector)
            loss_dict["dvector"] = dvector_loss
            loss += dvector_loss
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "measures": out,
            "dvector": dvector_pred,
        }

class VarianceConvolutionLayer(nn.Module):
    def __init__(self, in_channels, filter_size, kernel_size, dropout, depthwise):
        super().__init__()
        if not depthwise:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Conv1d(
                        in_channels,
                        filter_size,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    )
                ),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
            )
        else:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                            groups=in_channels,
                        ),
                        nn.Conv1d(in_channels, filter_size, 1),
                    )
                ),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.layers(x)