from torch import nn
import lco

from .transformer import UpsamplingTransformer
from .conformer_layer import ConformerLayer

def swish(x):
    return x * torch.sigmoid(x)

# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

class StepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_t1 = nn.Linear(
            lco["diffusion_vocoder"]["step_embed_dim_in"],
            lco["diffusion_vocoder"]["step_embed_dim_hidden"]
        )
        self.fc_t2 = nn.Linear(
            lco["diffusion_vocoder"]["step_embed_dim_hidden"],
            lco["diffusion_vocoder"]["step_embed_dim_out"]
        )
        self.fc_out = nn.Linear(
            lco["diffusion_vocoder"]["step_embed_dim_out"],
            256
        )

    def forward(self, steps):
        half_dim = diffusion_step_embed_dim_in // 2
        _embed = np.log(10000) / (half_dim - 1)
        _embed = torch.exp(torch.arange(half_dim) * -_embed).to(x.device)
        _embed = diffusion_steps * _embed
        diff_embed = torch.cat(
            (torch.sin(_embed), torch.cos(_embed)),
            1
        )
        diff_embed = swish(self.fc_t1(diff_embed))
        diff_embed = swish(self.fc_t2(diff_embed))
        diff_embed = self.fc_out(diff_embed)
        return diff_embed
    

class DiffusionVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_embedding = StepEmbedding()
        self.init_conv = nn.Sequential(
            Conv(in_channels, res_channels, kernel_size=1),
            nn.ReLU(inplace=False)
        )
        self.upsampling_transformer = UpsamplingTransformer(
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
            upsampling_factor=256,
        )
    
    def forward(self, x, steps):
        diff_embed = self.step_embedding(steps)
        diff_embed = diff_embed.view(x.shape[0], 256, -1)
        x = self.init_conv(x)
        x = self.upsampling_transformer(x, steps)

        return x
    