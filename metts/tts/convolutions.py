from torch import nn

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Transpose(nn.Module):
    def __init__(self, module, dims=(1, 2)):
        super().__init__()
        self.module = module
        self.dims = dims

    def forward(self, x):
        return self.module(x.transpose(*self.dims)).transpose(*self.dims)

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, filter_size, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout=0.2, depthwise=False):
        super().__init__()
        if depthwise:
            ConvClass = DepthwiseConv1d
        else:
            ConvClass = nn.Conv1d
            
        self.convs = nn.Sequential(
            ConvClass(in_channels, filter_size, kernel_size, stride, padding, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            ConvClass(filter_size, out_channels, kernel_size, stride, padding, dilation),
        )

        self.activation = nn.ReLU()
        self.layer_norm = Transpose(nn.LayerNorm(out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.convs(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x