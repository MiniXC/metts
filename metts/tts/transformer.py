import copy

import torch
from torch import nn

# from https://pytorch.org/docs/1.13/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(N)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            if src_key_padding_mask is not None:
                _skpm_dtype = src_key_padding_mask.dtype
                if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                    raise AssertionError("only bool and floating types of key_padding_mask are supported")
            
            output = src
            first_layer = self.layers[0]
            src_key_padding_mask_for_layers = src_key_padding_mask

            for mod in self.layers:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)

            if self.norm is not None:
                output = self.norm(output)

            return output

class UpsamplingTransformer(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        upsampling_factor=256,
    ):
        super(UpsamplingTransformer, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(N)])
        self.num_layers = num_layers
        self.norm = norm
        self.upsampling_factor = upsampling_factor
        self.upsampling_per_layer = upsampling_factor ** (1 / num_layers)
        conv_in = encoder_layer.conv_in
        for i in range(num_layers):
            self.layers[i].conv_in = conv_in
            self.layers[i].conv_out = conv_in // self.upsampling_per_layer
            conv_in = conv_in // self.upsampling_per_layer

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            if src_key_padding_mask is not None:
                _skpm_dtype = src_key_padding_mask.dtype
                if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                    raise AssertionError("only bool and floating types of key_padding_mask are supported")
            
            output = src
            first_layer = self.layers[0]
            src_key_padding_mask_for_layers = src_key_padding_mask

            for mod in self.layers:
                # upsampling
                output = torch.nn.functional.interpolate(
                    output,
                    scale_factor=self.upsampling_per_layer,
                    mode="linear",
                    align_corners=False
                )
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)

            if self.norm is not None:
                output = self.norm(output)

            return output