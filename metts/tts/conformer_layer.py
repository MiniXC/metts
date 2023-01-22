class ConformerLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        if "conv_out" in kwargs:
            conv_out = kwargs["conv_out"]
        else:
            conv_out = kwargs["conv_in"]
        if "conv_depthwise" in kwargs and kwargs["conv_depthwise"]:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_in"],
                    kwargs["conv_in"],
                    kernel_size=kwargs["conv_kernel"][0],
                    padding="same",
                    groups=kwargs["conv_in"],
                ),
                nn.Conv1d(kwargs["conv_in"], kwargs["conv_filter_size"], 1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_filter_size"],
                    kwargs["conv_filter_size"],
                    kernel_size=kwargs["conv_kernel"][1],
                    padding="same",
                    groups=conv_out,
                ),
                nn.Conv1d(kwargs["conv_filter_size"], conv_out, 1),
            )
        else:
            self.conv1 = nn.Conv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding="same",
            )
            self.conv2 = nn.Conv1d(
                kwargs["conv_filter_size"],
                conv_out,
                kernel_size=kwargs["conv_kernel"][1],
                padding="same",
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)