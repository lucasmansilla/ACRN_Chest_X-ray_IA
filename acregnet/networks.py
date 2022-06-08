import torch
import torch.nn.functional as F

from .modules import Conv, ConvELUConv, UpConvELUConv, SpatialTransformer, UpConv


class AutoEncoder(torch.nn.Module):

    def __init__(self, input_size, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.down_size = tuple([d // (2 ** 3) for d in input_size])

        self.encoder_cnn = torch.nn.Sequential(
            Conv(in_channels, 16, stride=2),
            torch.nn.ReLU(),
            Conv(16, 16),
            torch.nn.ReLU(),
            Conv(16, 32, stride=2),
            torch.nn.ReLU(),
            Conv(32, 32),
            torch.nn.ReLU(),
            Conv(32, 1, stride=2),
            torch.nn.ReLU(),
        )

        self.encoder_linear = torch.nn.Linear(self.down_size[0]**2, 32)

        self.decoder_linear = torch.nn.Sequential(
            torch.nn.Linear(32, 1024),
            torch.nn.ReLU()
        )

        self.decoder_cnn = torch.nn.Sequential(
            UpConv(1, 32),
            torch.nn.ReLU(),
            Conv(32, 32),
            torch.nn.ReLU(),
            UpConv(32, 16),
            torch.nn.ReLU(),
            Conv(16, 16),
            torch.nn.ReLU(),
            UpConv(16, 16),
            torch.nn.ReLU(),
            Conv(16, out_channels, use_batchnorm=False),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_linear(x.view(-1, self.down_size[0]**2))
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.decoder_cnn(x.view(-1, 1, *self.down_size))
        return x


class RegNet(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()

        in_channels = len(input_size)

        self.vector_cnn = _UNet(in_channels)
        self.warper = SpatialTransformer(input_size)

    def forward(self, mov, fix):
        x = torch.cat([mov, fix], dim=1)
        flow = self.vector_cnn(x)
        out = self.warper(mov, flow)

        return out, flow


class _UNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.enc_blocks = torch.nn.ModuleList()
        prev_channels = in_channels
        for next_channels in [16, 32, 64, 128]:
            self.enc_blocks.append(
                ConvELUConv(prev_channels, next_channels)
            )
            prev_channels = next_channels

        self.dec_blocks = torch.nn.ModuleList()
        for next_channels in [64, 32, 16]:
            self.dec_blocks.append(
                UpConvELUConv(prev_channels, next_channels)
            )
            prev_channels = next_channels

        self.conv_out = Conv(16, out_channels)

    def forward(self, x):
        blocks = []
        for i, enc_block in enumerate(self.enc_blocks):
            x = enc_block(x)
            if i != len(self.enc_blocks) - 1:
                blocks.append(x)
                x = F.elu(x)
                x = F.avg_pool2d(x, 2)
        x = F.elu(x)

        for i, dec_block in enumerate(self.dec_blocks):
            x = F.dropout(x, training=self.training)
            x = dec_block(x, blocks[-i - 1])
            x = F.elu(x)

        x = self.conv_out(x)

        return x
