from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms import GaussianBlur


class Convx2(nn.Module):
    def __init__(self, c_in, c_out, ks, use_bias=False, padding=1, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=padding, padding_mode=padding_mode, bias=use_bias)
        self.conv1 = nn.Conv2d(c_in, c_out, ks, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, ks, **conv_args)
        if not use_bias:
            self.bn1 = nn.BatchNorm2d(c_out)
            self.bn2 = nn.BatchNorm2d(c_out)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# Create the global class for conv block
CB = Convx2


class TransformerUNet(nn.Module):
    def __init__(self, channels: Tuple[int],
        num_heads=2, 
        sigma=0.1,
        is_residual=False,
        bias=False,
        withSE_FLAG=False,
        smoothFeatures=False
        ) -> None:
        super(TransformerUNet, self).__init__()

        if withSE_FLAG:
          CB = WithSE(Convx2)
        else:
          CB = Convx2

        self.channels = channels
        self.pos_encoding = PositionalEncoding()
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual, bias) for i in range(len(channels) - 2)])
        self.smooth = nn.ModuleList([GaussianBlur(kernel_size=5, sigma=(sigma*(i + 1), sigma*(i + 1))) for i in range(len(channels) - 2)])
        self.chandropouts = nn.ModuleList([nn.FeatureAlphaDropout(p=0.03*(i+1)) for i in range(len(channels) - 2)])
        self.bottle_neck = Convx2(channels[-2], channels[-1], 3, use_bias=bias, padding=1)
        self.mhsa = MultiHeadSelfAttention(channels[-1], num_heads, bias)
        self.mhca = nn.ModuleList([MultiHeadCrossAttention(channels[i], num_heads, channels[i], channels[i + 1], bias) for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual, bias) for i in reversed(range(1, len(channels) - 1))])
        self.smoothFeatures = smoothFeatures

        #self.output = Convx2(channels[1], 8, 3, use_bias=bias, padding=1)
        self.init_weights()
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
          if self.smoothFeatures:
            x = self.smooth[i](x)
          x = self.chandropouts[i](x)
          skip_x, x = self.encode[i](x)
          skip_x_list.append(skip_x)

        x = self.bottle_neck(x)
        x = self.pos_encoding(x)
        x = self.mhsa(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
          x = self.pos_encoding(x)
          skip_x = self.pos_encoding(skip_x)
          skip_x = self.mhca[i](skip_x, x)
          x = self.decode[i](skip_x, x)

        #x = self.output(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True, dropout=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).view((b, h * w, c))
        x, _ = self.mha(x, x, x, need_weights=False)
        return x.view((b, h, w, c)).permute(0, 3, 1, 2)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            CB(channel_S, channel_S, 3, use_bias=bias, padding=1)
        )

        self.conv_Y = CB(channel_Y, channel_S, 3, use_bias=bias, padding=1)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True, dropout=0.15)

        self.upsample = nn.Sequential(
            CB(channel_S, channel_S, 3, use_bias=bias, padding=1),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).view((b, h * w, c))

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).view((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2)
        
        y = self.upsample(y)

        return torch.mul(y, s_enc)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x = x.view((b, c, h * w)) + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=torch.float32, device=DEVICE)
        depths = torch.arange(depth, dtype=torch.float32, device=DEVICE) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SqueezeExcitation(nn.Module):
    """
    adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    See: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = int(math.ceil(channels / reduction))
        self.squeeze = nn.Conv2d(channels, reduced, 1)
        self.excite = nn.Conv2d(reduced, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.max_pool2d(x, x.shape[2:])
        y = self.relu(self.squeeze(y))
        y = torch.sigmoid(self.excite(y))
        return x * y


def WithSE(conv_block, reduction=16):
    def make_block(c_in, c_out, ks, use_bias=False, padding=0):
        return nn.Sequential(
            conv_block(c_in, c_out, ks, use_bias=False, padding=0),
            SqueezeExcitation(c_out, reduction=reduction)
        )
    make_block.__name__ = f"WithSE({conv_block.__name__})"
    return make_block


class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_residual: bool = False, bias=False) -> None:
        super(EncoderLayer, self).__init__()

        self.conv = Convx2(in_channels, out_channels, 3, use_bias=bias, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool


class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_residual: bool = False, bias=False) -> None:
        super(DecoderLayer, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=bias)
        self.conv = Convx2(in_channels, out_channels, 3, use_bias=bias, padding=1)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat((skip_x, x), dim=1)
        return self.conv(x)