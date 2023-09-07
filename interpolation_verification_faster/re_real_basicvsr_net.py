from logging import WARNING

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import numbers
from einops import rearrange

from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import flow_warp, make_layer
from mmagic.registry import MODELS
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init


class ResidualBlocksWithInputConv(BaseModule):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class LargeKernelAttn(nn.Module):
    def __init__(self,
                 channels):
        super(LargeKernelAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        return x * weight

class FuseUpsampling(nn.Module):
    def __init__(self, channels=6):
        super(FuseUpsampling, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, (1, 1))
        self.conv2 = nn.Conv2d(6, 3, (1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fuse_attn = LargeKernelAttn(channels)

        self.nearest_x2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.linear_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.linear_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.nearest_x4 = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, lq_clean, lq_raw):
        # Case: nearest x4 | bilinear x4 | nearest x2, bilinear x2 | bilinear x2, nearest x2

        # # Case here: nearest x2, bilinear x2
        # hr_clean = self.nearest_x2(lq_clean)
        # hr_raw = self.nearest_x2(lq_raw)

        # hr_clean = self.linear_x2(hr_clean)
        # hr_raw = self.linear_x2(hr_raw)

        # Case here: nearest x4
        hr_clean = self.nearest_x4(lq_clean)
        hr_raw = self.nearest_x4(lq_raw)

        # # Case here: bilinear x4
        # hr_clean = self.linear_x4(lq_clean)
        # hr_raw = self.linear_x4(lq_raw)

        hr_raw = self.conv1(hr_raw)

        hr_clean = self.fuse_attn(torch.cat([hr_clean, hr_raw], dim=1))
        hr_clean = self.lrelu(self.conv2(hr_clean))

        return self.linear_x4(lq_clean) + hr_clean


@MODELS.register_module()
class Re_RealBasicVSRNet(BaseModule):
    def __init__(self,
                 mid_channels=64):
        super().__init__()

        self.mid_channels = mid_channels

        # img cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, 20),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )
        self.image_cleaning.load_state_dict(torch.load("/content/re-realbasicvsr/clean.pth", map_location=lambda storage, loc: storage))

        self.image_cleaning.requires_grad_(False)

        self.fuse_upsampling = FuseUpsampling(3, 3, num_heads=3)

    def upsample(self, lqs_clean, lqs_raw):
        outputs = []

        for i in range(0, lqs_clean.size(1)):
            hr = self.fuse_upsampling(lqs_clean[:, i, :, :, :], lqs_raw[:, i, :, :, :])
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor, return_lqs=False):
        n, t, c, h, w = lqs.size()

        lqs_clean = lqs.detach().clone()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            lqs_clean = lqs_clean.view(-1, c, h, w)
            residues = self.image_cleaning(lqs_clean)
            lqs_clean = (lqs_clean + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < 1.0:
                break


        if return_lqs:
            return self.upsample(lqs_clean, lqs), lqs_clean
        else:
            return self.upsample(lqs_clean, lqs)
