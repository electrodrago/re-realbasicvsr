import torch.nn as nn
from mmengine.model import BaseModule

from mmagic.models.archs import ResidualBlockNoBN
from mmagic.models.utils import make_layer
from mmagic.registry import MODELS
from .With_REF import FE
from .LKA import LKA


@MODELS.register_module()
class NonRefNet(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels=64,
                 num_blocks=30,
                 res_scale=1.0):
        super().__init__()


        self.fe = FE(in_channels, mid_channels, num_blocks, res_scale)

        self.res_block_sum = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)
        
        self.conv_mid = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=3, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)

        self.aggregate_attn = LKA(mid_channels)

        self.shuffle_attn = LKA(mid_channels * 4)


    def forward(self, x):
        # Out: [b, 64, h, w]
        feat = self.fe(x)

        # Unshuffle then Shuffle
        feat_res = self.unshuffle(feat)                 # Out: [b, 256, h / 2, w / 2]
        feat_res = self.conv_mid(feat_res)
        feat_res = self.shuffle_attn(feat_res)  
        feat_res = self.shuffle(feat_res)               # Out: [b, 64, h, w]

        feat_res = self.res_block_sum(feat_res + feat)
        feat_res = self.aggregate_attn(feat_res)
        feat_res = self.conv_last(feat_res)

        return x + feat_res * 0.2
