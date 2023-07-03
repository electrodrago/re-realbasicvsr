import torch
import torch.nn as nn
from mmagic.models.archs import ResidualBlockNoBN
from mmagic.models.utils import make_layer
from mmengine.model import BaseModule
from mmagic.registry import MODELS
from .LKA import LKA

@MODELS.register_module()
class SemanticGuideEnhance(BaseModule):
    def __init__(self, mid_channels, num_blocks):
        super(SemanticGuideEnhance, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        self.sem_fusion = SemanticFusionUnit(mid_channels)

        self.sem_aggr = LKA(mid_channels)
        
        self.blocks = GrowthAttentionResidualBlock(mid_channels, mid_channels // 2, num_blocks)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=3, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x, sem):
        feat = self.in_conv(x)
        
        feat_res = feat + self.sem_aggr(self.sem_fusion(feat, sem))
        feat_res = self.blocks(feat_res)
        feat_res = self.out_conv(feat_res)

        return x + feat_res
    

class SemanticFusionUnit(BaseModule):
    def __init__(self, channels):
        super(SemanticFusionUnit, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels + 1, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, feat, sem):
        cat = torch.cat((feat, sem), dim = 1) # (b, c, h, w)
        fusion = self.conv(cat)
        return fusion
    

class GrowthAttentionResidualBlock(BaseModule):
    """Attention Residual Dense Block.

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, mid_channels=64, growth_channels=32, num_blocks=10):
        super().__init__()
        
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            # Growth block convolution
            self.add_module(
                f'block{i+1}',
                ResidualBlocksWithInputConv(mid_channels + i * growth_channels, out_channels, num_blocks))
            # Growth attention
            self.add_module(
                f'attn{i+1}',
                LKA(out_channels))
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.attn1(self.block1(x)))
        x2 = self.lrelu(self.attn2(self.block2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.attn3(self.block3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.attn4(self.block4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.attn5(self.block5(torch.cat((x, x1, x2, x3, x4), 1)))

        return x5 * 0.2 + x
    

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
