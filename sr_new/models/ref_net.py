import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mmengine.model import BaseModule

from mmagic.models.archs import ResidualBlockNoBN
from mmagic.models.archs import ImgNormalize
from mmagic.models.utils import make_layer
from mmagic.registry import MODELS


@MODELS.register_module()
class RefNet(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels=64,
                 texture_channels=64,
                 num_blocks=30,
                 res_scale=1.0):
        super().__init__()

        self.texture_channels = texture_channels

        self.fe = FE(in_channels, mid_channels, num_blocks // 2, res_scale)

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels + texture_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.res_block = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=3, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.LTE = LTE(requires_grad=True)

        self.search_attn = SearchAttention()


    def forward(self, x, ref_down_up, ref):
        x = self.LTE(x)
        ref_down_up = self.LTE(ref_down_up)
        ref = self.LTE(ref)

        soft_attention, texture = self.search_attn(x, ref_down_up, ref)

        # Out: [b, 64, h, w]
        feat = self.fe(x)

        # Out: [b, 128, h, w]
        feat_res = torch.cat((feat, texture), dim=1)
        feat_res = self.conv_first(feat_res)

        # soft-attention
        feat = feat + feat_res * soft_attention

        feat_res = self.res_block(feat)
        feat_res = self.conv_last(feat_res)

        return x + feat_res * 0.2


# Feature encoder module
class FE(BaseModule):
    def __init__(self, in_channels, mid_channels, num_blocks, res_scale):
        super().__init__()

        self.num_blocks = num_blocks
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x_res = x = F.leaky_relu(self.conv_first(x))
        x_res = self.body(x_res)
        x_res = self.conv_last(x_res)
        return x + x_res * 0.2


# Learnable texture encoder
class LTE(BaseModule):
    def __init__(self,
                 requires_grad=True,
                 pixel_range=1.):
        super().__init__()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * pixel_range, 0.224 * pixel_range,
                   0.225 * pixel_range)
        self.img_normalize = ImgNormalize(
            pixel_range=pixel_range, img_mean=vgg_mean, img_std=vgg_std)

        # use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(
            weights='DEFAULT').features

        self.slice = torch.nn.Sequential()

        for x in range(2):
            self.slice.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        x = self.img_normalize(x)

        return self.slice(x)


# Search texture attention from ref image
class SearchAttention(BaseModule):
    """Search texture reference by attention.

    Include relevance embedding, hard-attention and soft-attention.
    """

    def gather(self, inputs, dim, index):
        """Hard Attention. Gathers values along an axis specified by dim.

        Args:
            inputs (Tensor): The source tensor. (N, C*k*k, H*W)
            dim (int): The axis along which to index.
            index (Tensor): The indices of elements to gather. (N, H*W)

        results:
            outputs (Tensor): The result tensor. (N, C*k*k, H*W)
        """

        views = [inputs.size(0)
                 ] + [1 if i != dim else -1 for i in range(1, inputs.ndim)]
        expansion = [
            -1 if i in (0, dim) else d for i, d in enumerate(inputs.size())
        ]
        index = index.view(views).expand(expansion)
        outputs = torch.gather(inputs, dim, index)

        return outputs

    def forward(self, patch_lq_up, ref_down_up, ref):
        # query
        query = F.unfold(patch_lq_up, kernel_size=(3, 3), padding=1)

        # key
        key = F.unfold(ref_down_up, kernel_size=(3, 3), padding=1)
        key_t = key.permute(0, 2, 1)

        # values
        value = F.unfold(ref, kernel_size=(3, 3), padding=1)

        key_t = F.normalize(key_t, dim=2)  # [N, H*W, C*k*k]
        query = F.normalize(query, dim=1)  # [N, C*k*k, H*W]

        # Relevance embedding
        rel_embedding = torch.bmm(key_t, query)  # [N, H*W, H*W]
        max_val, max_index = torch.max(rel_embedding, dim=1)  # [N, H*W]

        # hard-attention
        texture = self.gather(value, 2, max_index)

        # to tensor
        h, w = patch_lq_up.size()[-2:]
        texture = F.fold(texture, output_size=(h, w), kernel_size=(3, 3), padding=1) / 9.

        soft_attention = max_val.view(max_val.size(0), 1, h, w)

        return soft_attention, texture
