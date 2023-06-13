# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import flow_warp, make_layer
from mmagic.registry import MODELS
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """Forward function."""
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


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


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, pretrained=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interpolation='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class Re_RealBasicVSRNet(BaseModule):
    """Re_RealBasicVSR network structure for real-world video super-resolution.

    Support only x4 upsampling.

    Undergradudate thesis

    Args:
        num_feat (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_block (int, optional): Number of residual blocks in
            each propagation branch. Default: 9.
        spynet_path (str, optional): Pre-trained model path of SPyNet.
            Default: None.
    """
    def __init__(self, 
                 mid_channels=64,
                 num_blocks=10,
                 num_cleaning_blocks=10,
                 max_residue_magnitude=10,
                 spynet_pretrained=None):
        super().__init__()

        self.mid_channels = mid_channels

        # Feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels // 8, 5)
        self.feat_extract_clean = ResidualBlocksWithInputConv(3, mid_channels - mid_channels // 8, 5)

        # Alignment
        self.spynet = SpyNet(pretrained=spynet_pretrained)

        # Deformable alignment
        self.deform_align = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)

        # Unidirectional propagation
        self.propagation = ResidualBlocksWithInputConv(mid_channels * 2, mid_channels, num_blocks)

        # Reconstruction
        self.fusion_attn = LargeKernelAttn(mid_channels)

        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # img cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        self.spynet.requires_grad_(True)

    def get_flow(self, x):
        """Get optical flow function for Re_RealBasicVSR.

        Args:
            x (tensor): Input low quality (LQ) sequence with
                shape (b, n, c, h, w).
        Returns:
            Tensor: flow tensor with shape (b, n, 2, h, w)
        """
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_forward: [b, 30, 2, h, w]
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        
        return flows_forward
    
    def propagate(self, feats, flows):
        """Propagate the latent features throughout the sequence.

        Args:
            feats (tensor): Features extracted shape (b, n, c, h, w).
            flows (tensor): Optical flows with shape (b, n - 1, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        # [b, 29, 2, h, w]
        b, n, _, h, w = flows.size()

        # PyTorch 2.0 could not compile data type of 'range'
        # frame_idx = range(0, n + 1)
        # flow_idx = range(-1, n)
        frame_idx = list(range(0, n + 1))
        flow_idx = list(range(-1, n))

        feat_prop = flows.new_zeros(b, self.mid_channels, h, w)
        feats_output = []
        for i, idx in enumerate(frame_idx):
            feat_current = feats[:, idx, :, :, :]
            # second-order deformable alignment
            if i > 0:
                flow_t1 = flows[:, flow_idx[i], :, :, :]

                cond_t1 = flow_warp(feat_prop, flow_t1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_t2 = torch.zeros_like(feat_prop)
                flow_t2 = torch.zeros_like(flow_t1)
                cond_t2 = torch.zeros_like(cond_t1)

                if i > 1:  # second-order features
                    feat_t2 = feats_output[-2]

                    flow_t2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_t2 = flow_t1 + flow_warp(flow_t2,
                                                  flow_t1.permute(0, 2, 3, 1))
                    cond_t2 = flow_warp(feat_t2, flow_t2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_t1, feat_current, cond_t2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_t2], dim=1)
                feat_prop = self.deform_align(feat_prop, cond, flow_t1, flow_t2)

            # concatenate and residual blocks
            feat = [feat_current] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.propagation(feat)
            feats_output.append(feat_prop)

        return feats_output

    def forward(self, lqs: torch.Tensor, return_lqs=False):
        """Forward function for Re_RealBasicVSR.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (b, n, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        # [b, 30, 3, h, w]
        b, n, c, h, w = lqs.size()
        
        lqs_clean = lqs.detach().clone()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            lqs_clean = lqs_clean.view(-1, c, h, w)
            residues = self.image_cleaning(lqs_clean)
            lqs_clean = (lqs_clean + residues).view(b, n, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < 1.0:
                break

        # compute optical flow
        flows_forward = self.get_flow(lqs_clean)
        
        # Input: shape(lqs): [b, 3, 64, 64]
        # Output: shape(feat): [b, 32, 64, 64]
        feat = self.feat_extract(lqs.view(-1, c, h, w))
        feat_clean = self.feat_extract_clean(lqs_clean.view(-1, c, h, w))

        feat = feat.view(b, n, -1, h, w)
        feat_clean = feat_clean.view(b, n, -1, h, w)
        feats = torch.cat([feat, feat_clean], dim=2)
        # feats: [b, 30, 64, 64, 64]


        propagated = self.propagate(feats, flows_forward)
        outputs = []

        for i in range(0, n):
            out = propagated[i]
            out = self.fusion_attn(out)
            
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lqs[:, i, :, :, :])
            out += base
            outputs.append(out)

        if return_lqs:
            return torch.stack(outputs, dim=1), lqs_clean
        else:
            return torch.stack(outputs, dim=1)
