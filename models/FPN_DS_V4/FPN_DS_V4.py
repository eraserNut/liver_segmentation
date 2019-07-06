import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .base_model import Model
import numpy as np
from PIL import Image
from torchvision import transforms

to_pil = transforms.ToPILImage()


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class DSBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # ==> FN Sub-module
        self.fn_extra = Conv3x3GNReLU(in_channels, in_channels)

        # ==> FP Sub-module
        self.fp_extra = Conv3x3GNReLU(in_channels, in_channels)

    def forward(self, x):
        # ==> FN Sub-module
        fnd = self.fn_extra(x)

        # ==> FN Sub-module
        fpd = self.fp_extra(x)

        return fnd, fpd


class modifyBlock(nn.Module):
    def __init__(self, n_downsamples, mod, degree):
        super().__init__()
        self.n_downsamples = n_downsamples
        self.mod = mod
        self.degree = degree
        self.sigmoid = nn.Sigmoid()
        # self.normalization = nn.InstanceNorm2d(1)

    def forward(self, x, mask):
        assert(self.mod in ['strengthen', 'weaken'], 'need correct mod name')
        mask_downsample = F.interpolate(mask, scale_factor=1/(2**self.n_downsamples), mode='bilinear', align_corners=True)
        # mask_downsample = self.sigmoid(x_fd_downsample)
        # mask_downsample = self.normalization(mask_downsample)

        # b=np.array(mask_downsample.squeeze().cpu())
        # a=np.array(mask.data.squeeze(0).cpu())
        # b=mask_downsample.expand(x.size())
        # c=mask_downsample.expand(x.size())*500
        x_ = x * mask_downsample.expand(x.size()) * self.degree
        if self.mod == 'strengthen':
            x = x + x_
        elif self.mod == 'weaken':
            x = x - x_
        return x


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        return x

class FPN_DS_V4(Model):

    def __init__(
            self,
            encoder_channels=[512, 256, 128, 64],
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            dropout=0.2,
    ):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        # ==> encoder layers
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_down2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_down3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_down4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        # ==> DS module
        self.d5 = DSBlock(encoder_channels[0])
        self.d4 = DSBlock(encoder_channels[1])
        self.d3 = DSBlock(encoder_channels[2])
        self.d2 = DSBlock(encoder_channels[3])

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        # ==> DS module, fnd
        self.fnd_s5 = SegmentationBlock(encoder_channels[0], segmentation_channels, n_upsamples=3)
        self.fnd_s4 = SegmentationBlock(encoder_channels[1], segmentation_channels, n_upsamples=2)
        self.fnd_s3 = SegmentationBlock(encoder_channels[2], segmentation_channels, n_upsamples=1)
        self.fnd_s2 = SegmentationBlock(encoder_channels[3], segmentation_channels, n_upsamples=0)

        # ==> DS module, fpd
        self.fpd_s5 = SegmentationBlock(encoder_channels[0], segmentation_channels, n_upsamples=3)
        self.fpd_s4 = SegmentationBlock(encoder_channels[1], segmentation_channels, n_upsamples=2)
        self.fpd_s3 = SegmentationBlock(encoder_channels[2], segmentation_channels, n_upsamples=1)
        self.fpd_s2 = SegmentationBlock(encoder_channels[3], segmentation_channels, n_upsamples=0)

        # ==> DS module, strengthen block
        self.fnd_m5 = modifyBlock(3, 'strengthen', 0.05)
        self.fnd_m4 = modifyBlock(2, 'strengthen', 0.05)
        self.fnd_m3 = modifyBlock(1, 'strengthen', 0.05)
        self.fnd_m2 = modifyBlock(0, 'strengthen', 0.05)

        # ==> DS module, weaken block
        self.fpd_m5 = modifyBlock(3, 'weaken', 0.005)
        self.fpd_m4 = modifyBlock(2, 'weaken', 0.005)
        self.fpd_m3 = modifyBlock(1, 'weaken', 0.005)
        self.fpd_m2 = modifyBlock(0, 'weaken', 0.005)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        # self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)
        self.dupsample = DUpsampling(inplanes=segmentation_channels, scale=4, num_class=1)

        self.sig_instanceNormal = nn.Sequential(
            nn.Sigmoid(),
            nn.InstanceNorm2d(1),
            nn.ReLU(),
        )
        # ==> DS module, final
        self.dropout_fnd = nn.Dropout2d(p=dropout, inplace=True)
        self.dropout_fpd = nn.Dropout2d(p=dropout, inplace=True)
        self.dupsample_fnd = DUpsampling(inplanes=segmentation_channels, scale=4, num_class=1)
        self.dupsample_fpd = DUpsampling(inplanes=segmentation_channels, scale=4, num_class=1)
        self.final_conv_fnd = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)
        self.final_conv_fpd = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        # ==> get encoder features
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        # c5, c4, c3, c2, _ = x

        # ==> DS module
        fnd5, fpd5 = self.d5(c5)
        fnd4, fpd4 = self.d4(c4)
        fnd3, fpd3 = self.d3(c3)
        fnd2, fpd2 = self.d2(c2)

        # ==> fnd upsample
        fnd_s5 = self.fnd_s5(fnd5)
        fnd_s4 = self.fnd_s4(fnd4)
        fnd_s3 = self.fnd_s3(fnd3)
        fnd_s2 = self.fnd_s2(fnd2)

        # ==> fpd upsample
        fpd_s5 = self.fpd_s5(fpd5)
        fpd_s4 = self.fpd_s4(fpd4)
        fpd_s3 = self.fpd_s3(fpd3)
        fpd_s2 = self.fpd_s2(fpd2)

        # ==> DS module, fnd fpd
        x_fnd = fnd_s5 + fnd_s4 + fnd_s3 + fnd_s2
        x_fpd = fpd_s5 + fpd_s4 + fpd_s3 + fpd_s2
        x_fnd = self.dropout_fnd(x_fnd)
        x_fpd = self.dropout_fpd(x_fpd)
        x_fnd_single = self.final_conv_fnd(x_fnd)
        x_fpd_single = self.final_conv_fpd(x_fpd)
        fnd_mask = self.sig_instanceNormal(x_fnd_single)
        fpd_mask = self.sig_instanceNormal(x_fpd_single)

        m5 = self.fnd_m5(c5, fnd_mask)
        m5 = self.fpd_m5(m5, fpd_mask)
        m4 = self.fnd_m4(c4, fnd_mask)
        m4 = self.fpd_m4(m4, fpd_mask)
        m3 = self.fnd_m3(c3, fnd_mask)
        m3 = self.fpd_m3(m3, fpd_mask)
        m2 = self.fnd_m2(c2, fnd_mask)
        m2 = self.fpd_m2(m2, fpd_mask)

        # x_fnd = F.interpolate(x_fnd, scale_factor=4, mode='bilinear', align_corners=True) #return
        # x_fpd = F.interpolate(x_fpd, scale_factor=4, mode='bilinear', align_corners=True) #return
        x_fnd = self.dupsample_fnd(x_fnd)
        x_fpd = self.dupsample_fpd(x_fpd)

        p5 = self.conv1(m5)
        p4 = self.p4([p5, m4])
        p3 = self.p3([p4, m3])
        p2 = self.p2([p3, m2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        x = s5 + s4 + s3 + s2

        x = self.dropout(x)
        # x = self.final_conv(x)
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.dupsample(x)

        if self.training:
            return F.sigmoid(x), F.sigmoid(x_fnd), F.sigmoid(x_fpd)
        else:
            # return F.sigmoid(x), [mask2, mask3, mask4, mask5]
            return F.sigmoid(x), [F.sigmoid(x_fpd)]
