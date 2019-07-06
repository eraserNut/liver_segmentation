import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .base_model import Model


class single_upsample(nn.Module):
    def __init__(self, n_upsamples=0):
        super().__init__()
        self.n_upsamples = n_upsamples

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2**self.n_upsamples, mode='bilinear', align_corners=True)
        return x


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
    def __init__(self, in_channels, final_channels):
        super().__init__()
        # ==> strength coff
        self.fn_augRate = 1
        self.fp_augRate = 1

        # ==> FN Sub-module
        self.fn_extra = Conv3x3GNReLU(in_channels, in_channels)
        self.fn_attention = nn.Conv2d(in_channels, 1, (3, 3), stride=1, padding=1, bias=False)
        self.fn_sigmoid = nn.Sigmoid()

        # self.fn_final = nn.Conv2d(in_channels, final_channels, (3, 3), stride=1, padding=1, bias=False)

        # ==> FP Sub-module
        self.fp_extra = Conv3x3GNReLU(in_channels, in_channels)
        self.fp_attention = nn.Conv2d(in_channels, 1, (3, 3), stride=1, padding=1, bias=False)
        self.fp_sigmoid = nn.Sigmoid()

        # self.fp_final = nn.Conv2d(in_channels, final_channels, (3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        # ==> FN Sub-module
        fnd = self.fn_extra(x)
        fn_mask = self.fn_attention(fnd)
        fn_mask_sig = self.fn_sigmoid(fn_mask)
        fnd_ = x * fn_mask_sig.expand(x.size()) * self.fn_augRate ## control mask strength coff
        x_fn = fnd_ + x

        # ==> FN Sub-module
        fpd = self.fp_extra(x)
        fp_mask = self.fp_attention(fpd)
        fp_mask_sig = self.fp_sigmoid(fp_mask)
        fpd_ = x * fp_mask_sig.expand(x.size()) * self.fp_augRate ## control mask strength coff
        x_ds = x_fn - fpd_
        return x_ds, fn_mask, fp_mask


class FPN_DS_V2(Model):

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
        self.d5 = DSBlock(encoder_channels[0], final_channels)
        self.d4 = DSBlock(encoder_channels[1], final_channels)
        self.d3 = DSBlock(encoder_channels[2], final_channels)
        self.d2 = DSBlock(encoder_channels[3], final_channels)

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        # ==> DS module, fnd
        self.fnd_s5 = single_upsample(n_upsamples=5)
        self.fnd_s4 = single_upsample(n_upsamples=4)
        self.fnd_s3 = single_upsample(n_upsamples=3)
        self.fnd_s2 = single_upsample(n_upsamples=2)

        # ==> DS module, fpd
        self.fpd_s5 = single_upsample(n_upsamples=5)
        self.fpd_s4 = single_upsample(n_upsamples=4)
        self.fpd_s3 = single_upsample(n_upsamples=3)
        self.fpd_s2 = single_upsample(n_upsamples=2)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        # ==> DS module, final
        # self.dropout_fnd = nn.Dropout2d(p=dropout, inplace=True)
        # self.dropout_fpd = nn.Dropout2d(p=dropout, inplace=True)
        # self.final_conv_fnd = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)
        # self.final_conv_fpd = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

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
        d5, fn_mask5, fp_mask5 = self.d5(c5)
        d4, fn_mask4, fp_mask4 = self.d4(c4)
        d3, fn_mask3, fp_mask3 = self.d3(c3)
        d2, fn_mask2, fp_mask2 = self.d2(c2)

        p5 = self.conv1(d5)
        p4 = self.p4([p5, d4])
        p3 = self.p3([p4, d3])
        p2 = self.p2([p3, d2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        # ==> DS module, fnd upsample
        fnd_s5 = self.fnd_s5(fn_mask5)
        fnd_s4 = self.fnd_s4(fn_mask4)
        fnd_s3 = self.fnd_s3(fn_mask3)
        fnd_s2 = self.fnd_s2(fn_mask2)

        # ==> DS module, fpd upsample
        fpd_s5 = self.fpd_s5(fp_mask5)
        fpd_s4 = self.fpd_s4(fp_mask4)
        fpd_s3 = self.fpd_s3(fp_mask3)
        fpd_s2 = self.fpd_s2(fp_mask2)

        x = s5 + s4 + s3 + s2

        x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # ==> DS module, fnd fpd
        # x_fnd = fnd_s5 + fnd_s4 + fnd_s3 + fnd_s2
        # x_fpd = fpd_s5 + fpd_s4 + fpd_s3 + fpd_s2
        # x_fnd = self.dropout_fnd(x_fnd)
        # x_fpd = self.dropout_fpd(x_fpd)
        # x_fnd = self.final_conv_fnd(x_fnd)
        # x_fpd = self.final_conv_fpd(x_fpd)
        # x_fnd = F.interpolate(x_fnd, scale_factor=4, mode='bilinear', align_corners=True)
        # x_fpd = F.interpolate(x_fpd, scale_factor=4, mode='bilinear', align_corners=True)

        if self.training:
            # return F.sigmoid(x), [fnd_s5, fnd_s4, fnd_s3, fnd_s2],\
            #        [fpd_s5, fpd_s4, fpd_s3, fpd_s2]
            return F.sigmoid(x), [F.sigmoid(fnd_s5), F.sigmoid(fnd_s4), F.sigmoid(fnd_s3), F.sigmoid(fnd_s2)],\
                   [F.sigmoid(fpd_s5), F.sigmoid(fpd_s4), F.sigmoid(fpd_s3), F.sigmoid(fpd_s2)]
        else:
            # return F.sigmoid(x), [mask2, mask3, mask4, mask5]
            return F.sigmoid(x), [F.sigmoid(fnd_s2), F.sigmoid(fnd_s3), F.sigmoid(fnd_s4), F.sigmoid(fnd_s5)]
