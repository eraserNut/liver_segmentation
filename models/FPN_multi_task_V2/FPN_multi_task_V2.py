import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .base_model import Model


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


class FPN_multi_task_V2(Model):

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

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.p1 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s1 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout)
        self.dropout_c = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)
        self.final_conv_c = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        # ==> get encoder features
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        # c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        p1 = self.p1([p2, c1])

        s1 = self.s1(p1)

        # x = s5 + s4 + s3 + s2

        x = self.dropout(s1)
        x_c = self.dropout_c(s1)
        x = self.final_conv(x)
        x_c = self.final_conv_c(x_c)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_c = F.interpolate(x_c, scale_factor=2, mode='bilinear', align_corners=True)
        if self.training:
            return F.sigmoid(x), F.sigmoid(x_c)
            # return F.sigmoid(x)
        else:
            return F.sigmoid(x), F.sigmoid(x_c)
            # return F.sigmoid(x)
