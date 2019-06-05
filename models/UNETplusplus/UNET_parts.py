# sub-parts of the U-Net plus plus model

import torch
import torch.nn as nn
import torch.nn.functional as F

class single_conv(nn.Module):
    '''(conv => BN => ReLU). kernel_size = 1, just modify channel size'''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        # self.conv = double_conv(in_ch, out_ch)
        self.conv = single_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if isinstance(x2, list):
            for idx in range(len(x2)):
                x1 = torch.cat([x1, x2[idx]], dim=1)
        else:
            x1 = torch.cat([x1, x2], dim=1)
        x = self.conv(x1)
        return x

class convrelu(nn.Module):
    def __init__(self, in_ch, out_ch, kenerl_size, padding=1):
        super(convrelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kenerl_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
