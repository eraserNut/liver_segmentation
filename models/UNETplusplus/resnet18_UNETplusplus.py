import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
from .UNET_parts import *


class UNETplusplus(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.ch_size = [64, 64, 128, 256, 512]
        # ==> down layers
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_down2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_down3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_down4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        # ==> up layers
        self.layer_up01 = up(self.ch_size[0]+self.ch_size[1], self.ch_size[0])
        self.layer_up11 = up(self.ch_size[1]+self.ch_size[2], self.ch_size[1])
        self.layer_up21 = up(self.ch_size[2]+self.ch_size[3], self.ch_size[2])
        self.layer_up31 = up(self.ch_size[3]+self.ch_size[4], self.ch_size[3])

        self.layer_up02 = up(self.ch_size[0]*2+self.ch_size[1], self.ch_size[0])
        self.layer_up12 = up(self.ch_size[1]*2+self.ch_size[2], self.ch_size[1])
        self.layer_up22 = up(self.ch_size[2]*2+self.ch_size[3], self.ch_size[2])

        self.layer_up03 = up(self.ch_size[0]*3 + self.ch_size[1], self.ch_size[0])
        self.layer_up13 = up(self.ch_size[1]*3 + self.ch_size[2], self.ch_size[1])

        self.layer_up04 = up(self.ch_size[0]*4+self.ch_size[1], self.ch_size[0])

        # ==> ori layers
        self.conv_original_size0 = convrelu(3, self.ch_size[0], 3, 1)
        self.conv_original_size1 = convrelu(self.ch_size[0], self.ch_size[0], 3, 1)
        self.conv_original_size2 = convrelu(self.ch_size[0] + self.ch_size[0], self.ch_size[0], 3, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ==> class layers
        self.conv_last = nn.Conv2d(self.ch_size[0], n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # ==> features layer 1
        x00 = self.layer_down0(input) #x00
        x10 = self.layer_down1(x00)
        x20 = self.layer_down2(x10)
        x30 = self.layer_down3(x20)
        x40 = self.layer_down4(x30)

        # ==> features layer 2
        x01 = self.layer_up01(x10, x00) #output1
        x11 = self.layer_up11(x20, x10)
        x21 = self.layer_up21(x30, x20)
        x31 = self.layer_up31(x40, x30)

        # ==> features layer 3
        x02 = self.layer_up02(x11, [x00, x01]) #output2
        x12 = self.layer_up12(x21, [x10, x11])
        x22 = self.layer_up22(x31, [x20, x21])

        # ==> features layer 4
        x03 = self.layer_up03(x12, [x00, x01, x02]) #output3
        x13 = self.layer_up13(x22, [x10, x11, x12])

        # ==> features layer 5
        x04 = self.layer_up04(x13, [x00, x01, x02, x03]) #output4

        # ==> fuse with original
        x4 = self.upsample(x04)
        x4 = torch.cat([x4, x_original], dim=1)
        x4 = self.conv_original_size2(x4)

        x3 = self.upsample(x03)
        x3 = torch.cat([x3, x_original], dim=1)
        x3 = self.conv_original_size2(x3)

        x2 = self.upsample(x02)
        x2 = torch.cat([x2, x_original], dim=1)
        x2 = self.conv_original_size2(x2)

        x1 = self.upsample(x01)
        x1 = torch.cat([x1, x_original], dim=1)
        x1 = self.conv_original_size2(x1)

        # ==> class layer
        out1 = self.conv_last(x1)
        out2 = self.conv_last(x2)
        out3 = self.conv_last(x3)
        out4 = self.conv_last(x4)

        if self.training:
            return [torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), torch.sigmoid(out4)]
        else:
            return torch.sigmoid((out1+out2+out3+out4)/4)
            # return  torch.sigmoid(out4)
