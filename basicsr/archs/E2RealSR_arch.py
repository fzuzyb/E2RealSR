import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBNDynamic, make_layer, Dynamic_conv2d,default_conv,DMDG


@ARCH_REGISTRY.register()
class E2RealSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(E2RealSR, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            DMDG(default_conv, num_feat, kernel_size=3, n_blocks=num_block) \
            for _ in range(self.n_groups)
        ]
        self.body = nn.Sequential(*modules_body)

        self.body_1x1 = default_conv(num_feat*self.n_groups, num_feat, kernel_size=1)
        self.body_3x3 = default_conv(num_feat, num_feat, kernel_size=3)

        if self.upscale in [2, 3]:
            self.upconv1 = default_conv(num_feat, num_feat * self.upscale * self.upscale, 3, )
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = default_conv(num_feat, num_feat * 4, 3)
            self.upconv2 = default_conv(num_feat, num_feat * 4, 3)
            self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_before_up = default_conv(num_feat, num_feat, 3)
        self.conv_hr = default_conv(num_feat, num_out_ch, 3)

        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)




    def forward(self, x , weights):

        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x,weights)
            distill.append(x)
        x=torch.cat(distill,dim=1)

        x=self.body_1x1(x)
        x=self.act(x)
        x=self.body_3x3(x)
        x=self.act(x)

        x=x+res
        x=self.conv_before_up(x)
        x=self.act(x)
        if self.upscale == 4:
           x = self.pixel_shuffle(self.upconv1(x))
           x= self.pixel_shuffle(self.upconv2(x))
        elif self.upscale in [2, 3]:
            x = self.act(self.pixel_shuffle(self.upconv1(x)))

        x=self.conv_hr(x)



        return x
