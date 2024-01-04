import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBNDynamic, make_layer, Dynamic_conv2d,DAG,default_conv,LFFB_dynamic,LFFB,LFFB_wo_Transformer,LFFB_wo_Simple_Attention,LFFB_DS_CONV,LFFB_w_o_split,LFFB_wo_CA_SCA,LFFB_dynamic_one_modulation,LFFB_dynamic_Control_number_modulation,LFFB_dynamic_Control_number_3_modulation,LFFB_dynamic_Control_number_4_modulation,LFFB_dynamic_Control_number_5_modulation,LFFB_dynamic_Control_number_6_modulation


@ARCH_REGISTRY.register()
class DASRDynamic_V3(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_V3, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_dynamic(default_conv, num_feat, kernel_size=3, reduction=16, n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 1)
        return x
    def forward(self, x , weights):
        H, W = x.shape[2:]
        if self.window_size!=0:
            x = self.check_image_size(x)

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


        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x
@ARCH_REGISTRY.register()
class DASRDynamic_wo_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_wo_modulation, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 1)
        return x
    def forward(self, x ):
        H, W = x.shape[2:]
        if self.window_size!=0:
            x = self.check_image_size(x)

        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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


        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x
@ARCH_REGISTRY.register()
class DASRD_wo_Transformer(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRD_wo_Transformer, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_wo_Transformer(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 1)
        return x
    def forward(self, x ):
        H, W = x.shape[2:]
        if self.window_size!=0:
            x = self.check_image_size(x)

        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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


        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x
@ARCH_REGISTRY.register()
class DASRD_wo_Simple_Attention(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRD_wo_Simple_Attention, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_wo_Simple_Attention(default_conv, num_feat,kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )


    def forward(self, x ):


        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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

@ARCH_REGISTRY.register()
class DASRD_DS_CONV(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRD_DS_CONV, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_DS_CONV(default_conv, num_feat,kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )


    def forward(self, x ):


        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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

@ARCH_REGISTRY.register()
class DASRD_w_o_split_CA_SCA(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRD_w_o_split_CA_SCA, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_w_o_split(default_conv, num_feat,kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )


    def forward(self, x ):


        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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

@ARCH_REGISTRY.register()
class DASRDynamic_wo_modulation_w_o_CA(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_wo_modulation_w_o_CA, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_wo_CA_SCA(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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

        # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.compress = nn.Sequential(
        # nn.Linear(5, 64, bias=False),
        # nn.LeakyReLU(0.1, True)
        # )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 1)
        return x
    def forward(self, x ):
        H, W = x.shape[2:]
        if self.window_size!=0:
            x = self.check_image_size(x)

        x = self.head(x)
        res = x

        distill=[]
        for i in range(self.n_groups):
            x = self.body[i](x)
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


        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x
@ARCH_REGISTRY.register()
class DASRDynamic_one_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_one_modulation, self).__init__()
        self.window_size=0
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)

        modules_body = [
            LFFB_dynamic_one_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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



    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 1)
        return x
    def forward(self, x , weights):
        H, W = x.shape[2:]
        if self.window_size!=0:
            x = self.check_image_size(x)

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


        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x
@ARCH_REGISTRY.register()
class DASRDynamic_ablation_two_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_ablation_two_modulation, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)
        modules_body = [
            LFFB_dynamic_Control_number_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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
      # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x , weights):
        H, W = x.shape[2:]

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
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x

@ARCH_REGISTRY.register()
class DASRDynamic_ablation_three_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_ablation_three_modulation, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)
        modules_body = [
            LFFB_dynamic_Control_number_3_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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
      # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x , weights):
        H, W = x.shape[2:]

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
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x

@ARCH_REGISTRY.register()
class DASRDynamic_ablation_four_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_ablation_four_modulation, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)
        modules_body = [
            LFFB_dynamic_Control_number_4_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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
      # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x , weights):
        H, W = x.shape[2:]

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
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x

@ARCH_REGISTRY.register()
class DASRDynamic_ablation_five_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_ablation_five_modulation, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)
        modules_body = [
            LFFB_dynamic_Control_number_5_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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
      # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x , weights):
        H, W = x.shape[2:]

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
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x

@ARCH_REGISTRY.register()
class DASRDynamic_ablation_six_modulation(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, num_group=5,upscale=4):
        super(DASRDynamic_ablation_six_modulation, self).__init__()
        self.upscale = upscale
        self.n_groups=num_group
        self.head =default_conv(num_in_ch, num_feat, kernel_size=3)
        modules_body = [
            LFFB_dynamic_Control_number_6_modulation(default_conv, num_feat, kernel_size=3,n_blocks=num_block) \
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
      # activation function
        self.act= nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x , weights):
        H, W = x.shape[2:]

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
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x