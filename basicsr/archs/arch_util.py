import math
import torch
import numbers
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torch
from basicsr.utils import get_root_logger
import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F



@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class ResidualBlockdyNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockdyNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv_basic_dy_nobn(num_feat, num_feat, 1)
        self.conv2 = conv_basic_dy_nobn(num_feat, num_feat, 1)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, inputs):
        # identity = x
        # out = self.conv2(self.relu(self.conv1(x)))
        # return identity + out * self.res_scale
        identity = inputs['x'].clone()
        out = self.relu(self.conv1(inputs))
        conv2_input = {'x': out, 'weights': inputs['weights']}
        out = self.conv2(conv2_input)
        out = identity + out * self.res_scale
        return {'x': out, 'weights': inputs['weights']}

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, if_bias=True, K=5, init_weight=False):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = if_bias
        self.K = K

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes), requires_grad=True)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.if_bias:
                nn.init.constant_(self.bias[i], 0)

    def forward(self, inputs):
        x = inputs['x']
        softmax_attention = inputs['weights']
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class ResidualBlockNoBNDynamic(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, num_models=5):
        super(ResidualBlockNoBNDynamic, self).__init__()
        self.res_scale = res_scale
        self.conv1 = Dynamic_conv2d(num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv2 = Dynamic_conv2d(num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.relu = nn.ReLU(inplace=True)

        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, inputs):
        identity = inputs['x'].clone()
        out = self.relu(self.conv1(inputs))
        conv2_input = {'x':out, 'weights':inputs['weights']}
        out = self.conv2(conv2_input)
        out = identity + out * self.res_scale
        return {'x':out, 'weights':inputs['weights']}

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    if hh % scale != 0:
        x = x[:, :, :-(hh%scale), :]
    if hw % scale != 0:
        x = x[:, :, :, :-(hw%scale)]
    b, c, hh, hw = x.size()
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

class DMDG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(DMDG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            LDMB(conv, n_feat) \
            for _ in range(self.n_blocks)
        ]
        self.body = nn.Sequential(*modules_body)

        self.attention = nn.Sequential(
            conv(n_feat * self.n_blocks, n_feat, kernel_size=1),
            nn.LeakyReLU(0.1, True),

            default_conv(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1,True),

            nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat),
            nn.LeakyReLU(0.1,True),

            SCA(n_feat,n_feat),
            nn.LeakyReLU(0.1, True),
            CSA(n_feat,num_heads=4,ffn_expansion_factor=2,bias=False,LayerNorm_type="BiasFree")
        )


    def forward(self, x, weights):

        res = x
        dist = []

        for i in range(self.n_blocks):
            x= self.body[i](x, weights)
            dist.append(x)

        x = torch.cat(dist, dim=1)
        x = self.attention(x)
        x = x + res
        return x
class SCA(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        attention = self.sca(x)
        return x * attention
class CSA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CSA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# from    ops.layernorm import LayerNorm2d
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class MDTA_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LDMB(nn.Module):
    def __init__(self, conv, n_feat):
        super(LDMB, self).__init__()

        self.H_channel = n_feat // 2
        self.L_channel = n_feat // 4


        self.conv1=  nn.Sequential(

            default_conv(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat),
            default_conv(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True)

        )

        self.conv2  = nn.Sequential(
            default_conv(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, groups=n_feat),
            default_conv(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True)

        )

        self.HFE =  Branch1(nf=self.H_channel)
        self.LFDE = Branch2(nf=self.L_channel)

    def forward(self, x, weights):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x

        x = self.conv1(x)

        f_d, f_c = torch.chunk(x, 2, dim=1)

        f_d = self.HFE(f_d, weights)

        f_c = self.LFDE(f_c)

        x = torch.cat([f_d, f_c], dim=1)

        x = self.conv2(x)

        x = x + res
        return x
class Branch1(nn.Module):
    def __init__(self,nf):
        super(Branch1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, 3, 1, 1, groups=nf),
            nn.LeakyReLU(0.1, True),



        )
        self.DAconv =PDMB(nf, nf, kernel_size=3)





        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, 3, 1, 1, groups=nf),
            nn.LeakyReLU(0.1, True),

        )


    def forward(self, x, weights):
        x=self.conv(x)
        x=self.DAconv(x,weights)
        x=self.conv_last(x)
        return x
class PDMB(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(PDMB, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, channels_in * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = nn.Conv2d(channels_in, channels_out, 1, 1, 0)


        self.relu = nn.LeakyReLU(0.1, False)

    def forward(self, x,weights):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x.size()

        kernel = self.kernel(weights).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b * c, padding=(self.kernel_size - 1) // 2))
        out = self.conv(out.view(b, -1, h, w))


        return out
class Branch2(nn.Module):
    def __init__(self, nf):
        super(Branch2, self).__init__()

        self.BS_CONV_3x3_SCA = nn.Sequential(

            nn.Conv2d(nf, nf, kernel_size=1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(nf, nf, kernel_size=3, padding=1, groups=nf),
            nn.LeakyReLU(0.1, True),


            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            SCA(nf,nf)

        )

    def forward(self, x):
        f_c1, f_c2 = torch.chunk(x, 2, dim=1)

        f_c1 = self.BS_CONV_3x3_SCA(f_c1)

        return torch.cat([f_c1, f_c2], dim=1)