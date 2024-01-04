import torch
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn as nn
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer

class ResidualUnit(nn.Module):
    def __init__(self, dim,kernel_size):
        super(ResidualUnit, self).__init__()
        self.Net=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim,padding=kernel_size//2),
            nn.GELU(),

        )
    def forward(self, x):
        return self.Net(x)+x


@ARCH_REGISTRY.register()
class ConvMixer(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_params=33, use_bias=True,patch_size=4,kernen_size=5,depth=20):
        super(ConvMixer, self).__init__()
        self.depth=depth
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_nc,nf,kernel_size=patch_size,stride=patch_size),
            nn.GELU(),

            *[nn.Sequential(
                ResidualUnit(nf,kernen_size),
                nn.Conv2d(nf, nf, kernel_size=1),
                nn.GELU(),
            )
                for i in range(self.depth)],
            nn.GELU(),

            nn.Conv2d(nf, num_params, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GELU(),

        )
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.MappingNet = nn.Sequential(*[
            nn.Linear(num_params, nf//2)
        ])

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)#B 33,1,1

        out_params = flat.view(flat.size()[:2])############B C
        mapped_weights = self.MappingNet(out_params)#B 5
        return out_params, mapped_weights#
