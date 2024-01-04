import logging
import torch
from os import path as osp
import torch.nn as nn
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from torchstat import stat

# from torchsummaryX import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def test_pipeline(root_path):
    opt = parse_options(root_path, is_train=False)
    model = build_model(opt)

    tensor = torch.rand(1, 3, 256, 256).to("cuda")
    tensor_weight = torch.rand(1, 32).to("cuda")


    with torch.no_grad():
        flops1 = FlopCountAnalysis(model.net_g, (tensor,tensor_weight))

        flops2 = FlopCountAnalysis(model.net_p, tensor)
    #

    print(parameter_count_table(model.net_p))
    print(parameter_count_table(model.net_g))
    print("FLOPs1: ", flops1.total()/1e9)
    print("FLOPs2: ", flops2.total()/1e9)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
#PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python E2RealSR/parameters.py -opt options/test/E2RealSR/test_E2RealSR.yml