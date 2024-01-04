import cv2
import numpy as np
import cv2
import os
import glob
import lpips

import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2):
    img_restored = img1.astype(np.float32) / 255.
    img_gt = img2.astype(np.float32) / 255.

    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
    # norm to [-1, 1]
    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    # calculate lpips
    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
    # log = f'{i + 1:3d}: {file_name:25}. \tLPIPS: {lpips_val.item():.6f}.'
    return lpips_val.item()



