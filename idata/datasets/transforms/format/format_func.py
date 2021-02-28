#!/usr/bin/env python3
# coding: utf-8

"""
@File      : format_func.py
@Author    : alex
@Date      : 2020/6/17
@Desc      :
"""

import cv2
import sys
import torch
import numpy as np
import collections
from PIL import Image
from idata.augment.utils import *

try:
    import accimage
except ImportError:
    accimage = None

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["normalize", "unnormalize"]


def normalize(tensor, mean, std, inplace=False):
    """ 对Tensor进行归一化处理, tensor = (tensor -mean) / std.
    :param tensor: Tensor image of size (C, H, W) to be normalized.
    :param mean, std: 每个通道的均值,方差
    :param inplace: 是否使用in-place操作
    :return: Normalized Tensor image
    """

    if not is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


def unnormalize(tensor, mean, std, inplace=False):
    """ normalize的逆操作
    """

    if not is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor
