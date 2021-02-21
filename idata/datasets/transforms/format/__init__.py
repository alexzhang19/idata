#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/14
@Desc      : 
"""

import sys
import random
import collections
from . import format_func as F
from ..compose import TS_IMG

__all__ = [
    "ToTensor",  # PIL Image or numpy.ndarray 转 Tensor
    "ToNumpy",  # PIL Image or Tensor 转 numpy.ndarray
    "ToPILImage",  # PIL Image or Tensor 转 numpy.ndarray
    "ToBGR",  # 通道顺序转化，RGB->BGR
    "ToRGB",  # 通道顺序转化，BGR->RGB
    "Normalize",  # Tensor依据均值方差标准化
    "UnNormalize",  # Normalize的反操作，用于显示真实图像
]

"""
    transpose实现对单张图的增广，约定：
    1、默认处理对象为图像，PIL与Numpy转换过程中不显式改变C通道顺序；
    2、PIL Image默认(H x W x C)，值域在[0, 255]，C通道为RGB形式；
    3、numpy.ndarray默认(H x W x C)，值域在[0, 255]，C通道为BGR形式；
    4、Tensor默认(C x H x W)，值域在[0.0, 1.0].
"""


class ToTensor(object):
    """将一个PIL Image或者numpy.ndarray转为tensor. 增加对Numpy的支持
    将一个(H x W x C)格式，值域在[0, 255]的PIL Image或者numpy.ndarray转为(C x H x W)，值域在[0.0, 1.0]的torch.FloatTensor。
    PIL Image为(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)模式中的一个；
    numpy.ndarray必须为dtype = np.uint8。
    其它格式的输入，则原样返回。
    """

    def __call__(self, result):
        result[TS_IMG] = F.to_tensor(result[TS_IMG])
        return result

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNumpy(object):
    """将一个PIL Image或者Tensor转为Numpy.
    pic: 输入图像矩阵.
        PIL转换：默认其通道为(h, w, c)，值域[0~255]；将PIL转为Numpy格式，；
        Tensor转换：默认其通道为(c, h, w)，值域[0.0~1.0]；调整通道顺序为(h, w, c)，将Tensor转为Numpy格式，并×255；
    """

    def __init__(self, trans_to_bgr=False):
        """
        trans_to_bgr: 是否转为bgr
        """

        self.trans_to_bgr = trans_to_bgr

    def __call__(self, result):
        result[TS_IMG] = F.to_numpy(result[TS_IMG], self.trans_to_bgr)
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.trans_to_bgr:
            format_string += 'trans_to_bgr=True, '
        format_string += ')'
        return format_string


class ToPILImage(object):
    """将一个Numpy或者Tensor转为PIL.Image
    pic: 输入图像矩阵.
        Numpy转换：默认其通道为(h, w, c)，值域[0~255]；将PIL转为Numpy格式， 保持取值范围不变；
        Tensor转换：默认其通道为(c, h, w)，值域[0.0~1.0]；调整通道顺序为(h, w, c)，将Tensor转为PIL.Image格式，并×255；
   """

    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, result):
        result[TS_IMG] = F.to_pil_image(result[TS_IMG], self.mode)
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class ToBGR(object):
    def __call__(self, result):
        result[TS_IMG] = F.rgb2bgr(result[TS_IMG])
        return result

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToRGB(object):
    def __call__(self, result):
        result[TS_IMG] = F.bgr2rgb(result[TS_IMG])
        return result

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """对Tensor进行归一化处理, tensor = (tensor - mean) / std.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, result, keys=[TS_IMG]):
        result[TS_IMG] = F.normalize(result[TS_IMG], self.mean, self.std, self.inplace)
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class UnNormalize(object):
    """Normalize的逆操作
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, result):
        result[TS_IMG] = F.unnormalize(result[TS_IMG], self.mean, self.std, self.inplace)
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
