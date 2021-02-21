#!/usr/bin/env python3
# coding: utf-8

"""
@File      : affine.py
@Author    : alex
@Date      : 2021/2/15
@Desc      : 以cv2风格为准, img-size: (h, w), size: (w, h)
"""

import math
import torch
import random
import numbers
import warnings
import numpy as np
from PIL import Image
from .compose import *
import idata.augment.affine as F
from collections.abc import Sequence, Iterable
from idata.augment.utils import _get_image_size, np_to_pil

__all__ = [
    "Resize", "Pad", "CenterCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
    "RandomRotation",
    # "RandomAffine", "RandomPerspective",
    # "RandomCrop", "RandomResizedCrop",
]


class Resize(object):
    """ Resize img & seg.
    等比例缩放.
            1) 若size为int,以输入图最短边作为缩放依据,长边等比例缩放;
            2) 若size为(w, h),输出图非等比例缩放至预定大小.
    :param img: PIL Image or Numpy Image.
    :param size: (sequence or int)
    :param interpolation: 'linear' or 'nearest', default='linear'
    :return: Resized image.
    example:
        ret = resize(img, size) # img, (w, h) = (100, 200)
        size = 50, ret, (w, h) = (50, 100);
        size = 200, ret, (w, h) = (200, 400);
        size = (30, 50), ret, (w, h) = (30, 50).
    """

    def __init__(self, size, interpolation="linear"):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, result):
        meta = dict()
        result[TS_IMG] = F.resize(result[TS_IMG], self.size, self.interpolation, meta=meta)
        # print("Resize meta:", meta)

        if TS_SEG in result:
            result[TS_SEG] = F.resize(result[TS_SEG], self.size, "nearest")

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    """ Resize img & seg.
    为图像增加padding
    :param img: Numpy Image or PIL Image.
    :param padding: int or tuple,几种形式:
            padding=50, 则等价于,  pad_left = pad_right = pad_top = pad_bottom = padding
            padding=(50, 100), 则等价于,
    :param padding_mode: 以具体图像类型定
    :param fill: borderType为CONSTANT时有效, 设置填充的像素值.
    :return: Padding image
    """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, result):
        meta = dict()
        result[TS_IMG] = F.pad(result[TS_IMG], self.padding, self.fill, self.padding_mode, meta=meta)
        # print("Pad meta:", meta)

        if TS_SEG in result:
            result[TS_SEG] = F.pad(result[TS_SEG], self.padding, result[TS_IGNORE_LABEL], self.padding_mode)

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1})'. \
            format(self.padding, self.fill)


class CenterCrop(object):
    """ Resize img & seg.
    中心剪裁，若剪裁区域大于图像区域，补Padding
    :param size：  (w, h) or int
    """

    def __init__(self, size, ignore_label=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_label = ignore_label

    def __call__(self, result):
        meta = dict(org_shape=_get_image_size(result[TS_IMG]))
        result[TS_IMG] = F.center_crop(result[TS_IMG], self.size, meta=meta)
        # print("CenterCrop meta:", meta)

        if TS_SEG in result:
            result[TS_SEG] = F.center_crop(result[TS_SEG], self.size)
            result[TS_SEG] = self.seg_fill(result[TS_SEG], meta)
        return result

    def seg_fill(self, seg, meta):
        m_crop = meta["crop"]
        oh, ow = meta["org_shape"]
        l, t, w, h = m_crop["left"], m_crop["top"], m_crop["width"], m_crop["height"]
        if l < 0:
            seg[:, 0:-l] = self.ignore_label
        if t < 0:
            seg[0:-t, :] = self.ignore_label
        if w > ow:
            seg[:, ow:w] = self.ignore_label
        if h > oh:
            seg[oh:h, :] = self.ignore_label
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(torch.nn.Module):
    """ 对Image进行水平翻转
    :param img: Tensor Image，[C, H, W]； Numpy or PIL Image
    :param p: 触发概率，默认为0.5
    :return: 水平翻转后图像
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, result):
        if torch.rand(1) < self.p:
            result[TS_IMG] = F.hflip(result[TS_IMG])

            if TS_SEG in result:
                result[TS_SEG] = F.hflip(result[TS_SEG])

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    """ 对Image进行垂直翻转
    :param img: Tensor Image，[C, H, W]； Numpy or PIL Image
    :param p: 触发概率，默认为0.5
    :return: 垂直翻转后图像
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, result):
        if torch.rand(1) < self.p:
            result[TS_IMG] = F.vflip(result[TS_IMG])

            if TS_SEG in result:
                result[TS_SEG] = F.vflip(result[TS_SEG])

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """ 在degrees度数内，随机旋转
    """

    def __init__(self, degrees, resample="linear", expand=True, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, result):
        angle = self.get_params(self.degrees)

        result[TS_IMG] = F.rotate(result[TS_IMG], angle, self.center, self.fill, self.resample, self.expand)

        if TS_SEG in result:
            result[TS_SEG] = F.rotate(result[TS_SEG], angle, self.center, 255, "nearest", self.expand)

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string


class RandomAffine(object):
    pass


class RandomPerspective(object):
    pass


"""
    基础功能扩展
"""


class RandomCrop(object):
    """ 随机剪裁
    :param size: sequence(w, h) or int
    :param padding: int, [left/right, top/bottom], [left, top, right, bottom]
    :param fill:
    :param padding_mode:
        example:  trans = RandomCrop((768, 500), pad_if_needed=True)
    """

    def __init__(self, size, padding=None, fill=0, padding_mode=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        h, w = _get_image_size(img)
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        img_size = _get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and img_size[1] < self.size[0]:
            img = F.pad(img, (self.size[0] - img_size[1], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img_size[0] < self.size[1]:
            img = F.pad(img, (0, self.size[1] - img_size[0]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomResizedCrop(object):
    """ 裁剪给定的PIL图像到随机大小和高宽比
    :param size: 输出的图像尺寸
    :param scale, ratio: 与目标检测时anchor生成参数类似，决定剪裁的宽高
    :param interpolation:
    """

    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=None):

        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        height, width = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string
