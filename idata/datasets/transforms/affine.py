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
from .compose import *
import idata.augment.affine as F
from collections.abc import Sequence, Iterable
from idata.augment.utils import get_image_shape

__all__ = [
    "Resize", "Pad", "CenterCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
    "RandomRotation", "RandomAffine", "RandomPerspective",
    "RandomCrop", "RandomResizedCrop", "ResizedCrop",
]

"""
    以cv2风格为准：
    get_image_shape: (h, w)
    size: (w, h)
"""


class Resize(object):
    """ Resize img & seg.
    等比例缩放.
            1) 若size为int,
                mode为'big'时，以输入图最短边作为缩放依据,长边等比例缩放，输出图较大(default);
                mode为'small'时，以输入图最长边作为缩放依据,短边等比例缩放，输出图较小;
            2) 若size为(w, h),输出图非等比例缩放至预定大小.
    :param img: PIL Image or Numpy Image.
    :param size: (sequence or int)
    :param interpolation: 'linear' or 'nearest', default='linear'
    :param mode: 'big' or 'small', default='big'
    :return: Resized image.
    example:
        ret = resize(img, size) # img, (w, h) = (100, 200)
        size = 50, ret, (w, h) = (50, 100);
        size = 200, ret, (w, h) = (200, 400);
        size = (30, 50), ret, (w, h) = (30, 50).
    """

    def __init__(self, size, interpolation="linear", mode="big"):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.mode = mode
        self.interpolation = interpolation

    def __call__(self, result):
        meta = dict()
        result[TS_IMG] = F.resize(result[TS_IMG], self.size, self.interpolation, mode=self.mode, meta=meta)
        print("Resize meta:", meta)

        if TS_SEG in result:
            result[TS_SEG] = F.resize(result[TS_SEG], self.size, "nearest", mode=self.mode)
        if TS_BOX in result:
            nh, nw = meta["shape"]
            oh, ow = meta["org_shape"]
            w_ratio, h_ratio = nw / ow, nh / oh
            for idx, gt_box in enumerate(result[TS_BOX]):
                [cls, x1, y1, x2, y2] = gt_box
                result[TS_BOX][idx] = [cls, x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio]

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    """ Resize img & seg.
    为图像增加padding
    :param img: Numpy Image or PIL Image.
    :param padding: int or tuple,几种形式:
            padding=50, 则等价于,  pad_left = pad_right = pad_top = pad_bottom = padding
            padding=(50, 100), 则等价于, (pad_left=pad_right=50, pad_top=pad_bottom=100)
            padding=(50, 100, 150, 200),等价于(pad_left, pad_top, pad_right, pad_bottom)
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
        print("Pad meta:", meta)

        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.pad(result[TS_SEG], self.padding, ignore_label,
                                   padding_mode="constant")
        if TS_BOX in result:
            [pad_left, pad_top, _, _] = meta["padding"]
            for idx, gt_box in enumerate(result[TS_BOX]):
                [cls, x1, y1, x2, y2] = gt_box
                result[TS_BOX][idx] = [cls, x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top]
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1})'. \
            format(self.padding, self.fill)


class CenterCrop(object):
    """ Resize img & seg.
    中心剪裁，若剪裁区域大于图像区域，补Padding
    :param size：  (w, h) or int
    """

    def __init__(self, size, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, result):
        meta = dict(org_shape=get_image_shape(result[TS_IMG]))
        result[TS_IMG] = F.center_crop(result[TS_IMG], self.size, fill=self.fill,
                                       padding_mode=self.padding_mode, meta=meta)
        print("CenterCrop meta:", meta)

        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.center_crop(result[TS_SEG], self.size, fill=ignore_label,
                                           padding_mode="constant")
        return result

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
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.rotate(result[TS_SEG], angle, self.center, ignore_label, "nearest", self.expand)

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
    def __init__(self, distortion_scale=1 / 8, p=1, dsize=None, interpolation="linear", fill=0):
        self.distortion_scale = distortion_scale
        self.p = p
        self.dsize = dsize
        self.interpolation = interpolation
        self.fill = fill

    @staticmethod
    def get_params(width, height, distortion_scale):
        t_height = int(height // 4)
        t_width = int(width // 4)

        start_pts = [(t_width, t_height), (t_width * 3, t_height), (t_width * 2, t_height * 3)]

        def range_size(x, scale):
            return x + random.randint(0, int(scale)) * 2 - scale

        scale_w = int(distortion_scale * width)
        scale_h = int(distortion_scale * height)

        end_pts = [(range_size(start_pts[0][0], scale_w), range_size(start_pts[0][1], scale_h)),
                   (range_size(start_pts[1][0], scale_w), range_size(start_pts[1][1], scale_h)),
                   (range_size(start_pts[2][0], scale_w), range_size(start_pts[2][1], scale_h))]
        return np.array(start_pts, dtype=np.float32), np.array(end_pts, dtype=np.float32)

    def __call__(self, result):
        if torch.rand(1) > self.p:
            return result

        height, width = get_image_shape(result[TS_IMG])
        start_pts, end_pts = self.get_params(height, width, self.distortion_scale)

        result[TS_IMG] = F.affine(result[TS_IMG], start_pts, end_pts, dsize=self.dsize,
                                  interpolation=self.interpolation, fill=self.fill)

        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.affine(result[TS_SEG], start_pts, end_pts, dsize=self.dsize,
                                      interpolation="nearest", fill=ignore_label)

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomPerspective(object):
    """ 按概率随机执行透视变换，支持Numpy Image or PIL Image
    :param distortion_scale:
        example：RandomPerspective(distortion_scale=0.5, p=1)
    """

    def __init__(self, distortion_scale=0.2, p=0.5, dsize=None, interpolation="linear", fill=0):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.fill = fill
        self.dsize = dsize

    def __call__(self, result):
        if torch.rand(1) > self.p:
            return result

        height, width = get_image_shape(result[TS_IMG])
        start_pts, end_pts = self.get_params(width, height, self.distortion_scale)

        result[TS_IMG] = F.perspective(result[TS_IMG], start_pts, end_pts, dsize=self.dsize,
                                       interpolation=self.interpolation, fill=self.fill)

        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.perspective(result[TS_SEG], start_pts, end_pts, dsize=self.dsize,
                                           interpolation="nearest", fill=ignore_label)

        return result

    @staticmethod
    def get_params(width, height, distortion_scale):
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """ 随机剪裁
    :param size: sequence(w, h) or int
    :param padding: int, [left/right, top/bottom], [left, top, right, bottom]
    :param fill:
    :param padding_mode: 'constant',
        example:  trans = RandomCrop((768, 500), pad_if_needed=True)
    """

    def __init__(self, size, fill=0, padding_mode="constant"):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        h, w = get_image_shape(img)
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        # top, left, height, width
        top = random.randint(0, h - th) if h > th else (h - th) // 2
        left = random.randint(0, w - tw) if w > tw else (w - tw) // 2
        return top, left, th, tw

    def __call__(self, result):
        # print("org img shape:", result[TS_IMG].shape)
        top, left, height, width = self.get_params(result[TS_IMG], self.size)
        # print("top, left, height, width:", top, left, height, width)

        result[TS_IMG] = F.crop(result[TS_IMG], top, left, height, width,
                                fill=self.fill, padding_mode=self.padding_mode)
        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.crop(result[TS_SEG], top, left, height, width,
                                    fill=ignore_label, padding_mode="constant")

        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomResizedCrop(object):
    """ 裁剪给定的PIL图像到随机大小和高宽比， 不会增加黑边
    :param size: 输出的图像尺寸
    :param scale, ratio: 与目标检测时anchor生成参数类似，决定剪裁的宽高
    :param interpolation: "linear",
    """

    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation="linear",
                 fill=0, padding_mode='constant'):

        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, scale, ratio):
        height, width = get_image_shape(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return top, left, h, w

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
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w

    def __call__(self, result):
        top, left, h, w = self.get_params(result[TS_IMG], self.scale, self.ratio)

        result[TS_IMG] = F.resized_crop(result[TS_IMG], top, left, h, w, self.size, self.interpolation,
                                        fill=self.fill, padding_mode=self.padding_mode)

        if TS_SEG in result:
            ignore_label = result[TS_IGNORE_LABEL]
            result[TS_SEG] = F.resized_crop(result[TS_SEG], top, left, h, w, self.size, "nearest",
                                            fill=ignore_label, padding_mode="constant")

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        return format_string


######
class ResizedCrop(object):
    """ Resize img & seg.
    等比例缩放.
            1) 若size为int,
                mode为'big'时，以输入图最短边作为缩放依据,长边等比例缩放，输出图较大(default);
                mode为'small'时，以输入图最长边作为缩放依据,短边等比例缩放，输出图较小;
            2) 若size为(w, h),输出图非等比例缩放至预定大小.
    :param img: PIL Image or Numpy Image.
    :param size: (sequence or int)
    :param interpolation: 'linear' or 'nearest', default='linear'
    :param mode: 'big' or 'small', default='big'
    :return: Resized image.
    example:
        ret = resize(img, size) # img, (w, h) = (100, 200)
        size = 50, ret, (w, h) = (50, 100);
        size = 200, ret, (w, h) = (200, 400);
        size = (30, 50), ret, (w, h) = (30, 50).
    """

    def __init__(self, size, interpolation="linear", mode="small"):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.mode = mode
        self.interpolation = interpolation

    def __call__(self, result):
        result = Resize(self.size, self.interpolation, mode=self.mode)(result)
        img_h, img_w = get_image_shape(result[TS_IMG])

        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
        if img_h > img_w:
            pad_left = (img_h - img_w) // 2
            pad_right = img_h - img_w - pad_left
        else:
            pad_top = (img_w - img_h) // 2
            pad_bottom = img_w - img_h - pad_top
        result = Pad((pad_left, pad_top, pad_right, pad_bottom))(result)
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
