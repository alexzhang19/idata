#!/usr/bin/env python3
# coding: utf-8

"""
@File      : enhance.py
@Author    : alex
@Date      : 2021/2/15
@Desc      :
"""

import numbers
import numpy as np
from .compose import *
import random as _random
from idata.augment import enhance as F
from idata.augment.utils import get_image_shape

__all__ = ["Gamma", "RandomGrayScale", "RandomColorScale", "ColorJitter"]


class Gamma(object):
    """
    对输入图像进行gamma校正，实现亮度变化。alfa=0.5亮，alfa=1.5暗
    use_mask使用mask分块更新
    """

    def __init__(self, p=1, alfa=None, ranges=[0.5, 1.5], use_mask=True):
        self.p = p
        self.alfa = alfa
        self.ranges = ranges
        self.random = True if alfa is None else False
        self.use_mask = use_mask

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        if self.random:
            assert len(self.ranges) == 2
            self.alfa = np.random.randint(self.ranges[0] * 100, self.ranges[1] * 100) / 100.0

        photo_mask = F.get_photo_mask(0.3, 1, 3.0) * 0.7 if self.use_mask else None
        # print("photo_mask：", np.min(photo_mask), np.max(photo_mask))
        result[TS_IMG] = F.adjust_gamma(result[TS_IMG], self.alfa, photo_mask)

        result[TS_META][self.__class__.__name__.lower()] = dict(alfa=self.alfa, p=self.p, use_mask=self.use_mask)
        return result


class RandomGrayScale(object):
    """
    returned image is 3 channel with r == g == b
    """

    def __init__(self, p=0.5, random=True):
        self.p = p
        self.random = random

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        avg = None
        if self.random:
            org_avg = int(np.mean(result[TS_IMG]))
            expend = max(min(org_avg, 20), min(20, 255 - org_avg))
            avg = int(_random.randint(max(org_avg - expend, 0), max(org_avg + expend, 0)))

        result[TS_IMG] = F.grayscale(result[TS_IMG], avg=avg)
        result[TS_META][self.__class__.__name__.lower()] = dict(
            p=self.p, random=self.random, org_avg=org_avg, avg=avg)
        return result


class RandomColorScale(object):
    """
    returned image is 3 channel with r != g != b
    """

    def __init__(self, p=0.5, expend=20):
        self.p = p
        self.expend = expend

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        avg = None
        if self.expend:
            org_avg = np.average(np.average(result[TS_IMG], axis=0), axis=0).astype(np.uint8)  # avgB, avgG, avgR
            avg = [self.adjust_avg(self.expend, v) for v in org_avg]

        result[TS_IMG] = F.grayscale(result[TS_IMG], avg=avg)
        result[TS_META][self.__class__.__name__.lower()] = dict(
            p=self.p, expend=self.expend, org_avg=org_avg, avg=avg)
        return result

    @staticmethod
    def adjust_avg(expand, org_avg):
        expand = _random.randint(-expand, expand)
        return int(org_avg + max(min(org_avg, expand), min(expand, 255 - org_avg)))


class ColorJitter(object):
    """ 随机调节：亮度、对比度、色彩饱和度、颜色
        值为None的关闭该功能
    :param brightness: 亮度, float or tuple, [1-brightness, 1+brightness] or [min, max],建议范围：[0.5, 1.5]
    :param contrast: 对比度, float or tuple, [1-contrast, 1+contrast] or [min, max],建议范围：[0.4, 1.6]
    :param saturation: 色彩饱和度, float or tuple, [1-saturation, 1+saturation] or [min, max],建议范围：[0.4, 1.6]
    :param hue: 颜色, float or tuple, [-saturation, saturation] or [min, max],建议范围：[-0.05, 0.05]
    """

    def __init__(self, p=1, brightness=[0.5, 1.5], contrast=[0.4, 1.6], saturation=[0.4, 1.6], hue=[-0.05, 0.05]):
        super().__init__()
        self.p = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if value is None:
            return None

        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, result):
        fn_idx = np.random.random(4)

        meta = dict()
        pic = result[TS_IMG]
        for fn_id, fn_p in enumerate(fn_idx):
            if fn_id == 0 and self.brightness is not None and fn_p < self.p:
                brightness = self.brightness
                brightness_factor = _random.uniform(brightness[0], brightness[1])
                pic = F.adjust_brightness(pic, brightness_factor)
                meta["brightness"] = brightness_factor

            if fn_id == 1 and self.contrast is not None and fn_p < self.p:
                contrast = self.contrast
                contrast_factor = _random.uniform(contrast[0], contrast[1])
                pic = F.adjust_contrast(pic, contrast_factor)
                meta["contrast_factor"] = contrast_factor

            if fn_id == 2 and self.saturation is not None and fn_p < self.p:
                saturation = self.saturation
                saturation_factor = _random.uniform(saturation[0], saturation[1])
                pic = F.adjust_saturation(pic, saturation_factor)
                meta["saturation_factor"] = saturation_factor

            if fn_id == 3 and self.hue is not None and fn_p < self.p:
                hue = self.hue
                hue_factor = _random.uniform(hue[0], hue[1])
                pic = F.adjust_hue(pic, hue_factor)
                meta["hue_factor"] = hue_factor

        result[TS_IMG] = pic
        result[TS_META][self.__class__.__name__.lower()] = meta
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
