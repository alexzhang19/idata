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
from idata.augment import quality as F
from idata.augment.utils import get_image_shape
from idata.utils.type import *

__all__ = ["MotionBlur", "GaussBlur", "GaussNoise", "AdjustCompress"]


class MotionBlur(object):
    """
    图像运动模糊，degree越大，模糊程度越高
    """

    def __init__(self, p=1, degree=10, angle=None):
        """
        :param degree: int
        :param angle: int, None
        """

        self.p = p
        self.degree = degree
        self.angle = angle

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        degree = np.random.randint(2, self.degree + 1)
        angle = np.random.randint(0, 360)

        result[TS_IMG] = F.motion_blur(result[TS_IMG], degree, angle)
        result[TS_META][self.__class__.__name__.lower()] = \
            dict(p=self.p, degree=degree, angle=angle)
        return result


class GaussBlur(object):
    """
    对焦模糊，degree越大，模糊程度越高
    """

    def __init__(self, p=1, degree=7, sigmaX=0, sigmaY=0):
        """
        :param degree: int
        :param angle: int, None
        """

        self.p = p
        self.degree = degree
        self.sigmaX = 100 if sigmaX == 0 else sigmaX
        self.sigmaY = 100 if sigmaY == 0 else sigmaY

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        degree = np.random.randint(2, self.degree + 1) // 2 * 2 + 1
        sigmaX = np.random.randint(0, self.sigmaX + 1)
        sigmaY = np.random.randint(0, self.sigmaY + 1)

        result[TS_IMG] = F.gauss_blur(result[TS_IMG], degree, sigmaX, sigmaY)
        result[TS_META][self.__class__.__name__.lower()] = \
            dict(p=self.p, degree=degree, sigmaX=sigmaX, sigmaY=sigmaY)
        return result


class GaussNoise(object):
    """
    在每个像素点添加随机扰动，添加椒盐噪声
    """

    def __init__(self, p=1, degree=255):
        """
        :param degree: int
        """

        self.p = p
        self.degree = degree

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        degree = np.random.randint(0, self.degree + 1)

        result[TS_IMG] = F.gauss_noise(result[TS_IMG], degree)
        result[TS_META][self.__class__.__name__.lower()] = \
            dict(p=self.p, degree=degree)
        return result


class AdjustCompress(object):
    """
    jpeg图像质量压缩, 越大越好，[0~100]
    """

    def __init__(self, p=1, degree=30):
        """
        :param degree: int, 最小值
        """

        self.p = p
        self.degree = degree

    def __call__(self, result):
        if _random.random() > self.p:
            return result

        degree = np.random.randint(self.degree, 80)

        result[TS_IMG] = F.adjust_compress(result[TS_IMG], degree)
        result[TS_META][self.__class__.__name__.lower()] = \
            dict(p=self.p, degree=degree)
        return result


if __name__ == "__main__":
    pass
