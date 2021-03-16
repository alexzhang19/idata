#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/6
@Desc      :
"""

import sys
import collections
from idata.augment.utils import *
from idata.augment.quality import quality_cv as F_cv

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["motion_blur", "gauss_blur", "gauss_noise", "adjust_compress"]


def motion_blur(pic, degree=10, angle=20):
    """ 图像运动模糊，degree越大，模糊程度越高, cv2实现
    :param pic: Numpy Image or PIL Image
    :param degree: int, >1, 600*800，最多degree=10
    :param angle: 运动角度，可随机设置，[0,360]
    """

    if is_numpy_image(pic):
        return F_cv.motion_blur(pic, degree, angle)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.motion_blur, degree=degree, angle=angle)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def gauss_blur(pic, degree=7, sigmaX=0, sigmaY=0):
    """ 对焦模糊，degree越大，模糊程度越高
    :param degree: 大于1的奇数
    """

    if is_numpy_image(pic):
        return F_cv.gauss_blur(pic, degree, sigmaX, sigmaY)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.gauss_blur, degree=degree, sigmaX=sigmaX, sigmaY=sigmaY)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def gauss_noise(pic, degree=None):
    """ 在每个像素点添加随机扰动
    :param degree: [0,1]
    url: https://www.cnblogs.com/arkenstone/p/8480759.html
    """

    if is_numpy_image(pic):
        return F_cv.gauss_noise(pic, degree)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.gauss_noise, degree=degree)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_compress(pic, ratio):
    """ jpeg图像质量压缩
    :param ratio: [0~100], 数值越小，压缩比越高，图片质量损失越严重
    """

    if is_numpy_image(pic):
        return F_cv.adjust_compress(pic, ratio)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.adjust_compress, degree=ratio)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


if __name__ == "__main__":
    pass
