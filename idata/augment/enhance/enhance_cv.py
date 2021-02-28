#!/usr/bin/env python3
# coding: utf-8

"""
@File      : enhance_cv.py
@Author    : alex
@Date      : 2021/2/10
@Desc      : 
"""

import cv2
import copy
import numpy as np
import random as _random
from idata.augment.utils import *
from typing import Any, List, Tuple, Sequence, Union

__all__ = [
    # union
    "adjust_gamma",

    # cv
    "grayscale", "his_equal_color"
]


def adjust_gamma(img, gamma: float = 0.5, photo_mask=None):
    """ 图像Gamma变换
    :param img: Numpy image.
    :param gamma: [0.5~2]，由亮变暗。
    """

    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    def gamma_gray(gray, gamma):
        if photo_mask is not None:
            if len(photo_mask.shape) != 2:
                raise TypeError("cv2 function `adjust_gamma` mask have shape of (h, w)")
            h, w = gray.shape
            gamma = cv2.resize(photo_mask, (w, h)) * gamma
        return (np.power(gray.astype(np.float16) / float(np.max(gray)), gamma) * 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    channels[0] = gamma_gray(channels[0], gamma)
    cv2.merge(channels, ycrcb)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)


def grayscale(img, avg=None):
    """ 相机自动白平衡-灰度世界假设, https://blog.csdn.net/dcrmg/article/details/53545510
    扩展: https://www.cnblogs.com/hangy/p/12569157.html
    returned image is 3 channel with r == g == b
    """

    if len(img.shape) == 2:
        return img

    img = img.astype(np.float16)
    # avgB = np.average(img[:, :, 0])
    # avgG = np.average(img[:, :, 1])
    # avgR = np.average(img[:, :, 2])
    avgB, avgG, avgR = np.average(np.average(img, axis=0), axis=0)

    if avg is not None:
        if isinstance(avg, int):
            avg_list = [avg for _ in range(3)]
        elif isinstance(avg, List) or isinstance(avg, Tuple):
            assert len(avg) == 3
            avg_list = list(avg)
        else:
            raise ValueError("avg should: None, int, list(3,), tuple(3,)")
    else:
        avg_list = [(avgB + avgG + avgR) / 3 for _ in range(3)]

    img[:, :, 0] = np.minimum(img[:, :, 0] * (avg_list[0] / avgB), 255)
    img[:, :, 1] = np.minimum(img[:, :, 1] * (avg_list[1] / avgG), 255)
    img[:, :, 2] = np.minimum(img[:, :, 2] * (avg_list[2] / avgR), 255)
    return img.astype(np.uint8)


def his_equal_color(img, limit=8.0, grid=(32, 32)):
    """ 分块直方图均衡化
    """

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    channels[0] = clahe.apply(channels[0])
    cv2.merge(channels, ycrcb)
    return cv2.cvtColor(ycrcb, cv2.COLOR_LAB2BGR)


def img_compress(img, suffix=".png", ratio=0):
    assert suffix in [".png", ".jpg"]

    if suffix == ".png":
        # 取值范围：0~9，数值越小，压缩比越低，图片质量越高
        params = [cv2.IMWRITE_PNG_COMPRESSION, ratio]  # ratio: 0~9
    elif suffix == ".jpg":
        # 取值范围：0~100，数值越小，压缩比越高，图片质量损失越严重
        params = [cv2.IMWRITE_JPEG_QUALITY, ratio]  # ratio:0~100
    else:
        raise ValueError(f"{suffix} not support compress.")

    msg = cv2.imencode(suffix, img, params)[1]
    msg = (np.array(msg)).tostring()
    img = cv2.imdecode(np.fromstring(msg, np.uint8), cv2.IMREAD_COLOR)
    return img


def random_color(src, dh=0, ds=0, dv=0):
    img = copy.deepcopy(src)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)
    dh = int(dh)
    ds = int(ds)
    dv = int(dv)

    if dh > 0:
        h = cv2.add(h, dh)
    else:
        h = cv2.subtract(h, dh)
    if ds > 0:
        s = cv2.add(s, ds)
    else:
        s = cv2.subtract(s, ds)

    if dv > 0:
        v = cv2.add(v, dv)
    else:
        v = cv2.subtract(v, dv)

    cv2.merge([h, s, v], img)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
