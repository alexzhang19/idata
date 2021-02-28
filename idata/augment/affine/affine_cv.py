#!/usr/bin/env python3
# coding: utf-8

"""
@File      : affine_cv.py
@Author    : alex
@Date      : 2021/2/13
@Desc      : 
"""

import cv2
import sys
import PIL
import numbers
import collections
import numpy as np
from idata.augment.utils import *

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["pad", "hflip", "vflip", "rotate"]


def _parse_fill(fill, img):
    assert len(img.shape) in [2, 3], "len(img.shape) should in [2, 3]."

    shape = img.shape
    fill = 0 if fill is None else fill
    if len(shape) == 3 and isinstance(fill, (int, float)):
        fill = tuple(fill for _ in range(3))
    return fill


def pad(img, padding, fill=0, padding_mode=cv2.BORDER_CONSTANT, meta=dict()):
    """ 为图像增加padding
    :param img: Numpy Image.
    :param padding: int or tuple,几种形式:
            padding=50, 则等价于,  pad_left = pad_right = pad_top = pad_bottom = 50
            padding=(50, 100), 则等价于, pad_left = pad_right = 50, pad_top = pad_bottom = 100
    :param fill: borderType为cv2.BORDER_CONSTANT时有效, 设置填充的像素值
    :param padding_mode: 持四种边界扩展方式
            cv2.BORDER_CONSTANT: 添加有颜色的常数值边界,还需要下一个参数(value),由fill提供.
            cv2.BORDER_REFLECT: 边界元素的镜像.cv2.BORDER_DEFAULT与之类似.
            cv2.BORDER_REPLICATE: 重复最后一个元素.
            cv2.BORDER_WRAP: 镜像位置互换.
    :return: Padding image
    """

    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple, list)):
        raise TypeError('Got inappropriate fill arg')
    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))
    assert padding_mode in [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_DEFAULT,
                            cv2.BORDER_REPLICATE, cv2.BORDER_WRAP], \
        'Padding mode should be correct cv2 type'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if cv2.BORDER_CONSTANT == padding_mode:
        if len(img.shape) == 3 and type(fill) == int:
            fill = [fill, fill, fill]

    meta["padding"] = dict(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)

    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=padding_mode, value=fill)


def hflip(img):
    """ 对Image进行水平翻转
    :param img: Numpy Image
    :return: 水平翻转后图像
    """

    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))
    return cv2.flip(img, 1)


def vflip(img):
    """ 对Image进行垂直翻转
    :param img: Numpy Image
    :return: 垂直翻转后图像
    """

    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))
    return cv2.flip(img, 0)


def rotate(img, angle, center=None, scale=1.0, fill=0, interpolation=cv2.INTER_LINEAR, expand=True):
    """ 图像旋转
    :param img: Numpy Image
    :param angle: 旋转角度，0~360
    :param center: 旋转中心点，默认为图像中心
    :param scale: 缩放倍数
    :param fill: 填充值
    :param interpolation: 插值方式， cv2.INTER_LINEAR、cv2.INTER_NEAREST
    :param expand: 自动扩充边界
    :return:
    """

    if center is not None and expand:
        raise ValueError('`auto_bound` conflicts with `center`')

    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    if expand:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=interpolation,
        borderValue=fill)
    return rotated


def affine(img, start_points, end_points, dsize=None, interpolation=cv2.INTER_LINEAR, fill=0):
    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    start_points = np.array(start_points, np.float32)
    end_points = np.array(end_points, np.float32)
    assert start_points.shape == end_points.shape and start_points.shape == (3, 2), "points shape should like (3,2)."

    dsize = get_image_shape(img)[::-1] if dsize is None else dsize

    fill = _parse_fill(fill, img)

    matrix = cv2.getAffineTransform(start_points, end_points)
    return cv2.warpAffine(img, matrix, dsize, flags=interpolation, borderValue=fill)


def perspective(img, start_points, end_points, dsize=None, interpolation=cv2.INTER_LINEAR, fill=0):
    if not is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    start_points = np.array(start_points, np.float32)
    end_points = np.array(end_points, np.float32)
    assert start_points.shape == end_points.shape and start_points.shape == (4, 2), "points shape should like (4,2)."

    dsize = get_image_shape(img)[::-1] if dsize is None else dsize

    fill = _parse_fill(fill, img)

    matrix = cv2.getPerspectiveTransform(start_points, end_points)
    return cv2.warpPerspective(img, matrix, dsize, flags=interpolation, borderValue=fill)
