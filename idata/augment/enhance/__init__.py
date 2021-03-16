#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/10
@Desc      : 
"""

import cv2
import copy
import numpy as np
from idata.augment.utils import *
from idata.augment.enhance import enhance_cv as F_cv
from idata.augment.enhance import enhance_pil as F_pil

__all__ = [
    # utils
    "get_photo_mask",

    # union
    "adjust_gamma",

    # cv
    "grayscale", "his_equal_color", "img_compress",

    # pil
    "adjust_brightness", "adjust_saturation", "adjust_contrast", "adjust_sharpness",
    "adjust_hue",
]


def get_photo_mask(l_val, m_val, r_val, width=12, height=12, dim=1, kernel=2, extend=False):
    if r_val <= l_val or m_val > r_val or m_val < l_val:
        raise ValueError('should r_val > m_val > l_val')

    r_mask = np.random.random((height, width, dim))
    if extend and dim > 1:
        for i in range(dim, 1):
            r_mask[:, :, i] = r_mask[:, :, 0]

    mask = copy.deepcopy(r_mask)
    mask[r_mask < 0.5] = r_mask[r_mask < 0.5] * (m_val - l_val) + l_val
    mask[r_mask >= 0.5] = r_mask[r_mask >= 0.5] * (r_val - m_val) + m_val
    # mask = np.random.random((height, width, dim)) * (r_val - l_val) + l_val

    mask = mask[:, :, 0] if dim == 1 else mask
    if kernel is not None:
        mask = cv2.blur(mask, (kernel, kernel))
    return mask


# union
def adjust_gamma(pic, gamma: float = 0.5, photo_mask=None):
    """ 图像Gamma变换
    :param pic: Numpy or PIL image.
    :param gamma: [0.5~2]，由亮变暗。
    :param photo_mask: mask模板
    """

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    if is_numpy_image(pic):
        return F_cv.adjust_gamma(pic, gamma, photo_mask=photo_mask)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.adjust_gamma, gamma=gamma, photo_mask=photo_mask)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


# cv
def grayscale(pic, avg=None):
    """ 相机自动白平衡-灰度世界假设, https://blog.csdn.net/dcrmg/article/details/53545510
    扩展: https://www.cnblogs.com/hangy/p/12569157.html
    returned image is 3 channel with r == g == b
    """

    if is_numpy_image(pic):
        return F_cv.grayscale(pic, avg=avg)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.grayscale, avg=avg)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def his_equal_color(pic, limit=4.0, grid=(8, 8)):
    if is_numpy_image(pic):
        return F_cv.his_equal_color(pic, limit=limit, grid=grid)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.his_equal_color, limit=limit, grid=grid)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def img_compress(pic, suffix=".png", ratio=0):
    if is_numpy_image(pic):
        return F_cv.img_compress(pic, suffix=suffix, ratio=ratio)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.img_compress, suffix=suffix, ratio=ratio)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


# pil
def adjust_brightness(pic, brightness_factor=1):
    """ 图像亮度调节,实现机制与Gamma矫正类似,只是参数意义不同.
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param brightness_factor: 浮点型,亮度调节因子,为0时返回黑图,为1时返回原图.建议值: [0.5, 1.5]
    :return: 亮度调整后图像
    """

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_brightness, brightness_factor=brightness_factor)
    elif is_pil_image(pic):
        return F_pil.adjust_brightness(pic, brightness_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_saturation(pic, saturation_factor=1):
    """ 色彩饱和度调节
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param saturation_factor: 浮点型,色彩饱和度调节因子,为0时返回灰度图,为1时返回原图,2时返回2倍色彩.建议值: [0.4, 1.6]
        值较小时,图像颜色降低,值较大时图像颜色更鲜艳。
    :return: 色彩饱和度调整后图像
    """

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_saturation, saturation_factor=saturation_factor)
    elif is_pil_image(pic):
        return F_pil.adjust_saturation(pic, saturation_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_contrast(pic, contrast_factor):
    """ 对比度调节
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param contrast_factor: 浮点型,对比度调节因子,为0时返回灰度图,为1时返回原图,2时返回2倍对比度.建议值: [0.4, 1.6]
            值较小时,由雾的感觉,值较大时更清晰,能简单去雾。
    :return: 对比度调整后图像
    """

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_contrast, contrast_factor=contrast_factor)
    elif is_pil_image(pic):
        return F_pil.adjust_contrast(pic, contrast_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_sharpness(pic, sharpness_factor):
    """ 锐度调节调节
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param sharpness_factor: 浮点型,锐度调节因子,值较小时,图像偏模糊,值较大时边缘更加锐利。建议值: [0, 3]
    :return: 对比度调整后图像
    """

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_sharpness, sharpness_factor=sharpness_factor)
    elif is_pil_image(pic):
        return F_pil.adjust_sharpness(pic, sharpness_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_hue(pic, hue_factor):
    """ 颜色调整
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param hue_factor: 浮点型,[-0.5, 0.5],只对颜色产生影响,对灰度图无影响,色彩越鲜艳,改变越明显
    :return: 颜色调整后图像
    """

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_hue, hue_factor=hue_factor)
    elif is_pil_image(pic):
        return F_pil.adjust_hue(pic, hue_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))
