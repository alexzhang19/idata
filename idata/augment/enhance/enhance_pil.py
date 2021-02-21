#!/usr/bin/env python3
# coding: utf-8

"""
@File      : enhance_pil.py
@Author    : alex
@Date      : 2021/2/10
@Desc      : 
"""

import numpy as np
from idata.augment.utils import *
from PIL import Image, ImageEnhance

__all__ = [
    "adjust_gamma",
    "adjust_brightness", "adjust_saturation", "adjust_contrast", "adjust_sharpness",
    "adjust_hue",
]


def adjust_gamma(pil_img, gamma, gain=1):
    """ 图像Gamma变换
      :param pil_img: PIL image.
      :param gamma: [0.5~2]，由亮变暗。
   """

    if not _is_pil_image(pil_img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(pil_img)))
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    input_mode = pil_img.mode
    pil_img = pil_img.convert('RGB')
    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    pil_img = pil_img.point(gamma_map)  # use PIL's point-function to accelerate this part
    pil_img = pil_img.convert(input_mode)
    return pil_img


def adjust_brightness(img, brightness_factor):
    """ 图像亮度调节,实现机制与Gamma矫正类似,只是参数意义不同.
    :param img: PIL Image
    :param brightness_factor: 浮点型,亮度调节因子,为0时返回黑图,为1时返回原图.建议值: [0.5, 1.5]
    :return: 亮度调整后图像
    """

    print("brightness_factorggggggggggg: ", brightness_factor)
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """ 色彩饱和度调节
    :param img: PIL Image
    :param saturation_factor: 浮点型,色彩饱和度调节因子,为0时返回灰度图,为1时返回原图,2时返回2倍色彩.建议值: [0.4, 1.6]
        值较小时,图像颜色降低,值较大时图像颜色更鲜艳。
    :return: 色彩饱和度调整后图像
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """ 对比度调节
    :param img: PIL Image
    :param contrast_factor: 浮点型,对比度调节因子,为0时返回灰度图,为1时返回原图,2时返回2倍对比度.建议值: [0.4, 1.6]
            值较小时,由雾的感觉,值较大时更清晰,能简单去雾。
    :return: 对比度调整后图像
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_sharpness(img, sharpness_factor):
    """ 锐度调节调节
    :param img: PIL Image
    :param sharpness_factor: 浮点型,锐度调节因子,值较小时,图像偏模糊,值较大时边缘更加锐利。建议值: [0, 3]
    :return: 对比度调整后图像
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    return img


def adjust_hue(img, hue_factor):
    """ 颜色调整
    :param img: PIL Image
    :param hue_factor: 浮点型,[-0.5, 0.5],只对颜色产生影响,对灰度图无影响,色彩越鲜艳,改变越明显
    :return: 颜色调整后图像
    """

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img
