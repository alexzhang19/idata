#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/10
@Desc      : 
"""

from idata.augment.utils import *
from idata.augment.enhance import enhance_cv as F_cv
from idata.augment.enhance import enhance_pil as F_pil

__all__ = [
    # union
    "adjust_gamma",

    # cv
    "grayscale", "his_equal_color", "img_compress",

    # pil
    "adjust_brightness", "adjust_saturation", "adjust_contrast", "adjust_sharpness",
    "adjust_hue",
]


# union
def adjust_gamma(pic, gamma: float = 0.5):
    """ 图像Gamma变换
    :param pic: Numpy or PIL image.
    :param gamma: [0.5~2]，由亮变暗。
    """

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    if _is_numpy_image(pic):
        return F_cv.adjust_gamma(pic, gamma)
    elif _is_pil_image(pic):
        return F_pil.adjust_gamma(pic, gamma)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


# cv
def grayscale(pic, random: bool = False):
    """ 相机自动白平衡-灰度世界假设, https://blog.csdn.net/dcrmg/article/details/53545510
    扩展: https://www.cnblogs.com/hangy/p/12569157.html
    returned image is 3 channel with r == g == b
    """

    if _is_numpy_image(pic):
        return F_cv.grayscale(pic, random)
    elif _is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.grayscale, random=random)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def his_equal_color(pic, limit=4.0, grid=(16, 16)):
    if _is_numpy_image(pic):
        return F_cv.his_equal_color(pic, limit=limit, grid=grid)
    elif _is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.his_equal_color, limit=limit, grid=grid)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def img_compress(pic, suffix=".png", ratio=0):
    if _is_numpy_image(pic):
        return F_cv.img_compress(pic, suffix=suffix, ratio=ratio)
    elif _is_pil_image(pic):
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

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_brightness, brightness_factor=brightness_factor)
    elif _is_pil_image(pic):
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

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_saturation, saturation_factor=saturation_factor)
    elif _is_pil_image(pic):
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

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_contrast, contrast_factor=contrast_factor)
    elif _is_pil_image(pic):
        return F_pil.adjust_contrast(pic, contrast_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_sharpness(pic, sharpness_factor):
    """ 锐度调节调节
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param sharpness_factor: 浮点型,锐度调节因子,值较小时,图像偏模糊,值较大时边缘更加锐利。建议值: [0, 3]
    :return: 对比度调整后图像
    """

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_sharpness, sharpness_factor=sharpness_factor)
    elif _is_pil_image(pic):
        return F_pil.adjust_sharpness(pic, sharpness_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def adjust_hue(pic, hue_factor):
    """ 颜色调整
    :param pic: PIL Image or Numpy Image,以PIL库实现，适配cv2
    :param hue_factor: 浮点型,[-0.5, 0.5],只对颜色产生影响,对灰度图无影响,色彩越鲜艳,改变越明显
    :return: 颜色调整后图像
    """

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.adjust_hue, hue_factor=hue_factor)
    elif _is_pil_image(pic):
        return F_pil.adjust_hue(pic, hue_factor)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))
