#!/usr/bin/env python3
# coding: utf-8

"""
@File      : affine_pil.py
@Author    : alex
@Date      : 2021/2/13
@Desc      : 
"""

import math
import torch
import numbers
import numpy as np
from ..utils import is_pil_image
from idata.augment.utils import *
from collections.abc import Sequence, Iterable
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

try:
    import accimage
except ImportError:
    accimage = None

__all__ = ["pad", "crop", "hflip", "vflip", "rotate"]


def _parse_fill(fill, img, min_pil_version):
    """Helper function to get the fill color for rotate and perspective transforms.
    """
    major_found, minor_found = (int(v) for v in PILLOW_VERSION.split('.')[:2])
    major_required, minor_required = (int(v) for v in min_pil_version.split('.')[:2])
    if major_found < major_required or (major_found == major_required and minor_found < minor_required):
        if fill is None:
            return {}
        else:
            msg = ("The option to fill background area of the transformed image, "
                   "requires pillow>={}")
            raise RuntimeError(msg.format(min_pil_version))

    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if not isinstance(fill, (int, float)) and len(fill) != num_bands:
        msg = ("The number of elements in 'fill' does not match the number of "
               "bands of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_bands))
    return {"fillcolor": fill}


def pad(img, padding, fill=0, padding_mode='constant', meta=dict()):
    """ 为图像增加padding
    :param img: PIL Image.
    :param padding: int or tuple,几种形式:
            padding=50, 则等价于,  pad_left = pad_right = pad_top = pad_bottom = 50
           padding=(50, 100), 则等价于, pad_left = pad_right = 50, pad_top = pad_bottom = 100
    :param fill: padding_mode为constant时有效, 设置填充的像素值.
    :param padding_mode: 持四种边界扩展方式
            constant: 添加有颜色的常数值边界,还需要下一个参数(value),由fill提供.
            reflect: 边界元素的镜像.cv2.BORDER_DEFAULT与之类似.
            edge: 重复最后一个元素.
            symmetric: pads with reflection of image
    :return: Padding image
    """

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'wrap'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

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

    meta["org_shape"] = get_image_shape(img)
    meta["padding"] = [pad_left, pad_top, pad_right, pad_bottom]

    padding = tuple([pad_left, pad_top, pad_right, pad_bottom])
    if padding_mode == 'constant':
        if isinstance(fill, numbers.Number):
            fill = (fill,) * len(img.getbands())
        if len(fill) != len(img.getbands()):
            raise ValueError('fill should have the same number of elements '
                             'as the number of channels in the image '
                             '({}), got {} instead'.format(len(img.getbands()), len(fill)))
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, fill=fill)
    else:
        # if isinstance(padding, int):
        #     pad_left = pad_right = pad_top = pad_bottom = padding
        # if isinstance(padding, Sequence) and len(padding) == 2:
        #     pad_left = pad_right = padding[0]
        #     pad_top = pad_bottom = padding[1]
        # if isinstance(padding, Sequence) and len(padding) == 4:
        #     pad_left = padding[0]
        #     pad_top = padding[1]
        #     pad_right = padding[2]
        #     pad_bottom = padding[3]

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
        return Image.fromarray(img)


def crop(img, top, left, height, width):
    """ 调用PIL Image剪切函数，超过图像区域，补充像素为0的Padding
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def hflip(img):
    """ 对Image进行水平翻转
    :param img: PIL  Image
    :return: 水平翻转后图像
    """

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """ 对Image进行垂直翻转
    :param img: PIL Image
    :return: 垂直翻转后图像
    """

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def rotate(img, angle, center=None, fill=None, interpolation=Image.BILINEAR, expand=False):
    """ 图像旋转
    :param img: PIL Image
    :param angle: 逆时针角度，0~360
    :param resample: 可选的重新采样过滤器。这可能是PIL.Image.NEAREST（使用最近的邻居）,
            PIL.Image.BILINEAR（2x2环境中的线性插值），PIL.Image.BICUBIC（4x4环境中的三次样条插值）。
    :param expand: 可选扩展标志。如果为真，则展开输出图像，使其足够大以容纳整个旋转图像。如果为false或省略，
            则使输出图像与输入图像大小相同。请注意，展开标志假定围绕中心旋转，不进行平移。
    :param center: 可选旋转中心（2元组）。原点是左上角。默认为图像的中心。
    :param fill: 旋转图像外部区域的可选颜色。
    :return:
    """

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img, '5.2.0')
    return img.rotate(angle, interpolation, expand, center, **opts)
