#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/13
@Desc      :
"""

import cv2
import sys
import PIL
import torch
import numbers
import collections
from torch import Tensor
from idata.augment.utils import *
from idata.augment.affine import affine_cv as F_cv
from idata.augment.affine import affine_pil as F_pil
from idata.augment.affine import affine_pil as F_t

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ["resize", "pad", "crop", "center_crop", "hflip", "vflip", "rotate", "five_crop"]

_cv_inter_str = dict(
    linear=cv2.INTER_LINEAR,
    nearest=cv2.INTER_NEAREST,
)

_pil_inter_str = dict(
    linear=PIL.Image.BILINEAR,
    nearest=PIL.Image.NEAREST,
)


def resize(pic, size, interpolation="linear", meta=dict()):
    """ 等比例缩放.
            1) 若size为int,以输入图最短边作为缩放依据,长边等比例缩放;
            2) 若size为(w, h),输出图非等比例缩放至预定大小.
    :param pic: Numpy or PIL Image.
    :param size: (sequence or int)
    :param interpolation: 'linear' or 'nearest'
    :return: Resized image.
    example:
        ret = resize(img, size) # img, (w, h) = (100, 200)
        size = 50, ret, (w, h) = (50, 100);
        size = 200, ret, (w, h) = (200, 400);
        size = (30, 50), ret, (w, h) = (30, 50).
    """

    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if interpolation not in ['linear', 'nearest']:
        raise TypeError(
            'interpolation should in [`linear`, `nearest`], Got interpolation string: {}'.format(interpolation))

    if isinstance(size, int):
        h, w = _get_image_size(pic)
        if (w <= h and w == size) or (h <= w and h == size):
            return pic
        elif w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        size = (ow, oh)
    meta["size"] = size

    if _is_numpy_image(pic):
        return cv2.resize(pic, size, interpolation=_cv_inter_str[interpolation])  # size[::-1]
    elif _is_pil_image(pic):
        return pic.resize(size, _pil_inter_str[interpolation])
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def pad(pic, padding, fill=0, padding_mode='constant', meta=dict()):
    """ 为图像增加padding
    :param img: Numpy or PIL Image.
    :param padding: int or tuple,几种形式:
            padding=50, 则等价于,  pad_left = pad_right = pad_top = pad_bottom = 50
            padding=(50, 100), 则等价于, pad_left = pad_right = 50, pad_top = pad_bottom = 100
    :param fill: borderType为cv2.BORDER_CONSTANT时有效, 设置填充的像素值
    :param padding_mode: 持四种边界扩展方式
            `constant`: 添加有颜色的常数值边界,还需要下一个参数(value),由fill提供.
            `reflect`: 边界元素的镜像.cv2.BORDER_DEFAULT与之类似.
            `edge`: 重复最后一个元素.
            `wrap`: 镜像位置互换.
    :return: Padding image
    """

    _str_pad_model = dict(
        constant=cv2.BORDER_CONSTANT,  # 添加有颜色的常数值边界,还需要下一个参数(value),由fill提供.
        edge=cv2.BORDER_REPLICATE,  # 重复最后一个元素.
        reflect=cv2.BORDER_REFLECT,  # 边界元素的镜像.cv2.BORDER_DEFAULT与之类似.
        wrap=cv2.BORDER_WRAP,  # 镜像位置互换.
    )

    assert padding_mode in _str_pad_model.keys(), \
        'Padding mode should be either constant, edge, reflect or wrap'

    if _is_numpy_image(pic):
        return F_cv.pad(pic, padding, fill, _str_pad_model[padding_mode], meta=meta)
    elif _is_pil_image(pic):
        return F_pil.pad(pic, padding, fill, padding_mode, meta=meta)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def crop(pic, top, left, height, width):
    """ 调用PIL Image剪切函数，超过图像区域，补充像素为0的Padding，适配cv2
        超出剪裁区域自动补充黑边。
    """

    if _is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.crop, top=top, left=left, height=height, width=width)
    elif _is_pil_image(pic):
        return F_pil.crop(pic, top, left, height, width)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def center_crop(pic, output_size, meta=dict()):
    """ 剪切PIL Image，并缩放至output_size尺寸.(w, h)
    """

    if not (_is_numpy_image(pic) or _is_pil_image(pic)):
        raise TypeError('img should be Numpy or PIL Image. Got {}'.format(type(pic)))

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_height, image_width = _get_image_size(pic)
    crop_width, crop_height = output_size
    # print("image_width, image_height", image_width, image_height)
    # print("crop_width, crop_height", crop_width, crop_height)

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    meta["crop"] = dict(
        left=crop_left,
        top=crop_top,
        width=crop_width,
        height=crop_height
    )
    return crop(pic, crop_top, crop_left, crop_height, crop_width)


def hflip(pic: Tensor) -> Tensor:
    """ 对Image进行水平翻转
    :param pic: Tensor Image，[C, H, W]； Numpy or PIL Image
    :return: 水平翻转后图像
    """

    if _is_numpy_image(pic):
        return F_cv.hflip(pic)
    elif _is_pil_image(pic):
        return F_pil.hflip(pic)
    elif isinstance(pic, torch.Tensor):
        return F_t.hflip(pic)
    else:
        raise TypeError('pic should be Numpy/PIL/Tensor Image. Got {}'.format(type(pic)))


def vflip(pic: Tensor) -> Tensor:
    """ 对Image进行垂直翻转
    :param pic: Tensor Image，[C, H, W]； Numpy or PIL Image
    :return: 垂直翻转后图像
    """

    if _is_numpy_image(pic):
        return F_cv.vflip(pic)
    elif _is_pil_image(pic):
        return F_pil.vflip(pic)
    elif isinstance(pic, torch.Tensor):
        return F_t.vflip(pic)
    else:
        raise TypeError('pic should be Numpy/PIL/Tensor Image. Got {}'.format(type(pic)))


def rotate(pic, angle, center=None, fill=None, interpolation="linear", expand=False):
    """ 图像旋转
    :param pic: Numpy or PIL Image
    :param angle: 旋转角度，0~360
    :param center: 旋转中心点，默认为图像中心
    :param scale: 缩放倍数
    :param fill: 填充值
    :param interpolation: 插值方式， `linear`、`nearest`
    :param expand: 自动扩充边界
    :return:
    """

    if _is_numpy_image(pic):
        return F_cv.rotate(pic, angle, center=center, fill=fill,
                           interpolation=_cv_inter_str[interpolation], expand=expand)
    elif _is_pil_image(pic):
        return F_pil.rotate(pic, angle, center=center, fill=fill,
                            interpolation=_pil_inter_str[interpolation], expand=expand)
    else:
        raise TypeError('pic should be Numpy/PIL/Tensor Image. Got {}'.format(type(pic)))


def five_crop(pic, size):
    """Crop the given Numpy or PIL Image into four corners and the central crop.
        size: (w, h)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_height, image_width = _get_image_size(pic)
    crop_width, crop_height = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(pic, 0, 0, crop_height, crop_width)
    tr = crop(pic, 0, image_width - crop_width, crop_height, image_width)
    bl = crop(pic, image_height - crop_height, 0, image_height, crop_width)
    br = crop(pic, image_height - crop_height, image_width - crop_width,
              image_height, image_width)
    center = center_crop(pic, (crop_width, crop_height))
    return [tl, tr, bl, br, center]
