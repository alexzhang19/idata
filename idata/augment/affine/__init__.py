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

__all__ = ["resize", "pad", "crop", "center_crop", "hflip", "vflip", "rotate",
           "affine", "perspective", "five_crop"]

_cv_inter_str = dict(
    linear=cv2.INTER_LINEAR,
    nearest=cv2.INTER_NEAREST,
)

_pil_inter_str = dict(
    linear=PIL.Image.BILINEAR,
    nearest=PIL.Image.NEAREST,
)


def pad_fill(pic, ):
    pass


def resize(pic, size, interpolation="linear", mode="big", meta=dict()):
    """ 等比例缩放.
            1) 若size为int,
                mode为'big'时，以输入图最短边作为缩放依据,长边等比例缩放，输出图较大(default);
                mode为'small'时，以输入图最长边作为缩放依据,短边等比例缩放，输出图较小;
            2) 若size为(w, h),输出图非等比例缩放至预定大小.
    :param pic: Numpy or PIL Image.
    :param size: (sequence or int)
    :param interpolation: 'linear' or 'nearest'
    :param mode: 'big' or 'small', default='big'
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

    if mode not in ['big', 'small']:
        raise TypeError(
            'mode should in [`big`, `small`], Got mode string: {}'.format(mode))

    if isinstance(size, int):
        h, w = get_image_shape(pic)
        if (w <= h and w == size) or (h <= w and h == size):
            return pic
        elif mode == "big":
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
        elif mode == "small":
            if w < h:
                oh = size
                ow = int(size * h / w)
            else:
                ow = size
                oh = int(size * h / w)

        img_shape = (oh, ow)
    meta["shape"] = img_shape
    meta["org_shape"] = (h, w)

    if is_numpy_image(pic):
        return cv2.resize(pic, img_shape[::-1], interpolation=_cv_inter_str[interpolation])  # size[::-1]
    elif is_pil_image(pic):
        return pic.resize(img_shape[::-1], _pil_inter_str[interpolation])
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

    if is_numpy_image(pic):
        return F_cv.pad(pic, padding, fill, _str_pad_model[padding_mode], meta=meta)
    elif is_pil_image(pic):
        return F_pil.pad(pic, padding, fill, padding_mode, meta=meta)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def crop(pic, top, left, height, width, fill=0, padding_mode='constant'):
    """ 调用PIL Image剪切函数，超过图像区域，补充像素为0的Padding，适配cv2
        超出剪裁区域自动补充黑边。
    """

    oh, ow = get_image_shape(pic)
    t, l, h, w = top, left, height, width

    pad_left = -l if l < 0 else 0
    pad_top = -t if t < 0 else 0
    pad_right = (w - ow) - pad_left if w > ow else 0
    pad_bottom = (h - oh) - pad_top if h > oh else 0
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    if sum(list(padding)) > 0:
        # print("padding:", padding)
        pic = pad(pic, padding, fill=fill, padding_mode=padding_mode)
        left, top = left + pad_left, top + pad_top

    if is_numpy_image(pic):
        return decorator_np_img(pic, F_pil.crop, top=top, left=left, height=height, width=width)
    elif is_pil_image(pic):
        return F_pil.crop(pic, top, left, height, width)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def center_crop(pic, output_size, fill=0, padding_mode='constant', meta=dict()):
    """ 剪切PIL Image，并缩放至output_size尺寸.(w, h)
    """

    if not (is_numpy_image(pic) or is_pil_image(pic)):
        raise TypeError('img should be Numpy or PIL Image. Got {}'.format(type(pic)))

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_height, image_width = get_image_shape(pic)
    crop_width, crop_height = output_size

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    meta["crop"] = dict(left=crop_left, top=crop_top, width=crop_width, height=crop_height)

    return crop(pic, crop_top, crop_left, crop_height, crop_width, fill=fill, padding_mode=padding_mode)


def resized_crop(pic, top, left, height, width, size, interpolation="linear",
                 fill=0, padding_mode='constant'):
    if not (is_numpy_image(pic) or is_pil_image(pic)):
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))

    pic = crop(pic, top, left, height, width, fill=fill, padding_mode=padding_mode)
    return resize(pic, size, interpolation)


def hflip(pic: Tensor) -> Tensor:
    """ 对Image进行水平翻转
    :param pic: Tensor Image，[C, H, W]； Numpy or PIL Image
    :return: 水平翻转后图像
    """

    if is_numpy_image(pic):
        return F_cv.hflip(pic)
    elif is_pil_image(pic):
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

    if is_numpy_image(pic):
        return F_cv.vflip(pic)
    elif is_pil_image(pic):
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

    if is_numpy_image(pic):
        return F_cv.rotate(pic, angle, center=center, fill=fill,
                           interpolation=_cv_inter_str[interpolation], expand=expand)
    elif is_pil_image(pic):
        return F_pil.rotate(pic, angle, center=center, fill=fill,
                            interpolation=_pil_inter_str[interpolation], expand=expand)
    else:
        raise TypeError('pic should be Numpy/PIL/Tensor Image. Got {}'.format(type(pic)))


def affine(pic, start_points, end_points, dsize=None, interpolation="linear", fill=0):
    if is_numpy_image(pic):
        return F_cv.affine(pic, start_points, end_points, dsize=dsize,
                           interpolation=_cv_inter_str[interpolation], fill=fill)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.affine, start_points=start_points,
                                 end_points=end_points, dsize=dsize,
                                 interpolation=_pil_inter_str[interpolation], fill=fill)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def perspective(pic, start_points, end_points, dsize=None, interpolation="linear", fill=0):
    if is_numpy_image(pic):
        return F_cv.perspective(pic, start_points, end_points, dsize=dsize,
                                interpolation=_cv_inter_str[interpolation], fill=fill)
    elif is_pil_image(pic):
        return decorator_pil_img(pic, F_cv.perspective, start_points=start_points,
                                 end_points=end_points, dsize=dsize,
                                 interpolation=_pil_inter_str[interpolation], fill=fill)
    else:
        raise TypeError('pic should be Numpy or PIL Image. Got {}'.format(type(pic)))


def five_crop(pic, size):
    """Crop the given Numpy or PIL Image into four corners and the central crop.
        size: (w, h)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_height, image_width = get_image_shape(pic)
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
