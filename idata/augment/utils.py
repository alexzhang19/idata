#!/usr/bin/env python3
# coding: utf-8

"""
@File      : utils.py
@Author    : alex
@Date      : 2020/6/17
@Desc      :
"""

import cv2
import sys
import torch
import numbers
import warnings
import numpy as np
import collections
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
    "_is_pil_image", "_is_numpy_image", "_is_tensor_image",
    "_get_image_size", "_is_numpy", "_interpolation_to_str",
    "decorator_pil_img", "decorator_np_img", "np_to_pil", "pil_to_np",
]

_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _is_pil_image(pic):
    if accimage is not None:
        return isinstance(pic, (Image.Image, accimage.Image))
    else:
        return isinstance(pic, Image.Image)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _get_image_size(img):
    """ 获取图像尺寸
    :param img: PIL Image or Numpy Image or Tensor
    :return: (h, w)
    """

    if _is_pil_image(img):  # (w,h)
        return img.size[::-1]
    elif _is_numpy_image(img):  # (h,w,c)
        return img.shape[:2]
    elif isinstance(img, torch.Tensor) and img.dim() > 2:  # (b, c, h, w)
        return img.shape[-2:]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def decorator_pil_img(pic, func, **kwargs):
    if not _is_pil_image(pic):
        raise TypeError("Unexpected type {}".format(type(pic)))

    np_img = np.array(pic)
    np_img.flags.writeable = True  # 将数组改为读写模式

    np_img = func(np_img, **kwargs)

    pil_img = Image.fromarray(np.uint8(np_img), mode="RGB")
    return pil_img


def decorator_np_img(pic, func, **kwargs):
    if not _is_numpy_image(pic):
        raise TypeError("Unexpected type {}".format(type(pic)))

    pil_img = np_to_pil(pic)

    pil_img = func(pil_img, **kwargs)

    np_img = np.array(pil_img)
    np_img.flags.writeable = True  # 将数组改为读写模式
    return np_img


def np_to_pil(pic, mode = None):
    if _is_pil_image(pic):
        return pic

    if not isinstance(pic, np.ndarray):
        raise TypeError('pic should be ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def pil_to_np(pic):
    if _is_numpy_image(pic):
        return pic

    np_img = np.array(pil_img)
    np_img.flags.writeable = True  # 将数组改为读写模式
    return np_img
