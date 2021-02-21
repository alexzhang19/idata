#!/usr/bin/env python3
# coding: utf-8

"""
@File      : affine_tensor.py
@Author    : alex
@Date      : 2020/6/20
@Desc      :
"""

from torch import Tensor

__all__ = ["_is_tensor_a_torch_image", "hflip", "vflip"]


def _is_tensor_a_torch_image(input):
    return input.ndim >= 2


def hflip(img: Tensor) -> Tensor:
    """ 对Image Tensor进行水平翻转
    :param img: Tensor Image，[C, H, W].
    :return: Horizontally flipped image Tensor.
    """

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-1)


def vflip(img: Tensor) -> Tensor:
    """ 对Image Tensor进行垂直翻转
    :param img: Tensor Image，[C, H, W].
    :return: Vertically flipped image Tensor.
    """

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-2)
