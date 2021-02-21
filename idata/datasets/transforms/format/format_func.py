#!/usr/bin/env python3
# coding: utf-8

"""
@File      : format_func.py
@Author    : alex
@Date      : 2020/6/17
@Desc      :
"""

import cv2
import sys
import torch
import numpy as np
import collections
from PIL import Image
from idata.augment.utils import *

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


def to_tensor(pic):
    """ 将一个PIL Image或者numpy.ndarray转为tensor.增加对Numpy的支持
    :param pic: pic (PIL Image or numpy.ndarray): 格式：(C x H x W)，值域：[0, 255].
    :return: Tensor: 格式：(C x H x W)，值域：[0.0, 1.0].
    """

    if _is_tensor_image(pic):
        return pic

    if not (_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    # 增加对Numpy的支持
    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def to_numpy(pic, trans_to_bgr: bool = False):
    """将一个PIL Image或者Tensor转为Numpy.
    :param pic: 输入图像矩阵.
        PIL转换：默认其通道为(h, w, c)，值域[0~255]；将PIL转为Numpy格式，；
        Tensor转换：默认其通道为(c, h, w)，值域[0.0~1.0]；调整通道顺序为(h, w, c)，将Tensor转为Numpy格式，并×255；
    :param trans_to_bgr: 为True时，将RGB格式,输出为RGB
    :return: Numpy pic
    """

    if _is_pil_image(pic):  # (h, w, c)
        img = np.array(pic)
        img.flags.writeable = True  # 将数组改为读写模式
    elif _is_tensor_image(pic):  # (c, h, w)
        img = np.transpose(pic.numpy(), (1, 2, 0)) * 255
    elif _is_numpy_image(pic):  # (h, w, c)
        img = pic
    else:
        raise AttributeError("pic type not support.")
    if trans_to_bgr:
        img = rgb2bgr(img)
    return img.astype(np.uint8)


def rgb2bgr(pic):
    if _is_numpy_image(pic):
        pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    elif _is_pil_image(pic):
        pic = pic.convert("BGR")
    else:
        raise TypeError('pic should be pil or numpy. Got {}.'.format(type(pic)))
    return pic


def bgr2rgb(pic):
    if _is_numpy_image(pic):
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    elif _is_pil_image(pic):
        pic = pic.convert("BGR")
    else:
        raise TypeError('pic should be pil or numpy. Got {}.'.format(type(pic)))
    return pic


def to_pil_image(pic, mode=None):
    """将一个Numpy或者Tensor转为PIL.Image
       pic: 输入图像矩阵.
           Numpy转换：默认其通道为(h, w, c)，值域[0~255]；将PIL转为Numpy格式， 保持取值范围不变；
           Tensor转换：默认其通道为(c, h, w)，值域[0.0~1.0]；调整通道顺序为(h, w, c)，将Tensor转为PIL.Image格式，并×255；
    """

    if _is_pil_image(pic):
        return pic

    if not (isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, torch.FloatTensor) and mode != 'F':
        pic = pic.mul(255).byte()
    if isinstance(pic, torch.Tensor):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

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


def normalize(tensor, mean, std, inplace=False):
    """ 对Tensor进行归一化处理, tensor = (tensor -mean) / std.
    :param tensor: Tensor image of size (C, H, W) to be normalized.
    :param mean, std: 每个通道的均值,方差
    :param inplace: 是否使用in-place操作
    :return: Normalized Tensor image
    """

    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


def unnormalize(tensor, mean, std, inplace=False):
    """ normalize的逆操作
    """

    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor
