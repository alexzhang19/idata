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
from ..utils import _is_pil_image
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

    if not _is_pil_image(img):
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
    meta["padding"] = dict(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)

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
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def hflip(img):
    """ 对Image进行水平翻转
    :param img: PIL  Image
    :return: 水平翻转后图像
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """ 对Image进行垂直翻转
    :param img: PIL Image
    :return: 垂直翻转后图像
    """

    if not _is_pil_image(img):
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

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img, '5.2.0')
    return img.rotate(angle, interpolation, expand, center, **opts)


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """ 对图像进行仿射变换，保持图像中心不变性，图像旋转,平移，缩放，错切
        初值： affine(img, 0, (0, 0), 1, (0, 0))
    :param img: PIL Image to be rotated.
    :param angle: (float or int)，旋转角度在-180和180之间的角度，顺时针方向。
    :param translate: 水平和垂直平移量
    :param scale: 输出图的放缩倍数
    :param shear: 切变角度值在-180到180度之间，顺时针方向。【x, y】方向错切
    :param resample: PIL 插值方式
    :param fillcolor: 填充颜色值
    :return:
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0,
             -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if int(PILLOW_VERSION.split('.')[0]) >= 5 else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC, fill=None):
    """ 图像透视变换
    :param img: PIL Image
    :param startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the original image
    :param endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
    :param interpolation: Default- Image.BICUBIC
    :param fill: 填充值
    :return: 变换后图像
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    def _get_perspective_coeffs(startpoints, endpoints):
        matrix = []
        for p1, p2 in zip(endpoints, startpoints):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = torch.tensor(matrix, dtype=torch.float)
        B = torch.tensor(startpoints, dtype=torch.float).view(8)
        res = torch.lstsq(B, A)[0]
        return res.squeeze_(1).tolist()

    opts = _parse_fill(fill, img, '5.0.0')
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation, **opts)
