#!/usr/bin/env python3
# coding: utf-8

"""
@File      : compose.py
@Author    : alex
@Date      : 2021/2/14
@Desc      : 
"""

import copy
import collections
from ..build import TRANSFORMS
from idata.augment.utils import *
from collections import OrderedDict
from idata.utils.type import *

__all__ = ["Compose", "TS_LABEL", "TS_IMG", "TS_SEG", "TS_BOX", "TS_META", "TS_ORG_SHAPE",
           "TS_IMG_SHAPE", "TS_IGNORE_LABEL"]

TS_IMG = "img"
TS_LABEL = "gt_label"  # 分类
TS_SEG = "gt_seg"  # 分割
TS_BOX = "gt_boxes"  # 检测, [[cls, x1, y1, x2, y2], ...]
TS_META = "metas"
TS_ORG_SHAPE = "org_shape"  # 原始大小
TS_IMG_SHAPE = "img_shape"  # 当前大小
TS_IGNORE_LABEL = "ignore_label"  # 分割忽略标记值，默认为255


@TRANSFORMS.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.
    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms, ignore_label=255):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        self.ignore_label = ignore_label

        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable')

    def __call__(self, data):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
           dict: Transformed data.
        """

        if data is None:
            return None

        result = dict()
        if is_numpy_image(data) or is_pil_image(data):
            result[TS_IMG] = data
        elif type(data) == dict:
            result = copy.deepcopy(data)
        else:
            # 图片以外的其它类型，真值标记类型
            for t in self.transforms:
                data = t(data)
            return data
            # raise TypeError('data must be dict or img')

        result[TS_META] = OrderedDict()
        result[TS_IGNORE_LABEL] = self.ignore_label
        for t in self.transforms:
            result = t(result)
        return result if type(data) == dict else result[TS_IMG]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
