#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import platform
import os.path as osp
from easydict import EasyDict as edict

from .simulate import ShapeData, ShapeType
from idata.datasets.classify import CIFAR100, ClsData, ClsDataVis
from .detect.yolo import YoloData

__all__ = [
    "DCFG", "ShapeData", "ShapeType",
    "CIFAR100", "ClsData", "ClsDataVis",
    "YoloData",
]

_root_dir = "/home/temp" if platform.system() == "Linux" else "F:/datasets/datasets"

# 数据目录
DCFG = edict(dict(
    root_dir=_root_dir,

    # 分类数据集
    cifar100=osp.join(_root_dir, "cifar100"),  # CIFAR100
    cls_data=osp.join(_root_dir, "../classify"),  # ClsData
    cls_data_vis=osp.join(_root_dir, "classify_vis"),  # ClsDataVis,

    # 分割数据集
    segment=osp.join(_root_dir, "../segment"),  # Segment

))
