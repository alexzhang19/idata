#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 
"""

from .cifar import CIFAR100
from .classify import ClsData
from .classify_vis import ClsDataVis

__all__ = ["CIFAR100", "ClsData", "ClsDataVis"]
