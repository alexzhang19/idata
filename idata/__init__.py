#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import os

__all__ = ["__version__", "PKG_DIR", "PRO_DIR"]

__version__ = "1.0"

# 工程目录
PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # E:\common\alcore\alcore
PRO_DIR = os.path.dirname(PKG_DIR)  # E:\common\alcore

if __name__ == "__main__":
    print("PKG_DIR:", PKG_DIR)
    print("PRO_DIR:", PRO_DIR)
