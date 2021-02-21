#!/usr/bin/env python3
# coding: utf-8

"""
@File      : type.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 
"""

import numpy as np
from typing import Any, List, Tuple, Sequence, Union

__all__ = ["is_str", "is_list", "is_tuple", "is_sequence", "is_numpy"]


def is_str(x):
    return isinstance(x, str)


def is_list(val: Any) -> bool:
    return isinstance(val, List)


def is_tuple(val: Any) -> bool:
    return isinstance(val, Tuple)


def is_sequence(val: Any) -> bool:
    """
    accept：List or Tuple
    """

    return isinstance(val, Sequence)


def is_numpy(val: Any) -> bool:
    return isinstance(val, np.ndarray)
