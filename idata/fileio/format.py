#!/usr/bin/env python3
# coding: utf-8

"""
@File      : format.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import time
import datetime

__all__ = [
    # time format
    "cur_time", "cur_date", "timestamp",
]


# time forma
def cur_time(format: str = None):
    """
    获取当前时间-秒
    """

    format = format if format is not None else "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.now().strftime(format)


def cur_date(format: str = None):
    """
    获取当前时间-天
    """

    format = format if format is not None else "%Y-%m-%d"
    return datetime.datetime.now().strftime(format)


def timestamp(format: str = "%Y%m%d_%H%M%S"):
    return time.strftime(format, time.localtime())
