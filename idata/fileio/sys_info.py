#!/usr/bin/env python3
# coding: utf-8

"""
@File      : sys_info.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 获取运行环境信息
"""

import re
import cv2
import sys
import uuid
import subprocess

__all__ = ["get_mac_address", "get_free_gpus",
           "get_py_major", "get_cv2_major", "get_os_type"]


def get_mac_address():
    """
    获取本机Mac地址
    :return: Mac地址字符串
    """

    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])


def get_free_gpus(max_gpu_cnt=4):
    '''
    获取空闲GPUID
    :return: gpuids
    '''

    info = subprocess.check_output("nvidia-smi")
    proc_idx = [i.start() for i in re.finditer('Processes:', info)]
    assert len(proc_idx) == 1
    gpu_ids = []
    up_str = info[0:proc_idx[0]]
    for id in list(range(max_gpu_cnt)):
        if 0 != len([i.start() for i in re.finditer(' %d ' % int(id), up_str)]):
            gpu_ids.append(id)

    filter_str = info[proc_idx[0]:-1]
    rets = []
    for id in gpu_ids:
        if 0 == len([i.start() for i in re.finditer(' %d ' % int(id), filter_str)]):
            rets.append(str(id))
    return rets


def get_py_major():
    """
    获取python住版本号
    return: 2 or 3
    """

    return int(sys.version_info.major)


def get_cv2_major():
    """
    返回Int型主版本号
    return: 主版本号
    """

    return int(cv2.__version__.split(".")[0])


def get_os_type():
    """
    判断操作系统类型
    return: "Windows", "Linux"
    """

    import platform
    return platform.system()
