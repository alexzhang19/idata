#!/usr/bin/env python3
# coding: utf-8

"""
@File      : __init__.py
@Author    : alex
@Date      : 2021/2/13
@Desc      :
"""

import cv2

__all__ = ["shadow_img"]


def shadow_img(img, gray, alfa=0.3, c=0):
    h, w = img.shape[:2]
    gray = cv2.resize(gray, (w, h))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(ycrcb)
    channels[0] = cv2.addWeighted(gray, alfa, channels[0], (1 - alfa), c)
    cv2.merge(channels, ycrcb)
    return cv2.cvtColor(ycrcb, cv2.COLOR_LAB2BGR)
