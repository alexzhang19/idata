#!/usr/bin/env python3
# coding: utf-8

"""
@File      : color.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import numpy as np

# 6类别颜色
_colors6 = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
            [0, 255, 255], [255, 0, 255], [255, 255, 0]]

# voc定义类别颜色
_colors19 = [[128, 64, 128],  # 0 - road
             [244, 35, 232],  # 1 - sidewalk
             [70, 70, 70],  # 2 - building
             [102, 102, 156],  # 3 - wall
             [190, 153, 153],  # 4 - fence
             [153, 153, 153],  # 5 - pole
             [250, 170, 30],  # 6 - traffic light
             [220, 220, 0],  # 7 - traffic sign
             [107, 142, 35],  # 8 - vegetation
             [152, 251, 152],  # 9 - terrain
             [70, 130, 180],  # 10 - sky
             [220, 20, 60],  # 11 - human
             [255, 0, 0],  # 12 - rider
             [0, 0, 142],  # 13 - car
             [0, 0, 70],  # 14 - truck
             [0, 60, 100],  # 15 - bus
             [0, 80, 100],  # 16 - train
             [0, 0, 230],  # 17 - motorcycle
             [119, 11, 32],  # 18 - bicycle]
             ]


def random_color(num: int, mode: str = "bgr") -> list:
    """
    产生颜色数组，如需增加全0通道，list.extend()方法实现.
    num: 颜色个数
    mode: 颜色格式,默认为cv2图像格式BGR
    return: 颜色数组，list[[bgr],[bgr],...]
    """

    assert mode.lower() in ["rgb", "bgr"], "mode mast in [’rgb‘, ’bgr‘]"

    def bitget(val, idx):
        return ((val & (1 << idx)) != 0)

    if num <= 6:
        colors = _colors6
    elif num <= 19:
        colors = _colors19
    else:
        colors = np.zeros((num, 3), dtype=np.uint8)
        for i in range(num):
            r = g = b = 0
            c = i
            for j in range(8):
                r |= (bitget(c, 0) << 7 - j)
                g |= (bitget(c, 1) << 7 - j)
                b |= (bitget(c, 2) << 7 - j)
                c >>= 3
            colors[i, :] = [r, g, b]
            colors.tolist()
    if mode.lower() == "bgr":
        colors = [[b, g, r] for [r, g, b] in colors]
    return colors[0:num]
