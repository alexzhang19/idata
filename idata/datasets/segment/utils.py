#!/usr/bin/env python3
# coding: utf-8

"""
@File      : utils.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 分割标签转换
"""

"""
    # 通用为 uint8类型
    label = 0  # (height, width), 不同数字表类型
    color = 1  # (height, width, 3), 按颜色表示类别
    order = 2  # (cnt, height, width), 按顺序表示的目标，class_ids为类别，cnt为目标个数
    segment = 3  # (n_cls, height, width), 按类别顺序表示的目标，n_cls为类别总数
"""

import cv2
import copy
import numpy as np
from idata.fileio import *

__all__ = [
    "order_to_label", "order_to_color", "order_to_segment",
    "label_to_color",
    "segment_to_label",
    "label_write", "label_read",
]


def order_to_label(masks, class_ids, ignore_label=255):
    _, height, width = masks.shape
    label = np.ones((height, width), np.uint8) * ignore_label
    for idx, cls in enumerate(class_ids):
        block = copy.deepcopy(masks[idx]).astype(np.uint8)
        label[block != 0] = cls
    return label


def order_to_color(masks, class_ids, cls_colors, ignore_label=255):
    _, height, width = masks.shape
    label = np.ones((height, width, 3), np.uint8) * ignore_label
    for idx, cls in enumerate(class_ids):
        block = copy.deepcopy(masks[idx]).astype(np.uint8)
        label[(block != 0)] = cls_colors[cls]
    return label


def order_to_segment(masks, class_ids, n_cls):
    _, height, width = masks.shape
    segs = np.zeros([n_cls, height, width], dtype=np.uint8)
    for idx, cls in enumerate(class_ids):
        segs[cls, :, :] = cv2.bitwise_or(segs[cls, :, :], masks[idx])
    return segs


def label_to_color(label, color_maps, ignore_label=255):
    assert label.dtype == np.uint8 and len(label.shape) == 2

    height, width = label.shape
    image = np.ones((height, width, 3), dtype=np.uint8) * ignore_label
    cls_ids = np.setdiff1d(np.unique(label), np.array([ignore_label]))

    for cls in cls_ids:
        try:
            image[label == cls] = color_maps[cls]
        except Exception as err:
            print("err:", err)
    return image


def segment_to_label(segs, ignore_label=255):
    n_cls, height, width = segs.shape
    label = np.ones((height, width), np.uint8) * ignore_label
    for cls in range(n_cls):
        label[segs[cls] != 0] = cls
    return label


def label_write(file_path, mask):
    assert suffix(file_path) == ".png", "label should .png"
    cv2.imwrite(file_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def label_read(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from idata.fileio import *
    from idata.visual.visual import *
    from idata.datasets import ShapeData, ShapeType

    dataset = ShapeData(1, shape=(256, 256), type=ShapeType.normal)
    image, (masks, class_ids) = dataset[0]
    print(image.shape, masks.shape, class_ids)
    cv2.imwrite(path.join(desktop, "img.jpg"), image)

    blank = np.ones((256, 256, 3), dtype=np.uint8) * 255
    label = order_to_label(masks, class_ids)
    label_write(path.join(desktop, "label.png"), label)

    c = label_to_color(label, dataset.PALETTE)
    label_write(path.join(desktop, "c.png"), c)

    color = order_to_color(masks, class_ids, dataset.PALETTE)
    label_write(path.join(desktop, "color.png"), color)
