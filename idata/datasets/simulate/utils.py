#!/usr/bin/env python3
# coding: utf-8

"""
@File      : utils.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 分割标签转换
    url: https://blog.csdn.net/jizhidexiaoming/article/details/108714844
"""

"""
    # 通用为 uint8类型
    label = 0  # (height, width), 不同数字表类型
    color = 1  # (height, width, 3), 按颜色表示类别
    order = 2  # default, (cnt, height, width), 按顺序表示的目标，class_ids为类别，cnt为目标个数
    segment = 3  # (n_cls, height, width), 按类别顺序表示的目标，n_cls为类别总数
"""

import cv2
import copy
import numpy as np
from idata.fileio import *

__all__ = [
    "order_to_label", "order_to_color", "order_to_segment", "order_to_detect",
    "label_to_color", "label_to_segment",
    "color_to_label", "color_to_segment",
    "segment_to_label", "segment_to_color",
    "label_write", "label_read", "unique_color",
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


def order_to_detect(masks, class_ids, mini_pixel=10):
    gt_labels = []
    for cls, mask in zip(class_ids, masks):
        if np.all(0 == mask):
            continue
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x0, y0, w0, h0) = cv2.boundingRect(cnt)
            if w0 < mini_pixel or h0 < mini_pixel:
                continue
            gt_labels.append([cls, int(x0), int(y0), int(x0 + w0), int(y0 + h0)])
    return np.array(gt_labels, dtype=np.int32)


def label_to_color(label, color_maps, ignore_label=255):
    assert label.dtype == np.uint8 and len(label.shape) == 2

    height, width = label.shape
    image = np.ones((height, width, 3), dtype=np.uint8) * ignore_label
    cls_ids = np.setdiff1d(np.unique(label), np.array([ignore_label]))

    for cls in cls_ids:
        try:
            image[label == cls, :] = color_maps[cls]
            # image[label == cls] = color_maps[cls] # 等价写法
        except Exception as err:
            print("err:", err)
    return image


def label_to_segment(gt_label, n_cls, ignore_label=255):
    # print("label_to_segment:", np.min(gt_label), np.max(gt_label), ignore_label)
    cls_ids = np.setdiff1d(np.unique(gt_label), np.array([ignore_label]))
    # print("cls_ids:", cls_ids)
    height, width = gt_label.shape
    gt_masks = np.zeros((n_cls, height, width), dtype=np.uint8)
    for idx in cls_ids:
        temp = copy.deepcopy(gt_label)
        gt_masks[idx][temp == idx] = 1
    return gt_masks.astype(np.float32)


def color_to_label(color_img, cmap, ignore_label=255):
    height, width, _ = color_img.shape
    label = np.ones((height, width), dtype=np.uint8) * ignore_label
    for idx, color in enumerate(cmap):
        label[np.all(color_img == list(color), axis=-1)] = idx
    return label


def color_to_segment(color_img, cmap):
    n_cls = len(cmap)
    height, width, _ = color_img.shape
    masks = np.zeros((n_cls, height, width), dtype=np.uint8)
    for idx, color in enumerate(cmap):
        temp = np.zeros((height, width), dtype=np.uint8)
        temp[np.all(color_img == list(color), axis=-1)] = 1
        masks[idx] = temp
    return masks.astype(np.float32)


def segment_to_label(segs, ignore_label=255):
    n_cls, height, width = segs.shape
    label = np.ones((height, width), np.uint8) * ignore_label
    for cls in range(n_cls):
        label[segs[cls] > 1e-4] = cls
    return label


def segment_to_color(masks, cmap, ignore_label=255):
    n_cls, height, width = masks.shape
    color_img = np.ones((height, width, 3), dtype=np.uint8) * ignore_label
    for idx, mask in enumerate(masks):
        color_img[mask > 1e-4] = cmap[idx]
    return color_img


def label_write(file_path, mask):
    assert suffix(file_path) == ".png", "label should .png"
    cv2.imwrite(file_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def label_read(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def unique_color(color_img):
    """ 获取三通道彩色图颜色值。
    """
    return np.unique(color_img.reshape(-1, 3), axis=0)


# img[np.all(img != [0, 0, 255], axis=-1)] = [255, 255, 255]
# img[np.all(img == [0, 0, 255], axis=-1)] = [0, 0, 0]
# cv2.imwrite(path.join(PRO_DIR, "data/dog1.bmp"), img)

# gt_label = np.ones(img.shape[:2], dtype=np.uint8) * 255
# gt_label[np.all(img == [0, 0, 255], axis=-1)] = 0
# gt_label[np.all(img == [0, 255, 0], axis=-1)] = 1


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
