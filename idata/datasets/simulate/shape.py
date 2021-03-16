#!/usr/bin/env python3
# coding: utf-8

import copy
import math
import random
import numpy as np
from enum import Enum, unique
from torch.utils.data import Dataset
from idata.datasets.simulate import *
from idata.datasets.simulate.shape_base import ShapeBase

__all__ = ["ShapeData", "ShapeType"]


@unique
class ShapeType(Enum):
    normal = 0  # 原始标签
    classify = 1  # 分类
    segment = 2  # 分割
    detect = 3  # 检测
    instance = 4  # 实例分割
    attribute = 5  # 多属性


class ShapeData(Dataset, ShapeBase):
    NAME = "ShapeData"
    IGNORE_LABEL = 255

    def __init__(self, total_cnt: int = 30000, shape=(128, 128), type: ShapeType = ShapeType.normal,
                 transform=None, target_transform=None):
        assert total_cnt > 0, "datasets image count mast > 0"

        if type == ShapeType.classify:
            self.MAX_NUM = 1

        super(ShapeData, self).__init__(total_cnt=total_cnt, shape=shape)

        self.type = type
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """ masks: (cnt, height, width), masks值为0,1， uint8
        """

        img, masks, class_ids = ShapeBase.__getitem__(self, index)
        target = self.data_trans(masks, class_ids)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def data_trans(self, masks, class_ids):
        # masks => dim, height, width, R=[0, 1]
        dim, height, width = masks.shape
        # print("class_ids:", class_ids)

        if self.type == ShapeType.normal:
            return masks, class_ids
        elif self.type == ShapeType.classify:
            assert len(class_ids) == 1, "classify datasets shoule only have one category."
            return int(class_ids[0])
        elif self.type == ShapeType.segment:
            return order_to_label(masks, class_ids, self.IGNORE_LABEL)
        elif self.type == ShapeType.detect:
            return order_to_detect(masks, class_ids, mini_pixel=10)
        else:
            raise ValueError(f"Unsupported Type: {self.type}")

    def color_mask(self, gt_mask):
        assert gt_mask.shape == self.shape, "gt_mask shape error."
        return label_to_color(gt_mask, color_maps=self.PALETTE, ignore_label=self.IGNORE_LABEL)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from idata.fileio import *
    from idata.visual.visual import *

    # 1.type=ShapeType.normal
    # img, (masks, class_ids) = ShapeData(type=ShapeType.normal)[0]
    # print(img.shape, masks.shape, np.min(masks), np.max(masks), class_ids)

    # 2.type=ShapeType.classify
    # img, gt_label = ShapeData(type=ShapeType.classify)[0]
    # print(img.shape, gt_label)

    # 3.type=ShapeType.classify
    # shape = ShapeData(total_cnt=10, type=ShapeType.segment)
    # img, gt_label = shape[0]
    # print(img.shape, gt_label.shape, np.unique(gt_label))
    # color_img = label_to_color(gt_label, shape.PALETTE)
    # print("color_img:", color_img.shape, unique_color(color_img))

    # 4.type=ShapeType.detect
    shape = ShapeData(total_cnt=10, type=ShapeType.detect)
    img, gt_labels = shape[0]
    print(img.shape, gt_labels)
    ret = draw_rectangle(img, gt_labels[:, 1:])
    cv2.imwrite(path.join("/home/temp/aaa.jpg"), ret)
