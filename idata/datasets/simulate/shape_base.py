#!/usr/bin/env python3
# coding: utf-8

"""
@File      : simulate.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 用于生成分类、分割、检测、实例分割、多属性数据集
url: https://github.com/JenifferWuUCLA/pulmonary-nodules-MaskRCNN
"""

import cv2
import math
import random
import numpy as np

__all__ = ["ShapeBase"]


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_miou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def compute_miou(box, boxes, box_area, boxes_area):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    # union = box_area + boxes_area[:] - intersection[:]
    compute_miou = intersection / np.minimum(box_area, boxes_area[:]) + 1e-4
    return compute_miou


class ShapeBase(object):
    NAME = "ShapeBase"
    MAX_NUM = 4
    MASK_VALUE = 1

    BG_COLOR_RANGE = [50, 200]  # 背景图像RGB颜色范围

    CLASSES = ["triangle", "circle", "square", "plus", "annulus"]
    PALETTE = [[128, 0, 0], [0, 128, 0], [0, 0, 128],
               [0, 128, 128], [128, 0, 128]]

    def __init__(self, total_cnt: int = 3000, shape=(128, 128)):
        """
        :param shape: (h, w)
        """

        assert total_cnt > 0, "datasets image count mast > 0"
        assert len(self.CLASSES) == len(self.PALETTE), "class num should match pipette length."

        self.shape = shape
        self.total_cnt = total_cnt

        # mesh_square, mesh_circle
        self._data_dicts = [tuple(self.random_image()) for _ in range(total_cnt)]

    @property
    def classes(self):
        return self.CLASSES

    @property
    def class_idx(self):
        """ 属性值，类别source_map索引
        """

        if self.classes is None:
            return
        return {i: v for i, v in enumerate(self.classes)}

    def __len__(self):
        return len(self._data_dicts)

    def __getitem__(self, index):
        """ 实例分割结果输出， masks: (cnt, height, width), masks值为0,1， uint8
        """

        bg_color, shapes = self._data_dicts[index]
        img, masks, class_ids = self.load_image(bg_color, shapes)
        return img, masks, class_ids

    def _random_shape(self, height, width):
        shape = random.choice(list(self.classes))
        color = tuple([random.randint(0, 255) for _ in range(3)])

        # Center x, y, r为半径
        b_size = int(min(self.shape[0], self.shape[1]) / 2)
        buffer = int(b_size * 0.15)
        x = random.randint(buffer, width - buffer - 1)
        y = random.randint(buffer, height - buffer - 1)
        r = random.randint(buffer, int(b_size * 0.4))
        return shape, color, (x, y, r)

    def random_image(self):
        bg_color = np.array([random.randint(self.BG_COLOR_RANGE[0], self.BG_COLOR_RANGE[1]) for _ in range(3)])

        shapes, boxes = [], []
        height, width = self.shape
        N = random.randint(1, self.MAX_NUM)
        for _ in range(N):
            shape, color, dims = self._random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, r = dims
            boxes.append([max(x - r, 0), max(y - r, 0),
                          min(x + r, width), min(y + r, height)])  # [x1, y1, x2, y2]

        # Apply non-max suppression wit 0.3 threshold to avoid shapes covering each other
        keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.5)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def load_image(self, bg_color, shapes):
        """ 加载图片
            image: (height, width, 3)
            masks: (cnt, height, width), cnt为图像上目标个数
            class_ids: [类别]
        """

        height, width = self.shape
        bg_color = np.array(bg_color).reshape([1, 1, 3])
        image = np.ones([height, width, 3], dtype=np.uint8) * bg_color.astype(np.uint8)
        masks = np.zeros([len(shapes), height, width], dtype=np.uint8)
        class_ids = np.array([self.classes.index(s[0]) for s in shapes])

        for idx, (shape, color, dims) in enumerate(shapes):
            if shape == "triangle":
                image, masks[idx] = self.add_triangle(image, masks[idx], dims, color)
            elif shape == "circle":
                image, masks[idx] = self.add_circle(image, masks[idx], dims, color)
            elif shape == "square":
                image, masks[idx] = self.add_square(image, masks[idx], dims, color)
            elif shape == "plus":
                image, masks[idx] = self.add_plus(image, masks[idx], dims, color)
            elif shape == "annulus":
                image, masks[idx] = self.add_annulus(image, masks[idx], dims, color, bg_color)
            else:
                raise ValueError(f"Unsupported Shape Type: {shape}")
        return image, masks, class_ids

    def add_triangle(self, image, mask, dims, color):
        x, y, r = dims
        points = np.array([[(x, y - r),
                            (x - r / math.sin(math.radians(60)), y + r),
                            (x + r / math.sin(math.radians(60)), y + r),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
        mask = cv2.fillPoly(mask, points, self.MASK_VALUE)
        return image, mask

    def add_circle(self, image, mask, dims, color):
        x, y, r = dims
        image = cv2.circle(image, (x, y), r, color, -1)
        mask = cv2.circle(mask, (x, y), r, self.MASK_VALUE, -1)
        return image, mask

    def add_square(self, image, mask, dims, color):
        x, y, r = dims
        image = cv2.rectangle(image, (x - r, y - r), (x + r, y + r), color, -1)
        mask = cv2.rectangle(mask, (x - r, y - r), (x + r, y + r), self.MASK_VALUE, -1)
        return image, mask

    def add_plus(self, image, mask, dims, color):
        x, y, r = dims
        height, width = self.shape
        ax1, ax2, ay1, ay2 = max(x - 2, 0), min(x + 2, height), max(y - r, 0), min(y + r, width)
        bx1, bx2, by1, by2 = max(x - r, 0), min(x + r, height), max(y - 2, 0), min(y + 2, width)
        image[ax1:ax2, ay1:ay2, :] = color
        image[bx1:bx2, by1:by2, :] = color
        mask[ax1:ax2, ay1:ay2] = self.MASK_VALUE
        mask[bx1:bx2, by1:by2] = self.MASK_VALUE
        return image, mask

    def add_annulus(self, image, mask, dims, color, bg_color):
        x, y, r = dims
        image = cv2.circle(image, (x, y), r, color, -1)
        mask = cv2.circle(mask, (x, y), r, self.MASK_VALUE, -1)
        image = cv2.circle(image, (x, y), int(r * 0.85), bg_color.reshape(-1).tolist(), -1)
        mask = cv2.circle(mask, (x, y), int(r * 0.85), 0, -1)
        return image, mask


if __name__ == "__main__":
    import cv2
    import numpy as np

    img, masks, class_ids = ShapeBase()[0]
    print(img.shape, masks.shape, np.min(masks), np.max(masks), class_ids)
