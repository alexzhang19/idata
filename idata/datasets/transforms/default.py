#!/usr/bin/env python3
# coding: utf-8

import torch as t
import numpy as np
import idata.datasets.transforms as T
from idata.datasets import ShapeData, ShapeType

__all__ = ["img_transform", "TClsTarget"]


def classify_target_transform(target):
    target = np.array(target)
    return t.from_numpy(target).type(t.LongTensor)


TClsTarget = T.Compose([T.Lambda(classify_target_transform)])


def img_transform(img_w: int, img_h: int):
    return T.Compose([
        T.Resize((img_w, img_h)),
        T.ToTensor()
    ])


if __name__ == "__main__":
    img, gt_label = \
        ShapeData(10, type=ShapeType.classify, transform=img_transform(112, 112), target_transform=TClsTarget)[0]
    print(img.shape, gt_label)
    pass
