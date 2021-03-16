#!/usr/bin/env python3
# coding: utf-8

"""
@File      : transform_test.py
@Author    : alex
@Date      : 2021/2/14
@Desc      :
"""

import cv2
import numpy as np
from idata import PRO_DIR
from idata.fileio import *
from idata.datasets.transforms import *


def dict_test():
    T = Compose([
        ToPILImage(),
        ToNumpy(),
        ToTensor(),
        Normalize(mean, std),
        UnNormalize(mean, std),
        ToNumpy(),
        # ToPILImage(),
        Resize((600, 800)),
        # Pad((50, 100, 150, 200), fill=200),  # pad_left, pad_top, pad_right, pad_bottom
        # ToNumpy(),
        # CenterCrop((800, 700), fill=200),
        # RandomHorizontalFlip(1),
        # RandomVerticalFlip(1),
        # RandomRotation(30, fill=(200, 0, 255)),
        # RandomPerspective(fill=(255, 0, 0)),
        # RandomAffine(fill=(255, 0, 0)),
        # RandomCrop(700, fill=(255, 0, 0)),  #
        # RandomResizedCrop((300, 400)),
    ], ignore_label=80)

    result = dict(
        img=img,
        gt_seg=gt_label
    )
    print("dict_test result:")
    print("img-shape:", img.shape)
    y = T(result)
    print("y-shape:", y["img"].shape, y["gt_seg"].shape, np.unique(y["gt_seg"]))
    img_write(path.join(PRO_DIR, "data/dog_ret2.jpg"), y["img"])
    img_write(path.join(PRO_DIR, "data/dog_seg.jpg"), y["gt_seg"])
    pass


if __name__ == "__main__":
    # test()
    # dict_test()

    img = cv2.imread(path.join(PRO_DIR, "data/dog.jpg"))

    gt_label = np.ones((720, 720), dtype=np.uint8)
    gt_label[200:500, 300:400] = 0
    gt_label = gt_label * 255
    print("xxx:", path.join(PRO_DIR, "data/label.png"))
    print("color:", gt_label.shape)
    img_write(path.join(PRO_DIR, "data/label.png"), gt_label.astype(np.uint8))
    cv2.imwrite(path.join(PRO_DIR, "data/label.jpg"), gt_label.astype(np.uint8))
    gt_label = img_read(path.join(PRO_DIR, "data/label.png"))

    mean = np.array([123.675, 116.28, 103.53]) / 255
    std = np.array([58.395, 57.12, 57.375]) / 255

    # test()
    # dict_test()
    pass
