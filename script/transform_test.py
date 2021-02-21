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
from PIL import Image


def test():
    # T = Compose([
    #     ToPILImage(),
    #     ToNumpy(),
    #     ToTensor(),
    #     Normalize(mean, std),
    #     UnNormalize(mean, std),
    #     ToNumpy(),
    #     Resize(200),
    # ])
    # y = T(img)
    #
    # print("test result:")
    # print("img:", img.shape)
    # print("y:", y.shape, np.min(y), np.max(y))
    # img_write(r"E:\common\idata\data\dog_ret1.jpg", y)
    pass


def dict_test():
    T = Compose([
        ToPILImage(),
        ToNumpy(),
        ToTensor(),
        Normalize(mean, std),
        UnNormalize(mean, std),
        ToNumpy(),
        # ToPILImage(),
        # Resize(200),
        # Pad(100),
        # ToNumpy(),
        CenterCrop(800),
        # RandomHorizontalFlip(1),
        # RandomVerticalFlip(1),
        RandomRotation(30)
    ])

    result = dict(
        img=img,
        gt_seg=gt_label,
        metas=dict(
            ori_shape=img.shape,
            img_shape=img.shape,
            file_path=r"E:\common\idata\data\dog.jpg",
        ),
    )
    print("dict_test result:")
    print("img-shape:", img.shape)
    y = T(result)
    print("y-shape:", y["img"].shape, y["gt_seg"].shape, np.unique(y["gt_seg"]))
    img_write(r"E:\common\idata\data\dog_ret2.jpg", y["img"])
    img_write(r"E:\common\idata\data\dog_label2.jpg", y["gt_seg"])
    pass


if __name__ == "__main__":
    # test()
    # dict_test()

    img = cv2.imread(path.join(PRO_DIR, "data/dog.jpg"))
    # img[np.all(img != [0, 0, 255], axis=-1)] = [255, 255, 255]
    # img[np.all(img == [0, 0, 255], axis=-1)] = [0, 0, 0]
    # cv2.imwrite(path.join(PRO_DIR, "data/dog1.bmp"), img)

    # gt_label = np.ones(img.shape[:2], dtype=np.uint8) * 255
    # gt_label[np.all(img == [0, 0, 255], axis=-1)] = 0
    # gt_label[np.all(img == [0, 255, 0], axis=-1)] = 1
    # img_write(path.join(PRO_DIR, "data/label.png"), gt_label)
    # img_write(path.join(PRO_DIR, "data/label.jpg"), gt_label)
    gt_label = img_read(path.join(PRO_DIR, "data/label.png"))
    # img2 = img_read(path.join(PRO_DIR, "data/label.jpg"))
    # print(np.unique(img1), np.unique(img2))
    # print("gt_label:", np.unique(gt_label))

    mean = np.array([123.675, 116.28, 103.53]) / 255
    std = np.array([58.395, 57.12, 57.375]) / 255

    # test()
    dict_test()
    pass
