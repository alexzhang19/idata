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
        # Gamma(alfa=2),
        # RandomGrayScale(p=1),
        # RandomColorScale(p=1),
        ColorJitter(p=1, brightness=None),
    ])

    print("dict_test result:")
    print("img-shape:", img.shape)
    val = 0
    for i in range(1):
        x = dict(img=img)
        y = T(x)
        img_write(path.join(PRO_DIR, "data/dog_ret2.jpg"), y["img"])
        print("aaaaaaaaaa:", np.sum(y["img"]) / float(np.sum(img)))
        print("metas:", y["metas"])
        val += np.sum(y["img"]) / float(np.sum(img))
    print(val / 100)


if __name__ == "__main__":
    PRO_DIR = "/home/temp/dog"
    img = cv2.imread(path.join(PRO_DIR, "data/dog.jpg"))

    # gamma = np.ones((h, w, 3), dtype=np.float32)
    # gamma = np.random.random((h // 50, w // 50, 3))
    # gamma = cv2.blur(img, (5, 5))
    # gamma = cv2.resize(gamma, (600, 800))
    # print(gamma.shape, np.min(gamma), np.max(gamma))
    # ret = (np.power(img.astype(np.float16) / float(np.max(img)), gamma) * 255).astype(np.uint8)
    # cv2.imwrite(path.join(PRO_DIR, "data/ret.jpg"), ret)
    dict_test()
    pass
