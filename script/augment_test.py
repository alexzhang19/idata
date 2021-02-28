#!/usr/bin/env python3
# coding: utf-8

"""
@File      : augment_test.py
@Author    : alex
@Date      : 2021/2/10
@Desc      :
"""

import cv2
import numpy as np
from idata import PRO_DIR
from PIL import Image
from idata.fileio import path
from idata.augment.affine import F_cv, F_pil
from idata.augment.affine import *
from idata.augment.utils import *


def test_img():
    print("test_img:")
    img = cv2.imread(img_path)
    print(img.shape)
    ret = F_cv.rotate(img, 30, expand=True)
    # angle, center=None, scale=1.0, border_value=0, interpolation=cv2.INTER_LINEAR, auto_bound=False
    print(ret.shape, path.join(PRO_DIR, "datasets/ret.jpg"))
    cv2.imwrite(path.join(PRO_DIR, "datasets/ret.jpg"), ret)
    pass


def test_pil_img():
    """
     constant: 添加有颜色的常数值边界,还需要下一个参数(value),由fill提供.
            reflect: 边界元素的镜像.cv2.BORDER_DEFAULT与之类似.
            edge: 重复最后一个元素.
            symmetric: pads with reflection of image
    """
    print("test_pil_img:")
    pil_img = Image.open(img_path)
    print(pil_img.size)
    ret = F_pil.pad(pil_img, (50, 100), fill=(0, 255, 0), padding_mode="wrap")
    print(ret.size, path.join(PRO_DIR, "datasets/ret_pil.jpg"))
    ret.save(path.join(PRO_DIR, "datasets/ret_pil.jpg"))
    pass


def test():
    print("test_img:")
    pil_img = Image.open(img_path)
    img = cv2.imread(img_path)
    print("org-shape:", img.shape)
    ret1 = rotate(img, 30, interpolation="nearest", expand=True)
    print("ret-shape:", ret1.shape)
    cv2.imwrite(path.join(PRO_DIR, "datasets/test_ret.jpg"), ret1)
    ret2 = rotate(pil_img, 30, interpolation="nearest", expand=True)
    ret2.save(path.join(PRO_DIR, "datasets/test_ret_pil.jpg"))
    pass


if __name__ == "__main__":
    img_path = path.join(PRO_DIR, "datasets/dog.jpg")

    test()
    pass
