#!/usr/bin/env python3
# coding: utf-8

"""
@File      : augment_test.py
@Author    : alex
@Date      : 2021/3/16
@Desc      : 
"""

import cv2
import copy
import random
import numpy as np
from idata import PRO_DIR
from idata.fileio import *
from idata.datasets import DCFG
from idata.augment.enhance import get_photo_mask, adjust_gamma, grayscale, his_equal_color, \
    img_compress, adjust_brightness, adjust_saturation, adjust_contrast, adjust_sharpness, \
    adjust_hue
from idata.augment.quality import motion_blur, gauss_blur, gauss_noise


def enhance_test(img):
    ret_dir = path.join(ret_root_dir, "enhance")
    mkdir(ret_dir)
    pic = copy.deepcopy(img)

    # 1.adjust_gamma
    for i in range(11):
        gamma = i / 10 * (2 - 0.5) + 0.5
        ret = adjust_gamma(pic, gamma, photo_mask=get_photo_mask(0.5, 1, 3.0) * 0.7)
        cv2.imwrite(path.join(ret_dir, "gamma_%.2f.jpg" % gamma), ret)

    # 2.grayscale
    org_avg = int(np.mean(pic))
    expend = max(min(org_avg, 20), min(20, 255 - org_avg))
    for avg in range(max(org_avg - expend, 0), min(org_avg + expend, 255), 10):
        ret = grayscale(pic, org_avg)
        cv2.imwrite(path.join(ret_dir, "gray_%02d.jpg" % avg), ret)

    # 3.his_equal_color
    ret = his_equal_color(pic, 4, (8, 8))
    cv2.imwrite(path.join(ret_dir, "equal.jpg"), ret)

    # 4.img_compress
    for i in range(10, 60, 10):
        ret = img_compress(pic, suffix=".jpg", ratio=i)
        cv2.imwrite(path.join(ret_dir, "compress_%02d.jpg" % i), ret)

    # 5.brightness, saturation, contrast, sharpness
    for i in range(6):
        factor = i / 5 * (2 - 0.5) + 0.5
        ret1 = adjust_brightness(pic, factor)
        cv2.imwrite(path.join(ret_dir, "brightness_%.2f.jpg" % factor), ret1)

        ret2 = adjust_saturation(pic, factor)
        cv2.imwrite(path.join(ret_dir, "saturation_%.2f.jpg" % factor), ret2)

        ret3 = adjust_contrast(pic, factor)
        cv2.imwrite(path.join(ret_dir, "contrast_%.2f.jpg" % factor), ret3)

        ret4 = adjust_sharpness(pic, factor * 2)
        cv2.imwrite(path.join(ret_dir, "sharpness_%.2f.jpg" % (factor * 2)), ret4)

    # 6.adjust_hue
    for i in range(7):
        factor = i / 6 - 0.5
        ret = adjust_hue(pic, factor)
        cv2.imwrite(path.join(ret_dir, "hue_%.2f.jpg" % factor), ret)


def quality_test(img):
    ret_dir = path.join(ret_root_dir, "quality")
    mkdir(ret_dir)
    pic = copy.deepcopy(img)

    # 1.motion_blur
    for degree in range(2, 11, 2):
        ret = motion_blur(pic, degree, random.randint(0, 360))
        cv2.imwrite(path.join(ret_dir, "motion_%02d.jpg" % degree), ret)

    # 2.gauss_blur
    for degree in range(3, 10, 2):
        ret = gauss_blur(pic, degree, random.randint(0, 360), random.randint(0, 360))
        cv2.imwrite(path.join(ret_dir, "blur_%02d.jpg" % degree), ret)

    # 3.gauss_noise
    for degree in range(0, 100, 10):
        ret = gauss_noise(pic, degree)
        cv2.imwrite(path.join(ret_dir, "noise_%02d.jpg" % degree), ret)


def main():
    img = cv2.imread(path.join(PRO_DIR, "data/imgs/dog.jpg"))
    # print(img.shape)

    # enhance_test(img)

    quality_test(img)
    pass


if __name__ == "__main__":
    ret_root_dir = path.join(DCFG.root_dir, "augment_tests")
    mkdir(ret_root_dir)

    main()
    pass
