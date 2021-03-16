#!/usr/bin/env python3
# coding: utf-8

"""
@File      : quality_cv.py
@Author    : alex
@Date      : 2021/2/6
@Desc      :
"""

import cv2
import numpy as np

__all__ = ["motion_blur", "gauss_blur", "gauss_noise", "adjust_compress"]


def motion_blur(img, degree=10, angle=20):
    """ 图像运动模糊，degree越大，模糊程度越高
    :param img: Numpy Image
    :param degree: int, >1, 600*800，最多degree=10
    :param angle: 运动角度，可随机设置，[0,360]
    url: https://www.cnblogs.com/arkenstone/p/8480759.html
    """

    if degree <= 1:
        raise ValueError("degree should > 1.")

    img = np.array(img)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def gauss_blur(img, degree=7, sigmaX=0, sigmaY=0):
    """ 对焦模糊，degree越大，模糊程度越高
    :param degree: 大于1的奇数
    url: https://www.cnblogs.com/arkenstone/p/8480759.html
    """

    if degree <= 1:
        raise ValueError("degree should > 1.")
    degree = degree // 2 * 2 + 1
    img = cv2.GaussianBlur(img, ksize=(degree, degree), sigmaX=sigmaX, sigmaY=sigmaY)
    return img


def gauss_noise(img, degree=None):
    """ 在每个像素点添加随机扰动
    :param degree:  > 0
    url: https://www.cnblogs.com/arkenstone/p/8480759.html
    """

    row, col, ch = img.shape
    mean = 0
    if not degree:
        var = np.random.uniform(0.004, 0.01)
    else:
        var = degree
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
    noisy = np.array(noisy, dtype=np.uint8)
    return noisy


def adjust_compress(img, ratio):
    """ jpeg图像质量压缩
    :param ratio: [0~100], 数值越小，压缩比越高，图片质量损失越严重
    """

    assert len(img.shape) == 3 and ratio <= 100 and ratio >= 0

    params = [cv2.IMWRITE_JPEG_QUALITY, ratio]  # ratio:0~100
    msg = cv2.imencode(".jpg", img, params)[1]
    msg = (np.array(msg)).tostring()
    img = cv2.imdecode(np.fromstring(msg, np.uint8), cv2.IMREAD_COLOR)
    return img


if __name__ == "__main__":
    pass
