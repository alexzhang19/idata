#!/usr/bin/env python3
# coding: utf-8

"""
@File      : visual.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import os
import cv2
import copy
import numpy as np
from idata.visual.color import random_color

__all__ = [
    "random_color",
    "find_counters", "draw_counters",
    "bounding_rect", "get_masks_rgns", "draw_rectangle", "put_text", "put_text_ex",
    "draw_mask", "merge_ret", "draw_detect_boxes",
]


############################
#       找轮廓、画轮廓     #
############################
def find_counters(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """
    查找mask轮廓点
    mode=RETR_TREE，返回所有轮廓；
    mode=cv2.RETR_EXTERNAL，只返回外轮廓。
    """

    is_v2 = cv2.__version__.startswith("2.")
    if is_v2:
        _, contours, _ = cv2.findContours(mask, mode, method)
    else:
        contours, _ = cv2.findContours(mask, mode, method)
    return contours


def draw_counters(image, contours, contourIdx=-1, color=(0, 0, 255), thickness=2):
    """
    图像上画轮廓
    contours: [cnt1, cnt2, ...], cnt1.shape = (N, 1, 2), cnt1.dtype= int32
    """

    if 2 == len(image.shape):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = copy.deepcopy(image)
    cv2.drawContours(image, contours, contourIdx, color, thickness)
    return image


##############################
#   找外接矩形、画外接矩形   #
##############################
def bounding_rect(mask):
    """
    获取mask上标记外接矩形框
    """

    mask[mask >= 128] = 255
    mask[mask < 128] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if np.all(0 == mask):
        return []
    rets = []
    counters = find_counters(mask)
    for cnt in counters:
        (x0, y0, w0, h0) = cv2.boundingRect(cnt)
        rets.append([int(x0), int(y0), int(x0 + w0), int(y0 + h0)])
    return rets


def get_masks_rgns(masks):
    '''
    获取masks中RGNS
    '''

    rets = []
    for mask in masks:
        rgns = bounding_rect(mask)
        for rgn in rgns:
            rets.append(rgn)
    return rets


def _rectangle_dot(image, pt1, pt2, color, thickness):
    (x1, y1) = pt1
    (x2, y2) = pt2
    xrange = (list(range(x1, x2, thickness * 3)))
    xrange.append(x2)
    yrange = (list(range(y1, y2, thickness * 3)))
    yrange.append(y2)
    for i in range(0, len(xrange), 2):
        try:
            ix1 = xrange[i]
            ix2 = xrange[i + 1]
            cv2.line(image, (ix1, y1), (ix2, y1), color, thickness=thickness)
            cv2.line(image, (ix1, y2), (ix2, y2), color, thickness=thickness)
        except:
            pass
    for i in range(0, len(yrange), 2):
        try:
            iy1 = yrange[i]
            iy2 = yrange[i + 1]
            cv2.line(image, (x2, iy1), (x2, iy2), color, thickness=thickness)
            cv2.line(image, (x1, iy1), (x1, iy2), color, thickness=thickness)
        except:
            pass
    return image


def draw_rectangle(image, boxes, colors=(0, 0, 255), thickness=2, dot=False):
    """
    画外接矩形
    boxes: [[x1, y1, x2, y2, cls-可选], ...]，若boxes定义类别
    colors: 若boxes无类别,则为单颜色；若boxes有类别，则可设置颜色数组实现多类别颜色显示
    """

    if 2 == len(image.shape):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = copy.deepcopy(image)

    boxes = np.array(boxes, dtype=np.int16)
    for box in boxes:
        color = colors
        [x1, y1, x2, y2] = box[:4]
        if len(box) == 5:
            color = colors[box[4]]
        if not dot:
            cv2.rectangle(image, (x1, y1), (x2, y2), color=tuple(color), thickness=thickness)
        else:
            _rectangle_dot(image, (x1, y1), (x2, y2), color=tuple(color), thickness=thickness)
    return image


def draw_detect_boxes(img, labels, cmaps=None, label_names=None,
                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale=1.2, thickness=2):
    """ yolo boxes
        img = cv2.imread("dog.jpg")
        ret = draw_detect_boxes(img, np.array([[0, 100, 100, 400, 400]]), [[0, 255, 255]], ["cat asda"])
        cv2.imwrite("dog111.jpg", ret)
    """

    image = copy.deepcopy(img)
    img_h, img_w = image.shape[:2]
    for idx, label in enumerate(labels):
        [cls, x1, y1, x2, y2] = label
        cls_color = cmaps[cls]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, thickness=thickness)
        try:
            obj_name = label_names[idx]
        except:
            obj_name = "%d" % cls
        # cv2.getTextSize(obj_name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2)
        (tw, th), _ = cv2.getTextSize(obj_name, fontFace, fontScale, thickness)
        is_cut = False
        while tw > x2 - x1:
            obj_name = obj_name[:int(len(obj_name) * 0.9)]
            (tw, th), _ = cv2.getTextSize(obj_name, fontFace, fontScale, thickness)
            is_cut = True
        if is_cut:
            obj_name += "..."
            (tw, th), _ = cv2.getTextSize(obj_name, fontFace, fontScale, thickness)

        th += 4
        image = cv2.rectangle(image, (max(x1 - thickness // 2, 0), max(y1 - th, 0)),
                              (min(img_w, x2 + thickness // 2), y1),
                              cls_color, thickness=cv2.FILLED)

        image = cv2.putText(image, obj_name, (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            color=(0, 0, 0), fontScale=fontScale * 1.0, thickness=thickness)
    return image


def put_text(image, locs, texts, color=(255, 0, 0)):
    """
    图片上写文字，支持中文.
    参考：https://baijiahao.baidu.com/s?id=1588536890529844280&wfr=spider&for=pc
    """

    if 2 == len(image.shape):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = copy.deepcopy(image)
    assert len(texts) == len(locs)
    for i, text in enumerate(texts):
        [x1, y1] = locs[i][:2]
        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, color, 0)  # 原生cv2.putText不支持中文.
    return image


def put_text_ex(img, txt, loc, fount=32, color=(255, 0, 0)):
    """
    写中文
    NotoSansCJK.ttc下载地址：
    链接：https://pan.baidu.com/s/1cDM45oycD_ciCCd23HAqBA
    提取码：xudy
    """

    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", "simhei.ttf"), fount)
    if (isinstance(txt, str)):
        txt.encode('gb2312')
    draw = ImageDraw.Draw(pil)
    draw.text(loc, txt, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_mask(img, mask, cmap=None, n_cls=6, alfa=0.5, ignore_label=255):
    """
    将mask按颜色显示在image上
    :param img: 原图
    :param mask: 按类别1、2、3、4...编码
    :param cmap: 颜色数组，默认由colormap()产生
    :param alfa: 不透明度
    :return: 显示的label mask
    """

    assert mask.dtype == np.uint8
    assert len(mask.shape) == 2

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    label_img = copy.deepcopy(img)
    cls_ids = np.setdiff1d(np.unique(mask), np.array([ignore_label]))

    if cmap is None:
        cmap = random_color(n_cls)

    for cls_id in cls_ids:
        try:
            label_img[mask == cls_id] = cmap[cls_id]
        except Exception as err:
            print("err:", err)

    label_img = (label_img * alfa + (1 - alfa) * img).astype(np.uint8)
    return label_img


def merge_ret(*imgs):
    """
    但类别图像结果显示
    :param *imgs: 图像序列连接
    :return: 合并连接后图像
    """
    if len(imgs) == 0:
        return None
    elif len(imgs) == 1:
        return imgs[0]

    seg_line = np.ones((imgs[0].shape[0], 5, imgs[0].shape[2])) * 128

    ret_imgs = []
    for idx, img in enumerate(imgs):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if idx != 0:
            ret_imgs.append(seg_line)
        ret_imgs.append(img)
    return np.hstack(tuple(ret_imgs))
