#!/usr/bin/env python3
# coding: utf-8

"""
@File      : gen_dataset.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 数据集数据生成
"""

import cv2
from idata.fileio import *
from idata.datasets import DCFG
from idata.datasets.simulate import *
from idata.visual import *


class GenData(object):
    def __init__(self, total_cnt, shape=(256, 256), train_rate=0.9):
        self.total_cnt = total_cnt
        self.shape = shape
        self.train_rate = train_rate
        self.root_dir = DCFG.root_dir

    def classify(self, uname):
        assert uname in ["classify"], f"{uname} not support."

        dataset = ShapeData(self.total_cnt, self.shape, type=ShapeType.classify)
        names = dataset.classes

        data_dir = path.join(self.root_dir, uname)
        mkdir(data_dir)
        print("data_dir:", data_dir)

        train_file = CText(path.join(data_dir, "train.txt"), is_clear=True)
        valid_file = CText(path.join(data_dir, "valid.txt"), is_clear=True)
        for idx, data in enumerate(dataset):
            img, target = data
            item_dir = path.join(data_dir, names[target])
            mkdir(item_dir)
            img_path = path.join(item_dir, "shape_%06d.jpg" % (idx + 1))
            cv2.imwrite(img_path, img)

            if idx < int(self.total_cnt * self.train_rate):
                train_file.append(img_path + "\n")
            else:
                valid_file.append(img_path + "\n")
        self.set_name(path.join(data_dir, "names.txt"), names)

    def classify_vis(self, uname):
        assert uname in ["classify_vis"], f"{uname} not support."

        dataset = ShapeData(self.total_cnt, self.shape, type=ShapeType.classify)
        names = dataset.classes
        data_dir = path.join(self.root_dir, uname)
        mkdir(data_dir)

        for idx, data in enumerate(dataset):
            img, target = data
            if idx < int(self.total_cnt * self.train_rate):
                data_set = "train"
            else:
                data_set = "valid"

            item_dir = path.join(data_dir, data_set, names[target])
            mkdir(item_dir)
            img_path = path.join(item_dir, "shape_%06d.jpg" % (idx + 1))
            cv2.imwrite(img_path, img)

        self.set_name(path.join(data_dir, "names.txt"), names)

    def segment(self, uname):
        assert uname in ["segment"], f"{uname} not support."

        dataset = ShapeData(self.total_cnt, self.shape, type=ShapeType.segment)
        names = dataset.classes
        data_dir = path.join(self.root_dir, uname)

        img_dir = path.join(data_dir, "imgs")
        label_dir = path.join(data_dir, "labels")
        vis_dir = path.join(data_dir, "vis")
        mkdir(img_dir)
        mkdir(label_dir)
        mkdir(vis_dir)

        train_file = CText(path.join(data_dir, "train.txt"), is_clear=True)
        valid_file = CText(path.join(data_dir, "valid.txt"), is_clear=True)
        for idx, data in enumerate(dataset):
            img, target = data
            img_path = path.join(img_dir, "shape_%06d.jpg" % (idx + 1))
            cv2.imwrite(img_path, img)
            label_write(path.join(label_dir, "shape_%06d.png" % (idx + 1)), target)
            label_write(path.join(vis_dir, "shape_%06d.png" % (idx + 1)), label_to_color(target, dataset.PALETTE))

            if idx < int(self.total_cnt * self.train_rate):
                train_file.append(img_path + "\n")
            else:
                valid_file.append(img_path + "\n")

        self.set_name(path.join(data_dir, "names.txt"), names)

    @staticmethod
    def set_name(name_file, names):
        cfile = CText(name_file, is_clear=True)
        for name in names:
            cfile.append(name + "\n")

    def yolo(self):
        dataset = ShapeData(self.total_cnt, self.shape, type=ShapeType.detect)
        names = dataset.classes
        data_dir = path.join(self.root_dir, "yolo")

        img_dir = path.join(data_dir, "imgs")
        vis_dir = path.join(data_dir, "vis")
        mkdir(img_dir)
        mkdir(vis_dir)

        train_file = CText(path.join(data_dir, "train.txt"), is_clear=True)
        valid_file = CText(path.join(data_dir, "valid.txt"), is_clear=True)
        for idx, data in enumerate(dataset):
            img, target = data
            img_path = path.join(img_dir, "shape_%06d.jpg" % (idx + 1))
            cv2.imwrite(img_path, img)
            cfile = CText(img_path[:-4] + ".txt", is_clear=True)
            img_h, img_w = img.shape[:2]
            for gt_label in target:
                [cls, x1, y1, x2, y2] = gt_label
                cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
                bw, bh = (x2 - x1) / img_w - 3e-5, (y2 - y1) / img_h - 3e-5
                cfile.append("%d %.6f %.6f %.6f %.6f\n" % (cls, cx, cy, bw, bh))

            label_names = [dataset.classes[v[0]] for v in target]
            vis_img = draw_detect_boxes(img, target, dataset.PALETTE, label_names, fontScale=1.0)
            cv2.imwrite(path.join(vis_dir, path.basename(img_path)), vis_img)

            if idx < int(self.total_cnt * self.train_rate):
                train_file.append(img_path + "\n")
            else:
                valid_file.append(img_path + "\n")

        self.set_name(path.join(data_dir, "names.txt"), names)


if __name__ == "__main__":
    print("keys:", DCFG.keys())
    dataset = GenData(3000, shape=(512, 512))
    # dataset.classify("classify")
    # dataset.classify_vis("classify_vis")
    # dataset.segment("segment")
    dataset.yolo()
    pass
