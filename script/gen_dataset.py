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
from idata.datasets import ShapeData, ShapeType
from idata.datasets.segment import *


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


if __name__ == "__main__":
    print("keys:", DCFG.keys())
    dataset = GenData(100)
    # datasets.classify("classify")
    # datasets.classify_vis("classify_vis")
    dataset.segment("segment")
    pass
