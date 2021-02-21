#!/usr/bin/env python3
# coding: utf-8

"""
@File      : segment.py
@Author    : alex
@Date      : 2021/2/6
@Desc      : 
"""

import cv2
import numpy as np
from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["SegData"]

"""
    分类数据集格式1：
    datasets:
        TRAIN_FILE(train.txt):  图片文件绝对路径列表 # 可省
        VALID_FILE(valid.txt): 图片文件绝对路径列表 # 可省
        NAME_FILE(names.txt)：由用户标注时准备,名字列表 # 可省，正确使用前必须生成
"""


@DATASETS.register_module(force=True)
class SegData(BaseDataset):
    NAME = "SegData"
    TRAIN_FILE = "train.txt"  # 生成训练文件列表
    VALID_FILE = "valid.txt"  # 生成测试文件列表
    NAME_FILE = "names.txt"  # 生成的名字列表文件名
    ANNO_DIR = "labels"

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None, need_path=False):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(SegData, self).__init__()

    def _data_prepare(self):
        """
        输出格式： [{img_path: xxx, label_path: xx}, {}, ...]
        """
        data_set = self.VALID_FILE if self.test_mode else self.TRAIN_FILE
        img_paths = CText(path.join(self.data_dir, data_set)).read_lines(is_split=False)
        # print(img_paths, len(img_paths))

        dicts = []
        for idx, img_path in enumerate(img_paths):
            label_path = path.join(self.data_dir, self.ANNO_DIR, key_name(img_path) + ".png")
            if not (path.exists(img_path) and path.exists(label_path)):
                continue

            r = Dict({
                "img_path": img_path,
                "label_path": label_path
            })
            dicts.append(r)
        return dicts

    def _load_meta(self):
        name_file = path.join(self.data_dir, self.NAME_FILE)
        if not path.exists(name_file):
            return None

        names = parse_txt_file(name_file)
        # print("names:", names)
        return names

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = self.img_read(item.img_path)
        target = self.img_read(item.label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # numpy, int, str
        if self.need_path:
            return img, target, item.img_path
        else:
            return img, target

    def vis(self, cnt=100):
        from idata.visual import draw_mask

        vis_dir = path.join(self.data_dir + "_vis")
        mkdir(vis_dir)

        cnt = len(self) if cnt is None else min(cnt, len(self))
        for idx in range(cnt):
            item = self._data_dicts[idx]
            img = self.img_read(item.img_path)
            blank = np.ones(img.shape, dtype=np.uint8) * 255
            target = self.img_read(item.label_path)
            show_img = draw_mask(blank, target, alfa=1)
            cv2.imwrite(path.join(vis_dir, "%06d.jpg" % idx), show_img)


if __name__ == '__main__':
    from idata.datasets import DCFG

    data_dir = path.join(DCFG.root_dir, DCFG.segment)
    print("data_dir:", data_dir)
    dataset = SegData(data_dir, test_mode=True, need_path=True)
    print("total cnt: ", len(dataset))
    dataset.vis(100)
    pass
