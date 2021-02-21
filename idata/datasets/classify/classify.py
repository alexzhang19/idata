#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import random
import numpy as np
from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["ClsData"]

"""
    分类数据集格式1：
    datasets:
        cls1_dir, cls2_dir, cls3_dir, cls4_dir, ...
        TRAIN_FILE(train.txt):  图片文件绝对路径列表 # 可省
        VALID_FILE(valid.txt): 图片文件绝对路径列表 # 可省
        NAME_FILE(names.txt)：由用户标注时准备,名字列表 # 可省，正确使用前必须生成
"""


@DATASETS.register_module(force=True)
class ClsData(BaseDataset):
    NAME = "ClsData"
    TRAIN_FILE = "train.txt"  # 生成训练文件列表
    VALID_FILE = "valid.txt"  # 生成测试文件列表
    NAME_FILE = "names.txt"  # 生成的名字列表文件名

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None, need_path=False):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(ClsData, self).__init__()

    def _data_prepare(self):
        if self.test_mode:
            return self._load_dicts(self.VALID_FILE)
        else:
            return self._load_dicts(self.TRAIN_FILE)

    def _load_meta(self):
        name_file = path.join(self.data_dir, self.NAME_FILE)
        if not path.exists(name_file):
            return None

        names = parse_txt_file(name_file)
        # print("names:", names)
        return names

    def set_names(self, names=None):
        if names is None:
            items = [v for v in ls(self.data_dir, real_path=True) if path.isdir(v)]
            names = [path.basename(v) for v in items if v not in ["train", "valid"]]

        self.classes = names
        print("class name is: ", self.classes)
        cfile = CText(path.join(self.data_dir, self.NAME_FILE), is_clear=True)
        for name in self.classes:
            cfile.append(name + "\n")

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = self.img_read(item.img_path)
        target = item.gt_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # numpy, int, str
        if self.need_path:
            return img, target, item.img_path
        else:
            return img, target

    def _load_dicts(self, data_set: str):
        """
        输出格式： [{img_path: xxx, label_path: xx}, {}, ...]
        """

        img_paths = self.get_img_paths(data_set)
        # print(img_paths, len(img_paths))

        dicts = []
        for idx, img_path in enumerate(img_paths):
            name = path.basename(path.dirname(img_path))
            gt_label = self.classes.index(name)
            r = Dict({
                "img_path": img_path,
                "gt_label": gt_label
            })
            dicts.append(r)
        return dicts

    def get_img_paths(self, data_set):
        """ 获取所有图像路径
        """
        img_paths = CText(path.join(self.data_dir, data_set)).read_lines(is_split=True)
        return [v[0] for v in img_paths]

    def split(self, train_rate=0.90, shuffle=True, reload=True):
        """
        将原始数据，分成训练、测试数据集
        :param train_rate: 训练集样本比例
        :param shuffle: 是否打乱数据集
        """

        assert train_rate <= 1 + 1e-6, "train rate should be < 1."

        img_paths = []
        for name in self.classes:
            img_paths += listdir(path.join(self.data_dir, name),
                                 filter="$|".join(self.SUFFIXES) + "$",
                                 real_path=True)
        # print("img_paths:", len(img_paths))

        if shuffle:
            random.shuffle(img_paths)

        total_cnt = len(img_paths)
        train_file = CText(path.join(self.data_dir, self.TRAIN_FILE), is_clear=True)
        valid_file = CText(path.join(self.data_dir, self.VALID_FILE), is_clear=True)
        for idx, file_path in enumerate(img_paths):
            if idx < int(total_cnt * train_rate):
                train_file.append(file_path + "\n")
            else:
                valid_file.append(file_path + "\n")

        if reload:
            self.reload()

    def vis(self, ret_dir=None, cnt=100, random=True):
        vis_name = "valid_vis" if self.test_mode else "train_vis"
        ret_dir = ret_dir if ret_dir is not None else path.join(self.data_dir, vis_name)
        shutil.rmtree(ret_dir, ignore_errors=True)
        print("vis ret_dir:", ret_dir)
        os.makedirs(ret_dir, exist_ok=True)

        cnt = min(len(self), len(self) if cnt is None else cnt)
        if not random:
            arry = np.arange(cnt)
        else:
            arry = np.random.choice(np.arange(len(self)), cnt, replace=False)

        for i in arry:
            item = self._data_dicts[i]
            img_path = item.img_path
            gt_label = item.gt_label
            # print("img_path:", img_path, gt_label)

            name = self.classes[gt_label]
            os.makedirs(path.join(ret_dir, name), exist_ok=True)
            shutil.copy(img_path, path.join(ret_dir, name, path.basename(img_path)))

    def export_to_vis_data(self, ret_dir):
        mode = "valid" if self.test_mode else "train"
        data_set_dir = path.join(ret_dir, mode)
        print(data_set_dir, len(self))
        self.vis(data_set_dir, cnt=None)

        self.test_mode = not self.test_mode
        mode = "valid" if self.test_mode else "train"
        self.reload()
        print(data_set_dir, len(self))
        data_set_dir = path.join(ret_dir, mode)
        self.vis(data_set_dir, cnt=None)

        self.test_mode = not self.test_mode
        shutil.copy(path.join(self.data_dir, self.NAME_FILE), path.join(ret_dir, self.NAME_FILE))


if __name__ == '__main__':
    from idata.datasets import DCFG

    data_dir = path.join(DCFG.root_dir, DCFG.cls_data)
    dataset = ClsData(data_dir, test_mode=False, need_path=True)
    print("total cnt: ", len(dataset))
    # datasets.set_names()
    print(dataset.class_idx)
    # datasets.split(0.9)
    # datasets.vis()
    # datasets.export_to_vis_data("/home/temp/clsdata/d2")
    pass
