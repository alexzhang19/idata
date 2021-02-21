#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import random
from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["ClsDataVis"]

"""
    分类数据集格式2：
    datasets:
        train:
            cls1_dir, cls2_dir, cls3_dir, cls4_dir, ...
        valid:
            cls1_dir, cls2_dir, cls3_dir, cls4_dir, ...
        NAME_FILE(names.txt)：由用户标注时准备,名字列表 # 可省，正确使用前必须生成
"""


@DATASETS.register_module(force=True)
class ClsDataVis(BaseDataset):
    NAME = "ClsDataVis"
    TRAIN_DIR = "train"  # 训练列表
    VALID_DIR = "valid"  # 测试列表
    NAME_FILE = "names.txt"  # 生成的名字列表文件名

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None, need_path=False):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(ClsDataVis, self).__init__()

    def _data_prepare(self):
        if self.test_mode:
            return self._load_dicts(self.VALID_DIR)
        else:
            return self._load_dicts(self.TRAIN_DIR)

    def _load_meta(self):
        name_file = path.join(self.data_dir, self.NAME_FILE)
        # print("name_file: ", name_file)
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
        # print("class name is: ", self.classes)
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
        img_paths = []
        for name in self.classes:
            item_dir = path.join(self.data_dir, data_set, name)
            if not path.exists(item_dir):
                continue
            img_paths += ls(item_dir,
                            filter="$|".join(self.SUFFIXES) + "$",
                            real_path=True)
        random.shuffle(img_paths)
        return img_paths

    def export_to_cls_data(self, ret_dir):
        def export():
            mode = "valid.txt" if self.test_mode else "train.txt"
            # print(path.join(ret_dir, mode), len(self))
            cfile = CText(path.join(ret_dir, mode), is_clear=True)
            for item in self._data_dicts:
                img_path = item.img_path
                gt_label = item.gt_label
                ret_item_dir = path.join(path.join(ret_dir, self.classes[gt_label]))
                os.makedirs(ret_item_dir, exist_ok=True)
                ret_img_path = path.join(ret_item_dir, path.basename(img_path))
                shutil.copy(img_path, ret_img_path)
                cfile.append(ret_img_path + "\n")

        os.makedirs(ret_dir, exist_ok=True)
        export()

        self.test_mode = not self.test_mode
        self.reload()
        export()

        self.test_mode = not self.test_mode
        shutil.copy(path.join(self.data_dir, self.NAME_FILE), path.join(ret_dir, self.NAME_FILE))


if __name__ == '__main__':
    from idata.datasets import DCFG

    data_dir = path.join(DCFG.root_dir, DCFG.cls_data_vis)
    dataset = ClsDataVis(data_dir, test_mode=True, need_path=True)
    print("total cnt: ", len(dataset))
    print(dataset.class_idx)
    # print(datasets[0])

    dataset.export_to_cls_data(path.join(DCFG.root_dir, "cls_vis_export"))
    pass
