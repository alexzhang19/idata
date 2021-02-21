#!/usr/bin/env python3
# coding: utf-8

"""
@File      : labelme.py
@Author    : alex
@Date      : 2020/9/30
@Desc      : labelme 标记格式解析
"""

import cv2
import random
from addict import Dict
from idata.datasets.build import DATASETS
from idata.fileio import *

"""
    充当接口的labelme标注工具输出。
"""


@DATASETS.register_module()
class LabelMe(object):
    NAME = "LabelMe"

    TRAIN_SET = "train"
    TEST_SET = "valid"
    NAME_FILE = "names.txt"  # 生成的名字列表文件名

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform

        # step1: 调用_data_prepare()，加载self._data_dicts
        # step2: _load_meta()，加载self.classes、self.class_to_idx信息
        super(LabelMe, self).__init__()

    def _data_prepare(self):
        if self.test_mode:
            return self._load_annotations(self.TEST_SET)
        else:
            return self._load_annotations(self.TRAIN_SET)

    def _load_meta(self):
        items = CText(path.join(self.data_dir, self.NAME_FILE)).read_lines(is_split=True)
        items = [v[0].split("-") for v in items]
        self.class_to_idx = {int(v[0]): v[1] for v in items}
        self.classes = [v[1] for v in items]

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = cv2.imread(item.img_path)
        target = item.gt_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_annotations(self, data_type: str):
        """
        输出格式： [{img_path: xxx, gt_label: xx}, {}, ...]
        """
        dicts = []
        file_paths = walk_file(path.join(self.data_dir, data_type))
        for idx, file_path in enumerate(file_paths):
            target = int(path.basename(path.dirname(file_path)).split("-")[0])
            r = Dict({
                "img_path": file_path,
                "gt_label": target,  # 类别标记
            })
            dicts.append(r)
        return dicts

    def extra_repr(self):
        return "Split: {}".format("Test" if self.test_mode is True else "Train")


if __name__ == "__main__":
    pass
