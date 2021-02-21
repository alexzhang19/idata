#!/usr/bin/env python3
# coding: utf-8

from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["NormalData"]

"""
    Normal数据，无标签背景图像，格式：
    dataset:
        1.jpg、2.jpg、3.jpg,... 
"""


@DATASETS.register_module(force=True)
class NormalData(BaseDataset):
    NAME = "NormalData"

    def __init__(self, data_dir: str, transform=None, target_transform=None, need_path=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path
        self.test_mode = False
        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(NormalData, self).__init__()

    def _data_prepare(self):
        dicts = []
        file_paths = listdir(self.data_dir, filter="$|".join(self.SUFFIXES) + "$", real_path=True)
        for file_path in file_paths:
            dicts.append(Dict({
                "img_path": file_path,
            }))
        return dicts

    def _load_meta(self):
        return None

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = self.img_read(item.img_path)
        target = []

        if self.transform is not None:
            img = self.transform(img)

        if self.need_path:
            return img, target, item.img_path
        else:
            return img, target

    def vis(self, ret_dir=None, cnt=100, random=False):
        # TODO: You don't need to implement
        pass


if __name__ == '__main__':
    data_dir = "/home/temp/data/stand/normal"
    dataset = NormalData(data_dir, need_path=True)
    print("total cnt: ", len(dataset))
    pass
