#!/usr/bin/env python3
# coding: utf-8

"""
@File      : base.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import os
import cv2
import shutil
import numpy as np
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

__all__ = ["BaseDataset"]
path = os.path


class BaseDataset(Dataset, metaclass=ABCMeta):
    NAME = "BaseDataset"
    _repr_indent = 4  # tab
    SUFFIXES = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]  # 支持的图片后缀名

    def __init__(self):
        super(BaseDataset, self).__init__()
        if not os.path.exists(self.data_dir):
            raise ValueError(f"data_dir: {self.data_dir} not exists.")

        self.classes = self._load_meta()  # 加载classes类别名称
        self._data_dicts = self._data_prepare()

    @abstractmethod
    def _load_meta(self):
        """ 加载类别文件，返回类别数组
        """

        raise NotImplementedError

    @abstractmethod
    def _data_prepare(self):
        """ 加载数据集，返回list形式图片路径及相关信息。
        """

        raise NotImplementedError

    def __len__(self):
        """ 返回数据集长度
        """

        return len(self._data_dicts)

    def __getitem__(self, idx):
        """ 返回数据集中索引为idx的元素，img, gt-labels, [img_path]
        """

        raise NotImplementedError

    def __str__(self):
        """  print(obj) or print(str(obj))
        """

        return "DataSet object (name: %s)" % self.NAME

    def __repr__(self):
        """ print(repr(obj)), debug info
        """

        head = "Dataset " + self.__class__.__name__
        body = ["Number of datasets points: {}".format(self.__len__())]
        if self.data_dir is not None:
            body.append("Root location: {}".format(self.data_dir))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return "Split: {}".format("Test" if self.test_mode is True else "Train")

    @property
    def class_idx(self):
        """ 属性值，类别source_map索引
        """

        if self.classes is None:
            return
        return {i: v for i, v in enumerate(self.classes)}

    def reload(self):
        self.classes = self._load_meta()  # 加载classes类别名称
        self._data_dicts = self._data_prepare()

    @staticmethod
    def help():
        print("The description text for the datasets comes from the readme.txt "
              "in the datasets directory.")

    @staticmethod
    def img_read(img_path):
        """ 读图统一接口，方便更换图像读取库
        """

        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    pass
