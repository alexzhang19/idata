#!/usr/bin/env python3
# coding: utf-8

"""
@File      : cifar.py
@author    : alex
@Date      : 2020/6/11
@Desc      :
"""

import os
import cv2
import copy
import shutil
import torchvision
import numpy as np
from addict import Dict
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = []

path = os.path


@DATASETS.register_module(force=True)
class CIFAR100(BaseDataset):
    NAME = "CIFAR100"

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform

        os.makedirs(self.data_dir, exist_ok=True)
        # step1: _load_meta()，加载self.classes、self.class_to_idx信息
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(CIFAR100, self).__init__()

    def _data_prepare(self):
        fileids = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                train=not self.test_mode,
                                                download=True,
                                                transform=None)
        self.CLASS_NAMES = copy.deepcopy(fileids.classes)  # 0~99
        return self._dict_prepare(fileids)

    def _load_meta(self):
        fileids = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                train=True,
                                                download=True,
                                                transform=None)
        return fileids.classes  # 类别名称

    def __getitem__(self, index):
        """
        return: numpy, int
        """

        item = self._data_dicts[index]
        img = item.img
        target = item.gt_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    @staticmethod
    def _dict_prepare(fileids):
        dicts = []
        for idx, fileid in enumerate(zip(fileids.data, fileids.targets)):
            data, target = fileid
            # https://blog.csdn.net/weixin_34910922/article/details/107922225
            r = Dict({
                "img": data,
                "gt_label": target,  # 类别标记
            })
            dicts.append(r)
        return dicts

    def extra_repr(self):
        return "Split: {}".format("Test" if self.test_mode is True else "Train")

    def vis(self, ret_dir=None, cnt=100, random=True):
        vis_name = "valid_vis" if self.test_mode else "train_vis"
        ret_dir = ret_dir if ret_dir is not None else path.join(self.data_dir, vis_name)
        shutil.rmtree(ret_dir, ignore_errors=True)
        os.makedirs(ret_dir, exist_ok=True)

        cnt = min(len(self), len(self) if cnt is None else cnt)
        print("vis ret_dir:", ret_dir, "total cnt is: ", cnt)
        if not random:
            arry = np.arange(cnt)
        else:
            arry = np.random.choice(np.arange(len(self)), cnt, replace=False)

        for i in arry:
            data = self[i]
            (img, target) = data[:2]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            item_dir = path.join(ret_dir, self.classes[target])
            os.makedirs(item_dir, exist_ok=True)
            cv2.imwrite(path.join(item_dir, "cifar_%05d.jpg" % i), img)

        cfile = CText(path.join(ret_dir, "names.txt"), is_clear=True)
        for name in self.classes:
            cfile.append(name + "\n")


if __name__ == '__main__':
    from idata.fileio import *
    from idata import DCFG

    data_dir = DCFG.cifar100
    print("data_dir:", data_dir)

    dataset = CIFAR100(data_dir)
    print("classes", dataset.classes)
    print("class_to_idx", dataset.class_idx)
    print(repr(dataset))
    print(dataset)
    print("total count:", len(dataset))
    # print("datasets", datasets[0])
    dataset.vis(cnt=100)

    # 类外测试
    # from alcore.datasets import CIFAR100, DATASETS
    # from alcore.utils.registry import Registry, build_from_cfg
    # from alcore.utils.config import Config
    #
    # cfg = Config.fromfile("/home/project/alcore/configs/_base_/datasets/cifar100.py")
    # datasets = build_from_cfg(cfg.datasets.train, DATASETS)
    # print("datasets", datasets[0])
    pass
