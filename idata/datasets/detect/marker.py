#!/usr/bin/env python3
# coding: utf-8

"""
Desc: 标准工具数据解析验证、导出至yolo格式
"""

import os
import shutil
import random
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.detect.yolo import YoloData

__all__ = ["ImageMarker"]

"""
标注工具生成结果,解析验证、导出：
    格式1：
    root_dir:
        img1.jpg, img2.jpg, img3.jpg, ...
        Result:
            img1.txt, img2.txt, img3.txt, ...
    格式2：
    root_dir:
        img1.jpg, img2.jpg, img3.jpg, ...
        Result_New:
            img1.txt, img2.txt, img3.txt, ...
        定义：ANNO_DIR = Result_New
         # dataset.ANNO_DIR = "Result2"
         # dataset.reload()
    格式3：
    root_dir:
        img1.jpg, img2.jpg, img3.jpg, ...
        img1.txt, img2.txt, img3.txt, ...
 
    .txt信息格式:
        cls_id cx cy bw bh\n
        cls_id cx cy bw bh\n
        ...
        值域：[0~1]
"""


@DATASETS.register_module(force=True)
class ImageMarker(YoloData):
    """标注工具结果验证、导出。
       ImageMarker格式没有训练、验证集之分。没有test_mode参数。
    """

    NAME = "ImageMarker"  # 数据集名称
    ANNO_DIR = "Result"  # 根目录下标注文件夹名称

    def get_img_paths(self, data_set):
        return listdir(self.data_dir, filter="$|".join(self.SUFFIXES) + "$", real_path=True)

    def get_label_path(self, img_path):
        self.ANNO_DIR = "Result" if self.ANNO_DIR is None else self.ANNO_DIR
        label_dir = path.join(self.data_dir, self.ANNO_DIR)
        if not path.exists(label_dir):
            label_dir = self.data_dir
        return path.join(label_dir, keyname(img_path) + ".txt")

    def export_to_yolo_format(self, yolo_data_dir, train_rate=0.90, shuffle=True):
        """导出为yolo训练格式，拷贝图像和标签
        """

        img_dir = path.join(yolo_data_dir, self.IMAGES_DIR)
        os.makedirs(img_dir, exist_ok=True)
        all_file = CText(path.join(yolo_data_dir, "all.txt"), is_clear=True)
        train_file = CText(path.join(yolo_data_dir, self.TRAIN_FILE), is_clear=True)
        valid_file = CText(path.join(yolo_data_dir, self.VALID_FILE), is_clear=True)

        file_indexes = list(range(len(self._data_dicts)))
        if shuffle:
            random.shuffle(file_indexes)

        total_cnt = len(self)
        for i, idx in enumerate(file_indexes, 1):
            if i % 500 == 0:
                print("export: %d / %d = %.3f" % (i, total_cnt, i / total_cnt))

            item = self._data_dicts[idx]
            img_path = item["img_path"]
            label_path = item["label_path"]
            # print(i, item)

            # copy file
            ret_img_path = path.join(img_dir, path.basename(img_path))
            shutil.copyfile(img_path, ret_img_path)
            ret_label_path = keyname(ret_img_path, real_path=True) + ".txt"
            shutil.copyfile(label_path, ret_label_path)

            # write info
            if i < int(total_cnt * train_rate):
                train_file.append(ret_img_path + "\n")
            else:
                valid_file.append(ret_img_path + "\n")
            all_file.append(ret_img_path + "\n")

        if self.classes is not None:
            name_file = CText(path.join(yolo_data_dir, self.NAME_FILE), is_clear=True)
            for name in self.classes:
                name_file.append(name + "\n")

        print("export ok.")


if __name__ == "__main__":
    # data_dir = "/home/temp/data/maker1"
    # dataset = YoloData(data_dir, test_mode=True, need_path=True)
    # # dataset.ANNO_DIR = "Result2"
    # # dataset.reload()
    # print("total cnt: ", len(dataset))

    data_dir = "/home/temp/data/maker2"
    dataset = ImageMarker(data_dir, test_mode=True, need_path=True)
    print("total cnt: ", len(dataset))
    # dataset.set_names(["1", "2", "3", "4", "5", "6", "7", "8"])
    # dataset.export_to_yolo_format(r"/home/temp/data/stand_test")
    # dataset.vis(cnt=1)
