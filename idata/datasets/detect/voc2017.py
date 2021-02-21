#!/usr/bin/env python3
# coding: utf-8

import os
import cv2
import xml.etree.ElementTree as ET
from easydict import EasyDict as edict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["VOC2017"]

"""
    标准VOC数据集格式。
    添加参数：
        NAME_FILE：类别文件名，若存在，则解析该文件中类别名称，覆盖默认值；
"""


@DATASETS.register_module(force=True)
class VOC2017(BaseDataset):
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                   "cat", "chair", "cow", "diningtable", "dog", "horse",
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor")

    NAME = "VOC2017"
    TRAIN_SET = "ImageSets/Main/train.txt"  # 2007_000027， 2007_000032, ...
    TEST_SET = "ImageSets/Main/val.txt"
    NAME_FILE = "ImageSets/names.txt"

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None, need_path=False,
                 suffix=".jpg"):  # VOC默认train/val .txt文件中不含文件名，此处需要补充".jpg"
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path
        self.suffix = suffix

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(VOC2017, self).__init__()

    def _data_prepare(self):
        if self.test_mode:
            return self._load_dicts(self.TEST_SET)
        else:
            return self._load_dicts(self.TRAIN_SET)

    def _load_meta(self):
        name_file = path.join(self.data_dir, self.NAME_FILE)
        if not path.exists(name_file):
            return list(self.VOC_CLASSES)
        names = parse_txt_file(name_file)
        print("new names:", names)
        return names

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = cv2.imread(item.img_path)
        target = self.anno_parse(item.label_path)

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

        anno_file = path.join(self.data_dir, data_set)
        # print("anno_file:", anno_file)
        items = CText(anno_file).read_lines()
        # print("total cnt is:", len(items))

        file_paths = [path.join(self.data_dir, "JPEGImages", item + self.suffix) for item in items]
        dicts = []
        for idx, file_path in enumerate(file_paths):
            label_path = path.join(self.data_dir, "Annotations",
                                   os.path.splitext(path.basename(file_path))[0] + ".xml")
            r = edict({
                "img_path": file_path,
                "label_path": label_path,
            })
            # print("r:", r)
            dicts.append(r)
        return dicts

    def anno_parse(self, xml_file):
        """ 标签解析
        """
        annos, shape = self.xml_parse(xml_file)

        rets = []
        for anno in annos:
            [name, x1, y1, x2, y2] = anno
            if name not in self.classes:
                print("warn: %s not in self.classes, it will be ignored." % name)
                continue
            cls = self.classes.index(name)
            rets.append([cls, x1, y1, x2, y2])
        return rets

    @staticmethod
    def xml_parse(xml_file):
        """ 返回xml文件解析的原始结果
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)
        # print(size, width, height, depth)

        annos = []
        for obj in root.findall("object"):
            name = obj.find("name").text

            bnd_box = obj.find("bndbox")
            x1 = int(bnd_box.find("xmin").text)
            y1 = int(bnd_box.find("ymin").text)
            x2 = int(bnd_box.find("xmax").text)
            y2 = int(bnd_box.find("ymax").text)
            annos.append([name, x1, y1, x2, y2])
        return annos, (height, width, depth)


if __name__ == '__main__':
    # data_dir = r"/home/datasets/Cam_Flat/voc2017"
    # dataset = VOC2017(data_dir, test_mode=False, need_path=True,suffix=".jpg")

    data_dir = r"/home/data/datasets/Cam_Flat/safety_helmet/safety_helmet_voc_200416"
    dataset = VOC2017(data_dir, test_mode=True, need_path=True, suffix=".jpg")
    print("class_idx：", dataset.class_idx)
    print("classes: ", dataset.classes)
    print(dataset[0])
    print(len(dataset))
    # dataset.vis(random=True, ret_dir="/home/innno/voc_vis")
    pass
