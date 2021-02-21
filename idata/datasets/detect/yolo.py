#!/usr/bin/env python3
# coding: utf-8

import random
from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

__all__ = ["YoloData"]

"""
    1.YOLO标准化的darknet训练数据集格式：STANDARD
    dataset:
        IMAGES_DIR(imgs): 1.jpg、1.txt, 2.jpg、2.txt,... 
        TRAIN_FILE(train.txt):  图片文件绝对路径列表 # 可省
        VALID_FILE(valid.txt): 图片文件绝对路径列表 # 可省
        NAME_FILE(names.txt)：由用户标注时准备,名字列表 # 可省
        info.data：用户准备，形如： # 可省
            classes = 7
            train = dataset/train.txt
            valid = dataset/valid.txt
            names = dataset/names.txt
            backup = backup
    标注格式：cls cx cy bw wh\n
             cls cx cy bw wh\n
             ...
            值域：[0~1]
    
    扩展格式：
    设置ANNO_DIR字段：数据集解析时将默认从ANNO_DIR目录下读取与图像对应的标记文件；
    dataset:
        IMAGES_DIR(imgs): 1.jpg, 2.jpg,... 
        ANNO_DIR(anno): 1.txt, 2.txt,... 
        TRAIN_FILE、VALID_FILE、NAME_FILE 可省
    
    add arg:
        normal： 设置为True时，__getitem__返回量化至0~1的标准yolo格式标签。默认为False，与其它数据集格式保持一致。
"""


@DATASETS.register_module(force=True)
class YoloData(BaseDataset):
    NAME = "YoloData"
    IMAGES_DIR = "imgs"  # 图像及标准文件夹
    TRAIN_FILE = "train.txt"  # 生成训练文件列表
    VALID_FILE = "valid.txt"  # 生成测试文件列表
    NAME_FILE = "names.txt"  # 生成的名字列表文件名
    ANNO_DIR = None  # 标记文件目录

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None, need_path=False,
                 normal=False):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path
        self.normal = normal  # 坐标标签是否量化至0~1

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(YoloData, self).__init__()

    def _data_prepare(self):
        if self.test_mode:
            return self._load_dicts(self.VALID_FILE)
        else:
            return self._load_dicts(self.TRAIN_FILE)

    def _load_meta(self):
        name_file = path.join(self.data_dir, self.NAME_FILE)
        if not path.exists(name_file):
            return None

        # print("name_file：", name_file)
        names = parse_txt_file(name_file)
        # print("names:", names)
        return names

    def set_names(self, names):
        self.classes = names
        print("class name is: ", self.classes)

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = self.img_read(item.img_path)
        img_shape = None if self.normal else img.shape
        target = self.anno_parse(item.label_path, img_shape)

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

        dicts = []
        for idx, img_path in enumerate(img_paths):
            anno_path = self.get_label_path(img_path)
            # print(idx, img_path, anno_path)
            if not path.exists(anno_path):
                continue

            r = Dict({
                "img_path": img_path,
                "label_path": anno_path
            })
            dicts.append(r)
        return dicts

    def get_img_paths(self, data_set):
        """ 获取所有图像路径
        """
        img_paths = CText(path.join(self.data_dir, data_set)).read_lines(is_split=True)
        return [v[0] for v in img_paths]

    def get_label_path(self, img_path):
        if self.ANNO_DIR is None:
            return path.join(path.dirname(img_path), keyname(img_path) + ".txt")
        else:
            return path.join(self.data_dir, self.ANNO_DIR, keyname(img_path) + ".txt")

    @staticmethod
    def anno_parse(file_path, shape=None):
        """输出标记格式： cls, x1, y1, x2, y2, shape=None时输出原始标签中至0~1的坐标，shape不为None时，输出绝对坐标
        """

        h, w = shape[:2] if shape is not None else (1, 1)
        items = CText(file_path).read_lines(is_split=True)

        gt_label = []
        for item in items:
            cls, cx, cy, iw, ih = item
            cx = float(cx) * w
            cy = float(cy) * h
            iw = float(iw) * w
            ih = float(ih) * h
            if shape is None:
                gt_label.append([int(cls), cx, cy, iw, ih])
                continue

            x1 = max(int(cx - iw / 2), 0)
            y1 = max(int(cy - ih / 2), 0)
            x2 = min(int(cx + iw / 2), w - 1)
            y2 = min(int(cy + ih / 2), h - 1)
            gt_label.append([int(cls), x1, y1, x2, y2])
        return gt_label

    @staticmethod
    def x1y1x2y2_to_cxcybwbh(shape, rgn):
        h, w = shape[:2]
        [x1, y1, x2, y2] = rgn
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return cx, cy, bw, bh

    @staticmethod
    def cxcybwbh_to_x1y1x2y2(shape, rgn):
        h, w = shape[:2]
        [cx, cy, bw, bh] = rgn
        cx = float(cx) * w
        cy = float(cy) * h
        bw = float(bw) * w
        bh = float(bh) * h

        x1 = int(max(0, cx - bw // 2))
        y1 = int(max(0, cy - bh // 2))
        x2 = int(min(w, cx + bw // 2))
        y2 = int(min(h, cy + bh // 2))
        return x1, y1, x2, y2

    def extra_repr(self):
        return "Split: {}".format("Test" if self.test_mode is True else "Train")

    def split(self, train_rate=0.90, shuffle=True, reload=False):
        """
        将原始数据，分成训练、测试数据集
        :param train_rate: 训练集样本比例
        :param shuffle: 是否打乱数据集
        """

        assert train_rate <= 1 + 1e-6, "train rate should be < 1."

        img_paths = listdir(path.join(self.data_dir, self.IMAGES_DIR), filter="$|".join(self.SUFFIXES) + "$",
                            real_path=True)
        # print("img_paths:", img_paths, len(img_paths))
        file_paths = []
        for idx, img_path in enumerate(img_paths):
            anno_path = self.get_label_path(img_path)
            if not path.exists(anno_path):
                print("warn: anno_path '%s' not exists, it will be ignored." % anno_path)
                continue
            file_paths.append(img_path)
        total_cnt = len(file_paths)
        # print("total img cnt:", len(img_paths), "total cnt:", total_cnt)

        if shuffle:
            random.shuffle(file_paths)

        train_file = CText(path.join(self.data_dir, self.TRAIN_FILE), is_clear=True)
        valid_file = CText(path.join(self.data_dir, self.VALID_FILE), is_clear=True)
        for idx, file_path in enumerate(file_paths):
            if idx < int(total_cnt * train_rate):
                train_file.append(file_path + "\n")
            else:
                valid_file.append(file_path + "\n")

        if reload:
            self.reload()

    def update_file_path(self):
        data_set = self.VALID_FILE if self.test_mode else self.TRAIN_FILE
        file_paths = CText(path.join(self.data_dir, data_set)).read_lines()
        img_names = [path.basename(v) for v in file_paths]
        img_paths = []
        for img_name in img_names:
            img_path = path.join(self.data_dir, self.IMAGES_DIR, img_name)
            if path.exists(img_path):
                img_paths.append(img_path)

        if len(img_paths) == 0:
            return

        cfile = CText(path.join(self.data_dir, data_set), is_clear=True)
        for img_path in img_paths:
            cfile.append(img_path + "\n")

        self.reload()


if __name__ == '__main__':
    # data_dir = "/home/temp/data/stand1"
    # dataset = YoloData(data_dir, test_mode=True, need_path=True)
    # print("total cnt: ", len(dataset))

    data_dir = r"/home/temp/data/stand/stand1"
    dataset = YoloData(data_dir, test_mode=False, need_path=True)
    print("total cnt: ", len(dataset))
    # dataset.update_file_path()
    dataset.vis(cnt=1)
    print("total cnt: ", len(dataset))

    # print(dataset[0])
    # dataset.ANNO_DIR = "anno"
    # dataset.reload()
    # print("total cnt: ", len(dataset))
    pass
