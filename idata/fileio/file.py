#!/usr/bin/env python3
# coding: utf-8

"""
@File      : file.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import warnings

warnings.filterwarnings("ignore")

import os
import cv2
import copy
import json
import yaml
import pickle
import numpy as np

path = os.path
desktop = r"C:\Users\Administrator\Desktop"

__all__ = [
    # image
    "tif_read", "img_read", "img_write", "img_compress", "img_crop",

    # file
    "CText", "CJson", "CPickle", "CYaml", "py_file2dict", "parse_txt_file",

    # utils
    "get_size", "calculate_md5",
]


# image
def tif_read(img_path: str):
    """
    读取.tif文件，量化至0~255。
    param img_path: 文件路径名
    return: numpy value
    """

    import tifffile as tiff
    def scale_percentile(matrix):
        w, h, d = matrix.shape
        matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
        mins = np.percentile(matrix, 5, axis=0)
        maxs = np.percentile(matrix, 95, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        matrix = np.reshape(matrix, [w, h, d])
        matrix = matrix.clip(0, 1)
        return matrix

    try_img_rgb = tiff.imread(img_path).transpose([1, 2, 0])
    img = cv2.cvtColor((255 * scale_percentile(try_img_rgb)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img


def img_read(file_path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def img_write(file_path, img, ratio=None):
    """ .png为无损压缩，.jpg为有损压缩，
    """

    suffix = os.path.splitext(file_path)[1]
    if suffix == ".png":
        ratio = ratio if ratio is not None else 0
        # 取值范围：0~9，数值越小，压缩比越低，图片质量越高
        assert ratio >= 0 and ratio <= 9, "ratio should in 0~9"
        params = [cv2.IMWRITE_PNG_COMPRESSION, ratio]  # ratio: 0~9
    elif suffix == ".jpg":
        ratio = ratio if ratio is not None else 100
        # 取值范围：0~100，数值越小，压缩比越高，图片质量损失越严重
        assert ratio >= 0 and ratio <= 100, "ratio should in 0~100"
        params = [cv2.IMWRITE_JPEG_QUALITY, ratio]  # ratio:0~100
    else:
        raise ValueError(f"{suffix} not support compress.")
    cv2.imwrite(file_path, img, params)


def img_compress(img, suffix=".jpg", ratio=0):
    assert suffix in [".png", ".jpg"]
    if suffix == ".png":
        # 取值范围：0~9，数值越小，压缩比越低，图片质量越高
        params = [cv2.IMWRITE_PNG_COMPRESSION, ratio]  # ratio: 0~9
    elif suffix == ".jpg":
        # 取值范围：0~100，数值越小，压缩比越高，图片质量损失越严重
        params = [cv2.IMWRITE_JPEG_QUALITY, ratio]  # ratio:0~100
    else:
        raise ValueError(f"{suffix} not support compress.")
    msg = cv2.imencode(suffix, img, params)[1]
    msg = (np.array(msg)).tostring()
    img = cv2.imdecode(np.fromstring(msg, np.uint8), cv2.IMREAD_COLOR)
    # cv2.imwrite("saveImg.jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # cv2.imwrite("saveImg.png", src, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return img


def img_crop(img, rgn, ratio=0.1, min_border=15, is_draw=False):
    """ 图像按比例扩充剪切，最小扩充边界不小于min_border。
    """

    img_copy = copy.deepcopy(img)
    h, w = img_copy.shape[:2]
    [x1, y1, x2, y2] = rgn
    if is_draw:
        img_copy = cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
    iw, ih = (x2 - x1, y2 - y1)
    x_border, y_border = max(int(iw * ratio), min_border), max(int(ih * ratio), min_border)
    x1 = max(0, x1 - x_border)
    y1 = max(0, y1 - y_border)
    x2 = min(x2 + x_border, w)
    y2 = min(y2 + y_border, h)
    return img_copy[y1:y2, x1:x2, :]


# file
class CText(object):
    """
    文件操作
    """

    def __init__(self, file_path: str, is_clear: bool = False) -> None:
        self.file_path = path.realpath(file_path)
        if is_clear:
            self.clear()

    def clear(self) -> None:
        if not path.exists(self.file_path):
            return
        fp = open(self.file_path, 'r+')
        fp.truncate()
        fp.close()

    def append(self, text: str) -> None:
        fp = open(self.file_path, 'a')
        fp.write(text)
        fp.close()

    def read_lines(self, is_split: bool = False, sep: str = None) -> list:
        """
        按行读取文件内容，结果放进list
        is_split: 是否对单行内容进行拆分
        sep: 拆分标志
        return: 读取的内容
        """

        lines = []
        fp = open(self.file_path)
        for item in fp.readlines():
            if is_split:
                lines.append(item.strip().split(sep))
            else:
                lines.append(item.strip())
        fp.close()
        lines = [v for v in lines if v]
        return lines


class CJson(object):
    def __init__(self, json_path):
        self.json_path = path.realpath(json_path)

    def write(self, obj):
        with open(self.json_path, mode='w') as fp:
            fp.write(json.dumps(obj))

    def load(self, encode='UTF-8'):
        with open(self.json_path, mode='r', encoding=encode) as fp:
            return json.load(fp)


class CPickle(object):
    def __init__(self, file_path):
        self.file_path = path.realpath(file_path)
        dir_name = path.dirname(self.file_path)
        if not path.exists(dir_name):
            os.makedirs(dir_name)

    def dump(self, data):
        with open(self.file_path, 'w') as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.file_path, 'r') as f:
            return pickle.load(f)


class CYaml(object):
    def __init__(self, file_path):
        self.file_path = path.realpath(file_path)
        dir_name = path.dirname(self.file_path)
        if not path.exists(dir_name):
            os.makedirs(dir_name)

    def dump(self, data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)

    def load(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            return yaml.load(f.read())


def py_file2dict(file_path):
    """ 加载py文件，以模块的形式解析其中__all__导出的变量
    """

    import sys
    import shutil
    import tempfile
    import platform
    import os.path as osp
    from importlib import import_module

    file_path = osp.abspath(osp.expanduser(file_path))
    if not osp.exists(file_path):
        return

    if osp.splitext(file_path)[1] != ".py":
        raise IOError('Only py type are supported!')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=".py")
        if platform.system() == 'Windows':
            temp_config_file.close()

        temp_config_name = osp.basename(temp_config_file.name)
        shutil.copyfile(file_path, temp_config_file.name)

        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)

        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if name in mod.__all__
        }
        sys.path.pop(0)
        del sys.modules[temp_module_name]
        temp_config_file.close()
    return cfg_dict


def parse_txt_file(file_path, blank="blank"):
    """ 类别文件解析，支持#注释，支持两种格式：
            name：car\n person\n ...
        idx_name：0-car\n 1-person\n ...
    """

    items = CText(file_path).read_lines()
    items = [v.split("#")[0].strip() for v in items if v.split("#")[0].strip()]

    if "-" in items[0]:
        rets = [blank for _ in range(len(items))]
        for v in items:
            idx, name = v.split("-")
            idx = int(idx)
            if len(rets) <= idx:
                for _ in range(idx - len(rets) + 1):
                    rets.append(blank)
            rets[idx] = name
        items = rets
    return items


# utils
def get_size(file_path, unit="kb"):
    units = ["BIT", "KB", "MB", "GB"]
    assert unit.upper() in units, "unit should in ['BIT', 'KB', 'MB', 'GB']."

    return os.path.getsize(file_path) / (1024 ** units.index(unit.upper()))


def calculate_md5(fpath, chunk_size=1024 * 1024):
    import hashlib
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()
