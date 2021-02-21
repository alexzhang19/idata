#!/usr/bin/env python3
# coding: utf-8

"""
@File      : path.py
@Author    : alex
@Date      : 2021/2/4
@Desc      : 
"""

import os
import re
import shutil
import platform

path = os.path
desktop = r"C:\Users\Administrator\Desktop"

__all__ = [
    # utils
    "path", "desktop", "is_set", "is_contain_zh",

    # path
    "path_split", "key_name", "suffix",

    # os
    "mkdir", "cp", "rm", "mv", "ls", "pwd", "walk_file", "symlink",

    # other
    "IMG_SUFFIX",
]


# utils
def is_set(val_name: str) -> bool:
    """
    判断系统中某一变量名是否存在.
    param val_name: 变量名字符串
    return: True or False.
    """

    try:
        type(eval(val_name))
        return True
    except:
        return False


def is_contain_zh(word: str) -> bool:
    """
    是否有中文字符判断
    param word: 字符串
    return: True or False.
    """

    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(word)


# path
def path_split(str_path: str) -> list:
    """
    路径拆分，按目录存储至rets中.
    str_path: 路径
    return: list[item]
    """

    rets = []
    while True:
        str_path, item = path.split(str_path)
        if item is "":
            if str_path is not "":
                rets.insert(0, str_path)
            break
        rets.insert(0, item)
    return rets


def key_name(file_path, real_path=False):
    if real_path:
        return os.path.splitext(file_path)[0]
    else:
        return os.path.splitext(path.basename(file_path))[0]


def suffix(file_path):
    return os.path.splitext(file_path)[1]


# os
def mkdir(dir_name: str, mode: int = 0o777) -> None:
    """
    创建文件夹
    dir_name: 文件夹目录路径
    """

    if dir_name == "":
        return
    dir_name = path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    import platform

    if platform.system() == "Linux":
        if os.path.lexists(dst) and overwrite:
            os.remove(dst)
        os.symlink(src, dst, **kwargs)
    else:
        cp(src, dst)


def cp(src: str, dist: str, is_print: bool = True) -> None:
    """
    文件、文件夹拷贝
    src: 源文件、文件夹地址
    dist: 目标文件、文件夹地址
    return: None
    """

    if path.isfile(src):
        dirname = path.dirname(dist)
        if not path.isdir(dirname):
            mkdir(dirname)
        if is_print:
            print('cp file:\r\n', 'src:' + src + '\r\n', 'dist:' + dist + '\r\n')
        return shutil.copy(src, dist)
    elif path.exists(src):
        if is_print:
            print('cp folder:\r\n', 'src:' + src + '\r\n', 'dist:' + dist + '\r\n')
        if path.exists(dist):
            shutil.rmtree(dist)
        return shutil.copytree(src, dist)
    else:
        print(src + " not existence!")


def rm(dist: str, _async: bool = False) -> None:
    """
    删除文件or目录
    dist: 地址
    """

    print('rm ' + dist + '\n')
    if not path.exists(dist):
        return

    if not _async:
        shutil.rmtree(dist, ignore_errors=True)
        return

    from threading import Thread

    def run():
        temp = dist + ".part"
        if path.exists(temp):
            shutil.rmtree(temp, ignore_errors=True)

        shutil.move(dist, temp)
        if platform.system() == "Windows":
            shutil.rmtree(temp, ignore_errors=True)
            return

        trash_dir = path.join(path.expanduser('~'), ".local/share/Trash/files")
        dst_file = path.join(trash_dir, path.basename(dist))
        if path.exists(dst_file):
            shutil.rmtree(dst_file, ignore_errors=True)
        shutil.move(temp, dst_file)

    t1 = Thread(target=run, args=())
    t1.start()


def mv(src: str, dist: str) -> None:
    """
    移动src到dist
    """

    print('mv:')
    print('src:', src)
    print('dist:', dist)
    return shutil.move(src, dist)


def ls(dir_name: str, filter: str = None, real_path=False) -> list:
    rets = os.listdir(dir_name)
    if filter is not None:
        rets = [ret for ret in rets if re.search(filter, ret)]
    if real_path:
        rets = [path.join(dir_name, ret) for ret in rets]
    return sorted(rets)


def pwd() -> str:
    """
    当前文件路径.
    """

    return os.path.realpath(__file__)


def walk_file(dir_path: str, sub_dir: str = "", filter: str = "") -> list:
    """
    递归获取文件
    dir_path: 目标文件夹路径
    sub_dir: 子文件夹名
    filter: 过滤器
    return: list[文件路径]
    """

    file_paths = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if sub_dir and not re.search(sub_dir, dirpath):
            continue
        for filename in filenames:
            if filter and not re.search(filter, filename):
                continue
            file_paths.append(os.path.join(dirpath, filename))
    return sorted(file_paths)


IMG_SUFFIX = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]
