#!/usr/bin/env python3
# coding: utf-8

"""
@File      : wrapper.py
@Author    : alex
@Date      : 2020/4/29
@Desc      : 
"""

import os
import functools

path = os.path

__all__ = ["file_create_safety"]


def file_create_safety(file_path: str):
    """确保新建文件时，上级目录存在."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """装饰器函数体，在函数之前调用."""
            if "." not in path.basename(file_path):
                return
            up_dir = path.dirname(file_path)
            if not path.exists(up_dir):
                os.makedirs(up_dir)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def log(func):
    """无参数装饰器"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('call %s():' % func.__name__)
        print('args = {}'.format(*args))
        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    @log
    def test(p):
        print(test.__name__ + " param: " + p)


    test("I'm a param")


    @file_create_safety("param")
    def test_with_param(p):
        print(test_with_param.__name__)
    pass
