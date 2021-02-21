#!/usr/bin/env python3
# coding: utf-8

"""
@File      : build.py
@author    : alex
@Date      : 2020/6/10
@Desc      :
"""

import torch
from idata.utils.registry import Registry, build_from_cfg

DATASETS = Registry("datasets")
TRANSFORMS = Registry("transform")


def build_data_set(cfg, default_args=None):
    data_set = build_from_cfg(cfg, DATASETS, default_args)
    return data_set


def get_train_loader(cfg):  # cfg = cfg.datasets
    train_set = build_data_set(cfg.train)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size)
    return train_loader


def get_val_loader(cfg):  # cfg = cfg.datasets
    val_set = build_data_set(cfg.val)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=True,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size)
    return val_loader
