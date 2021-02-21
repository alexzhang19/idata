#!/usr/bin/env python3
# coding: utf-8

import os
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


def export_torch_script():
    """ 导出为 torch script 模型
    """

    model = torchvision.models.resnet18(False)
    model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))  # 增加 softmax 作为输出结果
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    traces_script_module = torch.jit.trace(model, example)
    output = traces_script_module(torch.ones(1, 3, 224, 224))
    print('out[0:5]: ', output[0, :5])
    traces_script_module.save('traces_script_module.pt')


def export_onnx():
    """ 导出为 onnx 模型
    """

    model = torchvision.models.resnet18(False)
    model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))  # 增加 softmax 作为输出结果
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    input_names = ["input_0"]
    output_names = ["output_0"]
    torch.onnx.export(model, example,
                      "resnet18.onnx",
                      verbose=True,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)
    print("out_put_shape:", model(example).shape)


def parallel_model_weights_to_model():
    """ 多卡训练的模型保存为单卡模型，去掉模型名字中的 `module.`
    """

    new_state_dict = OrderedDict()
    state_dict = torch.load("weights.pth", map_location='cpu')
    for k, v in state_dict.items():
        # print("k:", k, k[7:])
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # model.load_state_dict(new_state_dict)
    pass


def load_unique_weights():
    model = torchvision.models.resnet18(False)
    model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))  # 增加 softmax 作为输出结果
    model.eval()

    weights_path = "resnet18.pth"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        torch.save(model.state_dict(), weights_path)

    example = torch.ones(1, 3, 224, 224)
    print(model(example), model(example).shape)


if __name__ == "__main__":
    # export_torch_script()
    pass
