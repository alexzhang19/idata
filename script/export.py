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
    import onnx
    from onnxsim import simplify

    model = torchvision.models.resnet18(False)
    model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))  # 增加 softmax 作为输出结果
    model.eval()

    example = torch.rand(1, 3, 112, 112)
    input_names = ["input_0"]
    output_names = ["output_0"]
    output_path = "resnet18.onnx"
    torch.onnx.export(model, example,
                      output_path,
                      verbose=True,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)
    print("out_put_shape:", model(example).shape)

    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')


def parallel_model_weights_to_model():
    """ 多卡训练的模型保存为单卡模型，去掉模型名字中的 `module.`
    """

    new_state_dict = OrderedDict()
    state_dict = torch.load("weights.pth", map_location='cpu')

    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    for k, v in state_dict.items():
        # print("k:", k, k[7:])
        name = k.replace('module.', '')  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # model.load_state_dict(new_state_dict)
    return new_state_dict


def model_to_parallel_model(model_path, map_location=None):
    checkpoint = torch.load(model_path, map_location)
    # print(checkpoint.keys())
    for key in list(checkpoint.keys()):
        checkpoint["module." + key.replace('module.', '')] = checkpoint.pop(key)
    return checkpoint


def load_pretrained_weights(model_path, n_cls, map_location=None):
    checkpoint = torch.load(model_path, map_location=map_location)

    # 最后一层为线性分类层:
    checkpoint["module.linear.weight"] = checkpoint["module.linear.weight"][:n_cls, :]
    checkpoint["module.linear.bias"] = checkpoint["module.linear.bias"][:n_cls]
    return checkpoint


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


def onnx_to_pb():
    """ tensorflow-gpu2.3.1, onnx=1.7.0, onnx_tf=1.7.0,
        onnx-10, pytorch=1.6, cuda=10.1
    """

    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load("../t_cls/mnist_fc.onnx")  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph("mnist_fc.pb")  # export the model


def h5_to_pb(h5_path, export_path):
    import tensorflow as tf

    model = tf.keras.models.load_model(h5_path, compile=False)
    model.summary()
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )


if __name__ == "__main__":
    # export_torch_script()
    export_onnx()
    pass
