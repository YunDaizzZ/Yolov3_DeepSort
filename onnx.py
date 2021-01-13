# coding:utf-8
from __future__ import division
import torch
import torch.onnx
from nets.yolo3 import YoloBody
from utils.config import Config

def pthtoonnx():
    model = YoloBody(Config)

    pthfile = '/home/bhap/Pytorch_test/YoloV3/history/20200825/yolo_weights.pth'
    dict = torch.load(pthfile)
    model.load_state_dict(dict)
    model.eval()

    dummy_input1 = torch.randn(1, 3, 416, 416)

    input_names = ["input1"]
    output_names = ["output1", "output2", "output3"]

    torch.onnx.export(model, dummy_input1, "yolov3.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=10)
    # 虽然有warning但是opset_version好像只能取10, 取11没有warning但是在c++里resize又会有error

def pthtoonnx_bs():
    model = YoloBody(Config)

    pthfile = '/home/bhap/Pytorch_test/YoloV3/history/20200825/yolo_weights.pth'
    dict = torch.load(pthfile)
    model.load_state_dict(dict)
    model.eval()

    dummy_input1 = torch.randn(1, 3, 416, 416)

    input_names = ["input1"]
    output_names = ["output1", "output2", "output3"]
    dynamic_axes = {'input1': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'},
                    'output3': {0: 'batch_size'}}

    torch.onnx.export(model, dummy_input1, "yolov3.onnx", verbose=True, input_names=input_names,
                      output_names=output_names, opset_version=10, dynamic_axes=dynamic_axes)

if __name__ == "__main__":
    dynamic_bs = True

    if dynamic_bs:
        pthtoonnx_bs()
    else:
        pthtoonnx()