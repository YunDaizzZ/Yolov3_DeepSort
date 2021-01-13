# coding:utf-8
import torch
from utils.config import Config
from nets.yolo3 import YoloBody

model = YoloBody(Config)
model.load_state_dict(torch.load("/home/bhap/Pytorch_test/YoloV3/history/20200910/Epoch27-Total_Loss9.5468-Val_Loss12.5104.pth"))

model.eval()
example = torch.rand(1, 3, 416, 416)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/bhap/Pytorch_test/YoloV3/yolov3.pt")