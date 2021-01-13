# coding:utf-8
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from nets.darknet import Darknet
from .utils import letterbox_image
from .config import Config

def extract_image_patch(image, bbox):

    sx = bbox[0]
    sy = bbox[1]
    ex = bbox[0] + bbox[2]
    ey = bbox[1] + bbox[3]
    image = image.crop((sx, sy, ex, ey))

    img = np.array(letterbox_image(image, (Config["img_w"], Config["img_h"])))
    photo = np.array(img, dtype=np.float32)
    photo /= 255.
    photo = np.transpose(photo, (2, 0, 1))
    photo = photo.astype(np.float32)
    images = []
    images.append(photo)

    return images

class Extractor(object):
    # 初始化YOLO
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.generate()

    def generate(self):
        self.net = Darknet([1, 2, 8, 8, 4], reid=True)
        model_dict = self.net.state_dict()
        state_dict = torch.load(self.model_path)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict)
        self.net = self.net.eval()
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()

    def create_box_encoder(self, image, bboxes):
        patches = []
        for box in bboxes:
            patch = extract_image_patch(image, box)
            patches.append(patch)
        if patches:
            with torch.no_grad():
                feature = self.net(torch.from_numpy(np.squeeze(np.asarray(patches), axis=1)).cuda())
            features = feature.cpu().numpy()
        else:
            features = []

        return features