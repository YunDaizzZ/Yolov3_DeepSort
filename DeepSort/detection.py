# coding:utf-8
from __future__ import division
import numpy as np

class Detection(object):
    # 用于保存检测框，并提供不同的bbox格式转换法
    def __init__(self, tlwh, confidence, feature, class_labels):
        self.tlwh = np.asarray(tlwh, dtype=np.float)  # top left x, top left y, width, height
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_labels = class_labels

    def to_tlbr(self):
        # top left x, top left y, width, height -> top, left, bottom, right
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]

        return ret

    def to_xyah(self):
        # top left x, top left y, width, height -> x, y, w/h, h
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]

        return ret