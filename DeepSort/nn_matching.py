# coding:utf-8
from __future__ import division
import numpy as np

def _pdist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        # 需要将余弦相似度转化成类似欧氏距离的余弦距离
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        # 求向量范式，默认L2范式，等同于向量的欧氏距离
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)

    # 余弦距离 = 1 - 余弦相似度
    return 1. - np.dot(a, b.T)

def _nn_euclidean_distance(x, y):
    distances = _pdist(x, y)

    return np.maximum(0.0, distances.min(axis=0))

def _nn_cosine_distrance(x, y):
    distances = _cosine_distance(x, y)

    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):
    # 对于每个目标，返回一个最近的距离
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distrance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")

        self.budget = budget  # budget预算，控制feature的多少
        self.matching_threshold = matching_threshold  # 在级联匹配的函数中调用
        self.samples = {}  # 字典 {id->feature list}

    def partial_fit(self, features, targets, active_targets):
        # 部分拟合，用新的数据更新测量距离
        # 在特征集更新模块部分调用，tracker.update()中
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            # 对应目标下添加新的feature，更新feature集合
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]  # 每个类最多budget个，超过的忽略
        self.samples = {k: self.samples[k] for k in active_targets}  # 筛选激活的目标

    def distance(self, features, targets):
        # 比较feature和target的距离，返回一个代价矩阵
        # 在匹配阶段，将distance封装为gated_metric，进行外观信息+运动信息
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)

        return cost_matrix