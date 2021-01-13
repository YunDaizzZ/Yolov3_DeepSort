# coding:utf-8
from __future__ import division

class TrackState:

    # 新创建的轨迹在条件完备前归类为1，确定后更改为2，不再存在的轨迹归类为3
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    # 存储一个轨迹的信息，包含(x,y,a,h) & v
    def __init__(self, mean, covariance, track_id, n_init, max_age, class_label, feature=None):
        self.mean = mean  # 初始状态分布的均值向量（好像也就是框的位置和速度信息）
        self.covariance = covariance  # 初始状态分布的协方差矩阵
        self.track_id = track_id  # 唯一标识符
        self.hits = 1  # 代表连续确认多少次，判断是否由不确定态转为确定态
        self.age = 1  # 同下（貌似重复了）
        self.time_since_update = 0  # 判断是否删除
        self.class_label = class_label

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        # x, y, w/h, h -> top left x, top left y, width, height
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        return ret

    def to_tlbr(self):
        # x, y, w/h, h -> top left x, top left y, width, height -> top, left, bottom, right
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]

        return ret

    def predict(self, kf):
        # kf预测
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        # kf更新
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        # 将轨迹标记为删除
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        # 判断轨迹状态是否是暂定
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        # 判断轨迹状态是否是已确认
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        # 判断轨迹状态是否是需要删除
        return self.state == TrackState.Deleted