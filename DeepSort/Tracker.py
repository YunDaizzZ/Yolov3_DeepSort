# coding:utf-8
from __future__ import division
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

class Tracker(object):
    # 是一个多目标tracker，保存了多个track轨迹
    # 负责调用kf预测track新状态+匹配+初始化第一帧
    # Tracker调用update或predict的时候，其中每个track也会调用各自的update或predict
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric  # 用于计算距离
        self.max_iou_distance = max_iou_distance  # iou匹配的最大iou
        self.max_age = max_age  # 指定级联匹配的cascade_depth参数
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []  # 保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id

    def predict(self):
        # 遍历每个track都进行一次预测
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # 进行测量的更新和轨迹管理
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set
        # 1 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # track更新对应的detection
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 2 针对未匹配的tracker，调用mark_missed标记
        # track失配，若待定则删除，若update时间很久也删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 3 针对未匹配的detection，detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 得到最新的tracks列表，保存的是标记为confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # 获取所有confirmed状态的track_id
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # 将tracks列表拼接到features列表
            # 获取每个feature对应的track_id
            targets += [track.track_id for _ in track.features]
            track.features = []

        # 距离度量中的特征集更新
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 主要功能是进行匹配，找到匹配的，未匹配的部分
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 用于计算track和detection之间的距离，代价函数
            # 需要使用在KM算法之前
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # 1 通过最近邻计算出代价矩阵 cosine_distance
            cost_matrix = self.metric.distance(features, targets)
            # 2 计算马氏距离得到新的状态矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        # 划分不同轨迹的状态
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 进行级联匹配，得到匹配的track，不匹配的track，不匹配的detection
        # 仅对confirmed的轨迹进行级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed_tracks)

        # 对所有状态为unconfirmed的轨迹和刚刚没有匹配上的轨迹组合成iou_track_candidates进行iou匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1  # 刚刚没有匹配上
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1  # 已经很久没有匹配上
        ]
        # 对级联匹配中还没有匹配成功的目标再进行iou匹配
        # 虽然和级联匹配中使用的都是min_cost_matching作为核心，但这里使用的metric是iou cost和上面不同
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # 组合两部分match得到的结果

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, detection.class_labels, detection.feature))
        self._next_id += 1