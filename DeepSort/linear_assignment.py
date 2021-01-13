# coding:utf-8
from __future__ import division
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from . import kalman_filter

INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections,
                      track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match

    # gated_distance->
    # 1. cosine distance
    # 2. 马氏距离
    # 得到代价矩阵

    # iou_cost->
    # 仅仅计算track和detection之间的iou距离
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)

    # gated_distance中设置距离中最高上限，这里最远距离实在deepsort中max_dist参数设置的
    # 默认max_dist=0.2，距离越小越好

    # iou_cost情况下，max_distance的设置对应tracker中的max_iou_distance
    # 默认max_iou_distance=0.7，结果为1-iou，所以越小越好
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # 匈牙利算法或者KM算法
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # 这几个for循环用于对匹配结果进行筛选，得到匹配和未匹配的结果
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections,
                     track_indices=None, detection_indices=None):
    # 级联匹配
    # 1 分配track_indices和detection_indices
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    # cascade_depth = max_age 默认70
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        # 2 级联匹配核心函数
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections)
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices,
                     gated_cost=INFTY_COST, only_position=False):
    # 门控矩阵：计算卡尔曼滤波的状态分布和测量值之间的距离，对代价矩阵进行限制
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix