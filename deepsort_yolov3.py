# coding:utf-8
from __future__ import division
import time
import numpy as np
import cv2
from PIL import Image
from yolo import YOLO
from DeepSort import nn_matching
from DeepSort import Tracker
from DeepSort.detection import Detection
from utils import feature_extractor
from collections import deque

pts = [deque(maxlen=30) for _ in range(9999)]
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

mp4 = cv2.VideoCapture('/home/bhap/Documents/Video/test3.MP4')
model_path = '/home/bhap/Pytorch_test/YoloV3/history/20200825/yolo_weights.pth'
CAMERA = True

def main(yolo):

    # 设置参数
    max_cosine_distance = 0.5  # 余弦距离的控制阈值
    nn_budget = None

    counter = []

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker.Tracker(metric)
    extractor = feature_extractor.Extractor(model_path)

    capture = cv2.VideoCapture(0)

    fps = 0.0

    while True:

        t1 = time.time()

        if CAMERA:
            ref, frame = capture.read()
        else:
            ref, frame = mp4.read()
        if ref != True:
            break

        # 格式转变 BGR2RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        boxs, class_labels = yolo.deepsort_detect_image(frame)
        features = extractor.create_box_encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature, class_label) for bbox, feature, class_label in zip(boxs, features, class_labels)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        i = int(0)
        indexIDs = []

        for det in detections:
            bbox = det.to_tlbr()
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

            frame = cv2.putText(frame, str(det.class_labels), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (255, 255, 255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            frame = cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, color, 2)
            # frame = cv2.putText(frame, str(track.class_label), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, color, 2)
            # 如果在这里打印目标类别信息 有可能会因为目标关联的问题而出现类别错误 这点需要后续修正

            i += 1
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 2
            # center point
            frame = cv2.circle(frame, (center), 1, color, thickness)

            # 画移动轨迹
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(16 / float(j + 1)) * 2)
                frame = cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

        count = len(set(counter))
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        frame = cv2.putText(frame, "Total Object Counter: " + str(count), (0, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        frame = cv2.putText(frame, "Current Object Counter: " + str(i), (0, 90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)

        cv2.imshow('mp4', frame)

        c = cv2.waitKey(30) & 0xff

        if c == 27:
            capture.release()
            break

if __name__ == '__main__':
    main(YOLO())
