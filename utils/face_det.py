
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def py_nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def decode_det_outs(layerOutputs, det_size=None, src_size=None, con_thr=0.3, nms_thr=0.5):
    boxes = []

    per_anchor_c = 1 + 4 + 1

    anchors = [[192, 240],  # s==32, 16, 8
               [48, 60],
               [12, 15]]
    anchors = np.reshape(np.array(anchors), (-1, 2))

    anchor_idx = 0

    for output in layerOutputs:  # n c h w -> 1 6 13 13
        anchor = anchors[anchor_idx]
        anchor_idx += 1
        for detection in output:  # c h w -> 6 13 13
            detection = detection.transpose((1, 2, 0))  # 6 13 13 -> 13 13 6
            detection = detection.reshape((detection.shape[0], detection.shape[1], -1, per_anchor_c))  # 13 13 6 -> 13 13 1 6
            for h in range(detection.shape[0]):
                for w in range(detection.shape[1]):
                    for a in range(detection.shape[2]):
                        cur_det = detection[h, w, a]
                        score_pred = sigmoid(cur_det[4])
                        if score_pred >= con_thr:
                            x_pred = (w + cur_det[0]) / detection.shape[1]
                            y_pred = (h + cur_det[1]) / detection.shape[0]
                            w_pred = np.exp(cur_det[2]) * anchor[a * 2 + 0] / det_size[0]
                            h_pred = np.exp(cur_det[3]) * anchor[a * 2 + 1] / det_size[1]

                            half_w = w_pred * 0.5
                            half_h = h_pred * 0.5
                            x0 = np.clip(x_pred - half_w, 0, 1)
                            y0 = np.clip(y_pred - half_h, 0, 1)
                            x1 = np.clip(x_pred + half_w, 0, 1)
                            y1 = np.clip(y_pred + half_h, 0, 1)
                            boxes.append([x0, y0, x1, y1, score_pred])

    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 5)
    boxes[:, 0:4] = boxes[:, 0:4] * ((src_size[0], src_size[1]) * 2)

    keep = py_nms(boxes, thresh=nms_thr)
    boxes = boxes[keep]

    return boxes

class FaceDet(object):
    def __init__(self, conf_threshold):

        self.net = cv2.dnn.readNet('weights/face_model/FaceDet.caffemodel', 'weights/face_model/FaceDet.prototxt')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outNames = self.net.getUnconnectedOutLayersNames()

        self.conf_threshold = conf_threshold
        self.det_size = (416, 416)

    def __call__(self, frame):

        blob = cv2.dnn.blobFromImage(frame, size=self.det_size, swapRB=True, ddepth=cv2.CV_8U)
        self.net.setInput(blob=blob, scalefactor=None, mean=None)
        outs = self.net.forward(self.outNames)

        # (x0, y0, x1, y1, s) in src image
        face_boxs = decode_det_outs(outs, self.det_size, (frame.shape[1], frame.shape[0]), self.conf_threshold)

        return face_boxs


if __name__ == '__main__':
    
    print("end process face det!!!")