
import os
import sys
import numpy as np

from numpy.lib import utils

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import re

import cv2
import glob

from utils import face_det 
from utils import cv_utils

def get_all_list(root_path):
    print(root_path)

    label_boxs_txt_name = root_path + 'list_bbox_celeba.txt'
    label_attr_txt_name = root_path + 'list_attr_celeba.txt'
    label_land_txt_name = root_path + 'list_landmarks_celeba.txt'

    with open(label_boxs_txt_name, 'r') as f:
        label_boxs = f.readlines()
    with open(label_attr_txt_name, 'r') as f:
        label_attr = f.readlines()
    with open(label_land_txt_name, 'r') as f:
        label_land = f.readlines()

    return label_boxs, label_attr, label_land

def do_celeba():
    FaceDet = face_det.FaceDet(conf_threshold=0.1)

    root_path = "/home/hu/work/CV/dataset/celebface/img_celeba/"
    
    image_save_root = "/home/hu/work/CV/dataset/celebface/images/"
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)

    label_boxs, label_attr, label_land = get_all_list(root_path)

    fw = open("dataset/celeba_list.txt", 'w')
    
    alpha = 8
    crop_image_size = 64

    cur_index = -2
    save_index = 0
    p_idx = 0
    n_idx = 0

    for box, attr, land in zip(label_boxs, label_attr, label_land):
        cur_index += 1
        if cur_index < 1:
            continue

        # print(box, attr, land)
        box_line = re.split(' +', box.strip())
        # print(box_line)

        attr_line = re.split(' +', attr.strip())
        # print(attr_line)

        land_line = re.split(' +', land.strip())
        # print(land_line)

        cur_image_path = root_path + 'img_celeba/' + box_line[0] # /home/hu/work/CV/dataset/cacd2000/CACD2000/62_William_Katt_0013.jpg
        # print(cur_image_path)
        
        image_src = cv_utils.decode_image(cur_image_path)
        if image_src is None:
            print("get none image->", cur_image_path)
            continue
        
        # image_draw = cv_utils.resize_image(image_src.copy())
        # d_h, d_w, _ = image_draw.shape

        s_h, s_w, _ = image_src.shape

        # (x0, y0, x1, y1, s) in src image
        face_boxs = FaceDet(frame=image_src.copy())

        # print(face_boxs, type(face_boxs), len(face_boxs))

        # 解析gt信息
        gt_boxes = []
        box_line = box_line[1:]
        for l in range(len(box_line)):
            gt_boxes.append(int(box_line[l]))

        land_line = land_line[1:]
        for l in range(len(land_line)):
            gt_boxes.append(int(land_line[l]))

        attr_line = attr_line[1:]
        for l in range(len(attr_line)):
            attr_line[l] = int(attr_line[l])

        Eyeglasses, Male, Smiling, Wearing_Hat = int(attr_line[15]), int(attr_line[20]), int(attr_line[31]), int(attr_line[35])
        Eyeglasses = 0 if Eyeglasses < 1 else 1
        Male = 0 if Male < 1 else 1
        Smiling = 0 if Smiling < 1 else 1
        Wearing_Hat = 0 if Wearing_Hat < 1 else 1

        gt_boxes.append(Eyeglasses)
        gt_boxes.append(Male)
        gt_boxes.append(Smiling)
        gt_boxes.append(Wearing_Hat)
        gt_boxes[2] += gt_boxes[0]
        gt_boxes[3] += gt_boxes[1]

        gt_boxes_list = np.array(gt_boxes, dtype=np.float32).reshape(-1, 4 + 10 + 4)

        if len(face_boxs) < 1:
            continue
            # face_boxs = np.array([[0, 0, s_w - 1, s_h - 1, 1.0]])
        
        for box in face_boxs:
            w = box[2] - box[0]
            h = box[3] - box[1]

            box[0] = max(box[0] - w / alpha, 0)
            box[1] = max(box[1] - h / alpha, 0)
            box[2] = min(box[2] + w / alpha, s_w - 1)
            box[3] = min(box[3] + h / alpha, s_h - 1)
            
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            size_w = box[2] - box[0]
            size_h = box[3] - box[1]

            cropped_im = image_src[y1:y2 + 1, x1:x2 + 1, :]
            resized_im = cv2.resize(cropped_im, (crop_image_size, crop_image_size), interpolation=cv2.INTER_LINEAR)

            Iou = cv_utils.get_iou_1vsN(box, gt_boxes_list)

            save_index += 1
            cur_save_path = image_save_root + ("%s.jpg" % save_index)
            
            if np.max(Iou) < 0.4:
                n_idx += 1
                fw.write(cur_save_path + '|score:0.00\n')
                cv2.imwrite(cur_save_path, resized_im)

            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gt_boxes_list[idx]
                x1_gt, y1_gt, x2_gt, y2_gt, l_eye_x, l_eye_y, r_eye_x, r_eye_y, node_x, node_y, \
                l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, glass, gender, smile, hat = assigned_gt

                if l_eye_x < x1 or l_eye_x > x2:
                    continue
                if r_eye_x < x1 or r_eye_x > x2:
                    continue
                if node_x < x1 or node_x > x2:
                    continue
                if l_mouth_x < x1 or l_mouth_x > x2:
                    continue
                if r_mouth_x < x1 or r_mouth_x > x2:
                    continue

                if l_eye_y < y1 or l_eye_y > y2:
                    continue
                if r_eye_y < y1 or r_eye_y > y2:
                    continue
                if node_y < y1 or node_y > y2:
                    continue
                if l_mouth_y < y1 or l_mouth_y > y2:
                    continue
                if r_mouth_y < y1 or r_mouth_y > y2:
                    continue
                
                p_idx += 1
                fw.write(cur_save_path + '|score:1.00|glass:%.2f|gender:%.2f|smile:%.2f|hat:%.2f|lex:%.2f|ley:%.2f|rex:%.2f|rey:%.2f|nosex:%.2f|nosey:%.2f|lmx:%.2f|lmy:%.2f|rmx:%.2f|rmy:%.2f\n' % 
                (float(glass), float(gender), float(smile), float(hat),
                (l_eye_x - x1) / float(size_w), (l_eye_y - y1) / float(size_h),
                (r_eye_x - x1) / float(size_w), (r_eye_y - y1) / float(size_h),
                (node_x - x1) / float(size_w), (node_y - y1) / float(size_h),
                (l_mouth_x - x1) / float(size_w), (l_mouth_y - y1) / float(size_h),
                (r_mouth_x - x1) / float(size_w), (r_mouth_y - y1) / float(size_h)))
                cv2.imwrite(cur_save_path, resized_im)

        #     x1 = int(box[0] / s_w * d_w)
        #     y1 = int(box[1] / s_h * d_h)
        #     x2 = int(box[2] / s_w * d_w)
        #     y2 = int(box[3] / s_h * d_h)
        #     cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(image_draw, "face:%.2f" % (box[4]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # print(image_draw.shape)
        # cv2.imshow("image_draw", image_draw)
        # cv2.waitKey(0)

        if cur_index % 100 == 0:
            print("do celeba %d in %d p_idx:%d, n_idx:%d" % (cur_index, len(label_boxs), p_idx, n_idx))

    fw.close()
    print("do celeba ok")

if __name__ == '__main__':

    do_celeba()

    print("end process celeba dataset !!!")