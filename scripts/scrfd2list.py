
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
import linecache
import cv2

from utils import face_det 
from utils import cv_utils

def get_file_len(file_path):
    return int(len(open(file_path).readlines()))#空行不计算在内

def do_scrfd():
    FaceDet = face_det.FaceDet(conf_threshold=0.4)

    image_save_root = "/home/hu/work/CV/dataset/scrfd/images/"
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)

    fw = open("dataset/scrfd_list.txt", 'w')
    
    alpha = 8
    crop_image_size = 64

    cur_index = 0
    save_index = 0
    p_idx = 0
    n_idx = 0

    image_sets = ['train', 'val']
    # image_sets = ['val']

    fileDir = "/home/hu/work/CV/dataset/widerface/"
    for sets in image_sets:

        file_path = fileDir + ('WIDER_%s/labelv2.txt' % sets)
        lines = get_file_len(file_path)

        for i in range(lines):
            line = linecache.getline(file_path, i)
            if re.search('.jpg', line):
                cur_index += 1

                line = line.split(' ')[1]
                position = line.index('/')
                file_name = line[position + 1: -4]
                folder_name = line[:position]
                # print(file_name, folder_name)

                i += 1
                # face_count = int(linecache.getline(file_path, i))

                face_count = 0
                t = 0
                while True:
                    count_line = linecache.getline(file_path, i + t)
                    if re.search('.jpg', count_line) or i+t > lines:
                        # print(file_name, i, t, i+t, lines)
                        break
                    else:
                        face_count += 1
                        t += 1
                # print(face_count)

                cur_image_path = fileDir + ("WIDER_%s/images/"%sets) + folder_name + "/" + file_name + '.jpg'
                # print(cur_image_path)

                image_src = cv_utils.decode_image(cur_image_path)
                if image_src is None:
                    print("get none image->", cur_image_path)
                    continue

                s_h, s_w, _ = image_src.shape

                # (x0, y0, x1, y1, s) in src image
                face_boxs = FaceDet(frame=image_src.copy())

                # boxes = np.zeros((face_count, 4), dtype=np.int32)
                boxes = []
                for j in range(face_count):
                    # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                    box_line = linecache.getline(file_path, i + j)  # x1, y1, w, h, x1,y1 为人脸框左上角的坐标
                    line_info = re.split(' ', box_line.strip())
                    # x1, y1, x2, y2, (land xy), score
                    x1 = float(line_info[0])
                    y1 = float(line_info[1])
                    x2 = float(line_info[2])
                    y2 = float(line_info[3])

                    if len(line_info) > 4:
                        l_eye_x = float(line_info[4])
                        l_eye_y = float(line_info[5])

                        r_eye_x = float(line_info[7])
                        r_eye_y = float(line_info[8])

                        node_x = float(line_info[10])
                        node_y = float(line_info[11])

                        l_mouth_x = float(line_info[13])
                        l_mouth_y = float(line_info[14])

                        r_mouth_x = float(line_info[16])
                        r_mouth_y = float(line_info[17])

                        score = float(line_info[18])
                    else:
                        l_eye_x = -1.0
                        l_eye_y = -1.0

                        r_eye_x = -1.0
                        r_eye_y = -1.0

                        node_x = -1.0
                        node_y = -1.0

                        l_mouth_x = -1.0
                        l_mouth_y = -1.0

                        r_mouth_x = -1.0
                        r_mouth_y = -1.0

                        score = 1.0

                    # print(x1, y1, w, h, l_eye_x, l_eye_y, r_eye_x, r_eye_y, node_x, node_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, score)
                    # exit(0)
                    boxes.append([x1, y1, x2, y2, l_eye_x, l_eye_y, r_eye_x, r_eye_y, node_x, node_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, score])

                i += i + j + 1

                gt_boxes_list = np.array(boxes, dtype=np.float32).reshape(-1, 4 + 10 + 1)

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
                    # print(cur_image_path)
                    # print(len(gt_boxes_list), len(face_boxs), cur_image_path)
                    
                    if len(gt_boxes_list) < 1 or np.max(Iou) < 0.4:
                        n_idx += 1
                        fw.write(cur_save_path + '|score:0.00\n')
                        cv2.imwrite(cur_save_path, resized_im)

                    else:
                        # find gt_box with the highest iou
                        idx = np.argmax(Iou)
                        assigned_gt = gt_boxes_list[idx]
                        x1_gt, y1_gt, x2_gt, y2_gt, l_eye_x, l_eye_y, r_eye_x, r_eye_y, node_x, node_y, \
                        l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y, score = assigned_gt

                        if l_eye_x > -1 and l_eye_y > -1:
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
                        
                        
                            fw.write(cur_save_path + '|score:1.00|lex:%.2f|ley:%.2f|rex:%.2f|rey:%.2f|nosex:%.2f|nosey:%.2f|lmx:%.2f|lmy:%.2f|rmx:%.2f|rmy:%.2f\n' % 
                            ((l_eye_x - x1) / float(size_w), (l_eye_y - y1) / float(size_h),
                            (r_eye_x - x1) / float(size_w), (r_eye_y - y1) / float(size_h),
                            (node_x - x1) / float(size_w), (node_y - y1) / float(size_h),
                            (l_mouth_x - x1) / float(size_w), (l_mouth_y - y1) / float(size_h),
                            (r_mouth_x - x1) / float(size_w), (r_mouth_y - y1) / float(size_h)))
                        else:
                            fw.write(cur_save_path + '|score:1.00\n')
                        p_idx += 1
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
                    print("do scrfd %d in %d p_idx:%d, n_idx:%d" % (cur_index, lines, p_idx, n_idx))

    fw.close()
    print("do scrfd ok")

if __name__ == '__main__':

    do_scrfd()

    print("end process scrfd dataset !!!")