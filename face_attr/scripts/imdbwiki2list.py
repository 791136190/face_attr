
import os
import sys
import numpy as np
from numpy.core.fromnumeric import argpartition

from numpy.lib import utils

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import scipy.io as scio
import re

import cv2

from utils import face_det 
from utils import cv_utils


def do_imdbwiki():
    FaceDet = face_det.FaceDet(conf_threshold=0.4)
    
    root_path = "/home/hu/work/CV/dataset/imdbwiki/"
    
    image_save_root = "/home/hu/work/CV/dataset/imdbwiki/images/"
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)
    
    fw = open("dataset/imdbwiki_list.txt", 'w')
    
    alpha = 8
    crop_image_size = 64

    cur_index = 0

    label_txt = ['wiki', 'imdb']
    for cur_txt in label_txt:
        mat_file = root_path + '%s_crop/%s.mat' % (cur_txt, cur_txt)

        mat_data = scio.loadmat(mat_file)
        a = (mat_data[cur_txt][0][0])
        for i in range(len(a[2][0])):
            image_name = a[2][0][i][0]
            gender = a[3][0][i]
            # name = list(name)
            # print(image_name, gender)

            # age_str = image_name.split('_') # ['17/10000217', '1981-05-05', '2009.jpg']
            # print(age_str)

            # age_left = age_str[1].split('-')[0]
            # age_right= age_str[-1].split('.')[0]
            # print(age_left, age_right)
            # age = int(age_right) - int(age_left)
            
            age_str = image_name[len(image_name) - 22: len(image_name) - 4]
            age_info = re.findall(r"\d+\.?\d*", age_str)
            # print(age_info)
            age = int(age_info[4]) - int(age_info[1])
            
            if age < 0 or age > 100:
                continue
            if 'nan' == gender:
                continue
            # print(age)
            
            cur_image_path = root_path + (('%s_crop/') % cur_txt) + image_name # 17/10000217_1981-05-05_2009.jpg
            
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

            if len(face_boxs) != 1:
                continue
                face_boxs = np.array([[0, 0, s_w - 1, s_h - 1, 1.0]])
            
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

                cropped_im = image_src[y1:y2 + 1, x1:x2 + 1, :]
                resized_im = cv2.resize(cropped_im, (crop_image_size, crop_image_size), interpolation=cv2.INTER_LINEAR)

                cur_save_path = image_save_root + ("%s.jpg" % cur_index)

                fw.write(cur_save_path + '|score:1.00|age:%.2f|gender:%.2f\n' % (float(age), float(gender)))

                cv2.imwrite(cur_save_path, resized_im)

                # string = cur_save_path + '|score:1.00|age:%.2f|gender:%.2f\n' % (float(age), float(gender))
                # number = cv_utils.get_number_after_str(string=string, part='age')
                # print('转化为数字:',number)

                # x1 = int(box[0] / s_w * d_w)
                # y1 = int(box[1] / s_h * d_h)
                # x2 = int(box[2] / s_w * d_w)
                # y2 = int(box[3] / s_h * d_h)
            #     cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.putText(image_draw, "face:%.2f" % (box[4]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # cv2.imshow("image_draw", image_draw)
            # cv2.waitKey(0)

            cur_index += 1

            if cur_index % 100 == 0:
                print("do imdbwiki %d" % (cur_index))

    fw.close()
    print("do imdbwiki ok")

if __name__ == '__main__':

    do_imdbwiki()

    print("end process imdbwiki dataset !!!")