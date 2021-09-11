import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import random
import cv2
import numpy as np
import math
import yaml

from utils import cv_utils

def calc_attr_labels(filename, cluster_number, config, shuffle = True):

    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    names = cfg['names']
    print(names, len(names))
    
    attr_all = np.zeros(len(names))
    age_all = np.zeros(101)

    with open(filename, 'r') as t:
        t = t.read().strip().splitlines()
        
        if shuffle:
            random.shuffle(t)
        
        if -1 == cluster_number:
            cluster_number = len(t)

        for i, line in enumerate(t[:cluster_number]):
            line_split = line.split('|') # 绝对路径 ‘|’ 分割不同字段
            attrs = np.zeros(len(names))

            if len(line_split) > 1:
                attr_str = line.replace(line_split[0], '')

                for idx, attr_name in enumerate(names):
                    attrs[idx] = cv_utils.get_number_after_str(attr_str, attr_name)
                    if -1 != attrs[idx]:
                        attr_all[idx] += 1
                        if 'age' == attr_name:
                            age_all[int(attrs[idx])] += 1
                    
                    # if idx >=3 and idx <= 12 and idx % 2 == 0:
                    #     cv2.circle(img, (int(attrs[idx - 1] * img_w), int(attrs[idx] * img_h)), 3, (0, 0, 255), -1)

            if i % 10000 == 0:
                print("cur %d in %d" % (i, cluster_number))
    
    # print('%12s' * len(names) % (names[0], names[1], names[2], names[3], names[0], names[0], names[0], names[0], names[0], names[0], 
    # names[0], names[0], names[0], names[0], names[0], names[0], names[0], ))
    # print('%12.3g' * 8 % (eacc[0], eacc[1], eacc[2], eacc[3], eacc[4], eacc[5], eacc[6], eacc[7]))
    # print('%12s' % (n for n in names), end='')
    for n in names:
        print('%12s' % n, end='')
    print('')
    for n in attr_all:
        print('%12g' % n, end='')
    print('')
    for n in attr_all:
        print('%12.3g' % (n / cluster_number), end='')
    print('')
    
    for i, n in enumerate(age_all):
        print('%10g' % n, end='')
        if i % 10 == 0:
            print('')
    print('max age num:', np.argmax(age_all), max(age_all))

    for i, n in enumerate(age_all):
        print('%10.2g, ' % (min(max(age_all) / (n + 1), 10)), end='')
        if i % 10 == 0:
            print('')

    print('')

def check_attr_labels(filename, cluster_number, config, shuffle = True):

    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    names = cfg['names']
    print(names, len(names))

    with open(filename, 'r') as t:
        t = t.read().strip().splitlines()
        
        if shuffle:
            random.shuffle(t)
        
        for i, line in enumerate(t[:cluster_number]):
            line_split = line.split('|') # 绝对路径 ‘|’ 分割不同字段
            attrs = np.zeros(len(names))

            if len(line_split) > 1:
                attr_str = line.replace(line_split[0], '')

                img = cv2.imread(line_split[0])
                img = cv2.resize(img, dsize=(256, 256))
                img_h, img_w, _ = img.shape

                skip = 10
                for idx, attr_name in enumerate(names):
                    attrs[idx] = cv_utils.get_number_after_str(attr_str, attr_name)
                    print(attrs[idx])
                    cv2.putText(img, '%s: %.2f' % (attr_name, attrs[idx]), (5, 10 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if idx >=3 and idx <= 12 and idx % 2 == 0:
                        cv2.circle(img, (int(attrs[idx - 1] * img_w), int(attrs[idx] * img_h)), 3, (0, 0, 255), -1)

                b_show = True
                if b_show:
                    cv2.imshow(line_split[0], img)
                    # cv2.imwrite(line_split[0].split('/')[-1], img)
                    # print(line_split[0].split('/')[-1])
                    # exit(0)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # cluster_number = -1
    # filename = "dataset/face_attr_train.txt"
    # filename = "dataset/celeba_list.txt"
    # filename = "dataset/idcard_list.txt"
    # filename = "dataset/imdbwiki_list.txt"
    # filename = "dataset/scrfd_list.txt"
    # filename = "dataset/utkface_list.txt"
    # filename = "dataset/maskface_list.txt"
    # config = "configs/face_attr.yaml"
    # check_attr_labels(filename=filename, cluster_number=cluster_number, config=config, shuffle=True)

    # calc labels info
    cluster_number = -1
    filename = "dataset/face_attr_train.txt"
    config = "configs/face_attr.yaml"
    calc_attr_labels(filename=filename, cluster_number=cluster_number, config=config, shuffle=False)
    print("end check process!!!")