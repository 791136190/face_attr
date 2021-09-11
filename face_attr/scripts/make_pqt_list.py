
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
import linecache

from utils import face_det 
from utils import cv_utils


def make_qat_list():

    with open("dataset/face_attr_val.txt", 'r') as fw:
        _list = fw.readlines()

    fw = open("dataset/qat_list.txt", 'w')
    for l in _list:
        file_name = l.split('|')[0] + '\n'
        fw.write(file_name)
    fw.close()

if __name__ == '__main__':
    make_qat_list()

    print("end process make_qat_list !!!")