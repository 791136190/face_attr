import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import random
import numpy as np

def split_attr_labels(dataset_list, train_file, val_file, val_ratio, shuffle = True):

    file_list = []

    for filename in dataset_list:

        with open(filename, 'r') as t:
            t = t.read().strip().splitlines()
            file_list += t

    if shuffle:
        random.shuffle(file_list)
    
    for line in file_list[:10]:
        print(line)

    total_num = len(file_list)
    
    val_num = val_ratio if val_ratio > 1 else int(total_num * val_ratio)
    train_num = total_num - val_num

    print(total_num, train_num, val_num)

    with open(val_file, 'w') as f:
        # f.write(str(file_list[:val_num]))
        f.write('\n'.join(file_list[:val_num]))
    f.close()

    with open(train_file, 'w') as f:
        f.write('\n'.join(file_list[val_num:]))
    f.close()
        

if __name__ == "__main__":

    train_file = "dataset/face_attr_train.txt"
    val_file = "dataset/face_attr_val.txt"
    dataset_list = ["dataset/afad_list.txt", "dataset/cacd2000_list.txt", "dataset/celeba_list.txt", "dataset/idcard_list.txt", 
                    "dataset/imdbwiki_list.txt", "dataset/scrfd_list.txt", "dataset/utkface_list.txt", "dataset/maskface_list.txt", 
                    "dataset/rmfd_list.txt"]
    val_ratio = 20 * 10000

    split_attr_labels(dataset_list, train_file, val_file, val_ratio, shuffle=True)
    print("end split process !!!")