import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import yaml
from utils import cv_utils
import torch
from torch.utils.data import Dataset
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import random
import math
import cv2

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, cfg, img_size=64, augment=False):
        self.augment = augment
        self.img_size = img_size
        self.path = path
        self.cfg = cfg
        self.img_files = None
        self.label_files = None
        
        # get image path labels
        self.get_all_image()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path

        label = self.label_files[index]

        # do augment
        if self.augment:
            # Augment imagespace
            img, label = self.random_perspective(img, label,
                                    self.cfg['degrees'], 
                                    self.cfg['translate'],
                                    self.cfg['scale'],
                                    self.cfg['shear'],
                                    self.cfg['perspective'])

            # Flip LR 不做上下翻转
            if random.random() < self.cfg['fliplr']:
                img = np.fliplr(img)
                
                label[3] = np.where(label[3] < 0, -1, 1 - label[3])
                label[5] = np.where(label[5] < 0, -1, 1 - label[5])
                label[7] = np.where(label[7] < 0, -1, 1 - label[7])
                label[9] = np.where(label[9] < 0, -1, 1 - label[9])
                label[11] = np.where(label[11] < 0, -1, 1 - label[11])

                # 翻转左右眼
                eye_left = np.copy(label[[3, 4]])
                mouth_left = np.copy(label[[9, 10]])
                label[[3, 4]] = label[[5, 6]]
                label[[5, 6]] = eye_left
                label[[9, 10]] = label[[11, 12]]
                label[[11, 12]] = mouth_left

            # Augment colorspace
            img = self.augment_hsv(img, hgain=self.cfg['hsv_h'], sgain=self.cfg['hsv_s'], vgain=self.cfg['hsv_v'])
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x64x64
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        labels_out = torch.from_numpy(label).float()

        return img, labels_out

    def random_perspective(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0):
        border = [-self.img_size//2, -self.img_size//2]

        height = img.shape[0]# + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] #+ border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale * 3)
        s = random.uniform(1, 1 + scale * 3)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                gray = random.randint(0, 255)
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(gray, gray, gray))

        # Transform label coordinates
        # ['score', 'gender', 'age',  'lex', 'ley', 'rex', 'rey', 'nosex', 'nosey', 'lmx', 'lmy', 'rmx', 'rmy', 'glass', 'smile', 'hat', 'mask']
        #   0     1     2     3      4     5     6     7     8     9     10    11    12  13    14    15    16
        # [ 1.   -1.   -1.    0.54  0.54  0.78  0.54  0.72  0.64  0.54  0.86  0.78  0.82 -1.   -1.   -1.   -1.  ]
        if targets[3] > -0.5 and targets[4] > -0.5:
            # warp points
            xy = np.ones((5, 3))
            xy[:, :2] = targets[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].reshape(5, 2)  # x1y1, x2y2...
            xy[:, :2] = xy[:, :2] * (self.img_size, self.img_size)

            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(1, 10)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(1, 10)

            landmarks = xy
            # mask = np.array(targets[3:13] > -0.5, dtype=np.int32) # 0 or 1
            # landmarks = landmarks * mask # 0, landmark
            # landmarks = landmarks + mask - 1 # -1, landmark

            landmarks = np.where(landmarks < 0, -1, landmarks)
            landmarks[:, [0, 2, 4, 6, 8]] = np.where(landmarks[:, [0, 2, 4, 6, 8]] > width, -1, landmarks[:, [0, 2, 4, 6, 8]])
            landmarks[:, [1, 3, 5, 7, 9]] = np.where(landmarks[:, [1, 3, 5, 7, 9]] > height, -1,landmarks[:, [1, 3, 5, 7, 9]])

            landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
            landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

            landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
            landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

            landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
            landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

            landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
            landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])

            landmarks[:, 8] = np.where(landmarks[:, 9] == -1, -1, landmarks[:, 8])
            landmarks[:, 9] = np.where(landmarks[:, 8] == -1, -1, landmarks[:, 9])

            landmarks[:, [0, 2, 4, 6, 8]] = np.where(landmarks[:, [0, 2, 4, 6, 8]] == -1, -1 * width, landmarks[:, [0, 2, 4, 6, 8]])
            landmarks[:, [1, 3, 5, 7, 9]] = np.where(landmarks[:, [1, 3, 5, 7, 9]] == -1, -1 * height,landmarks[:, [1, 3, 5, 7, 9]])

            targets[3:13] = landmarks / ((width, height) * 5)

        return img, targets

    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return img
    
    def get_mtime(self, path):
        # file_size = os.path.getsize(path)
        file_time = os.path.getmtime(path)
        return file_time

    def get_all_image(self):

        img_list = []
        labels_list = []

        cache_info = {}
        need_new_cache = True

        cache_path = self.path.replace('.txt', '.cache')

        if os.path.exists(cache_path):
            cache_info = torch.load(cache_path)
            mtime = cache_info.pop('mtime')
            print('load cache info...', mtime)

            if mtime == self.get_mtime(self.path):
                need_new_cache = False
            else:
                print('the file has no changed!!!', self.path, mtime)
        
        # updata all label info
        if need_new_cache:
            with open(self.path, 'r') as f:
                image_list = f.readlines()
                pbar = enumerate(image_list)
                pbar = tqdm(pbar, total=len(image_list), desc=self.path)
                for _, line in pbar:
                    line = line.strip()
                    line_split = line.split('|') # 绝对路径 ‘|’ 分割不同字段
                    
                    attrs = np.zeros(len(self.cfg['names']))

                    if len(line_split) > 0:
                        attr_str = line.replace(line_split[0], '')
                        for idx, attr_name in enumerate(self.cfg['names']):
                            attrs[idx] = cv_utils.get_number_after_str(attr_str, attr_name)

                        img_list.append(line_split[0])
                        labels_list.append(attrs)

                        cache_info[line_split[0]] = attrs
            
            # cache new file
            cache_info['mtime'] = self.get_mtime(self.path)
            torch.save(cache_info, cache_path)
            cache_info.pop('mtime')

        self.img_files = list(cache_info.keys()) # img_list
        self.label_files = list(cache_info.values()) # labels_list
    
def create_dataloader(cfg, path, imgsz, batch_size, augment=False, workers=8):
    data_set = LoadImagesAndLabels(path, cfg, imgsz, augment)

    batch_size = min(batch_size, len(data_set))
    workers = min(workers, os.cpu_count())
    prefetch_factor = max(batch_size // workers // 2, 2)
    print(path, ('load by btch size:%d, workers:%d, prefetch factor:%d' % (batch_size, workers, prefetch_factor)))

    data_loader = Data.DataLoader(data_set, batch_size=batch_size, shuffle=augment, num_workers=workers,
                                            prefetch_factor=prefetch_factor, pin_memory=True, drop_last=True)

    return data_loader


if __name__ == "__main__":
    config_file = "configs/face_attr.yaml"
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_loader = create_dataloader(cfg=cfg, path=cfg['val'], imgsz=cfg['image_size'], batch_size=cfg['batch_size'], workers=cfg['workers'])
    
    for i_batch, (imgs, targets) in enumerate(data_loader):
        print(i_batch)
        print(imgs.shape)
        # show target
        cls_names = cfg["names"]
        for i in range(imgs.shape[0]):
            image = imgs[i]
        #     image = (image * 255)
        #     image = np.clip(image, 0, 255)
            image = np.transpose(image.numpy(), (1, 2, 0))
            
            image = image.astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            attrs = targets[i]
            
            image = cv2.resize(image, dsize=(256, 256))

            skip = 10
            for idx, attr_name in enumerate(cls_names):
                cv2.putText(image, '%s: %.2f' % (attr_name, attrs[idx]), (5, 10 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            b_show = True
            if b_show:
                cv2.imshow('line_split[0]', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    

    print("end dataload process !!!")