
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import cv2
import numpy as np
import re
import random
import math
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from copy import deepcopy

def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        from torchstat import stat

        img = torch.zeros((1, 3, img_size, img_size), device=next(model.parameters()).device)  # input
        macs, params = profile(deepcopy(model), inputs=(img,), verbose=False)#[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        # fs = ', %.1f GFLOPS' % (macs * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        if model.training is False:
            sta = stat(model, (3, img_size[0], img_size[1]))

        macs = macs / 1E6
        flops = macs * 2
        params = params / 1E3
        fs = ', %.2f MMACS, %.2f MFLOPS, %.2f KB' % (macs, flops, params)  # 640x640 GFLOPS

    except ImportError as e:
        print("ImportError:", e)
        fs = ''
    except Exception as e:
        print("Exception:", e)
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
 
def decode_image(path):
    try:
        image_src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        image_src = None
    return image_src

def resize_image(image, width_new = 1280, height_new = 720):
    height, width = image.shape[0], image.shape[1]
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def get_number_after_str(string, part):
    number = re.findall((r"(?<=%s\:)\d+\.?\d*" % part), string) #提取指定字符后数字
    number = list(map(float, number))
    if len(number) < 1:
        return -1
    
    return number[0]

def get_iou_1vsN(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter + 1e-12)
    return ovr

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # torch.manual_seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

    if seed == 0:  # slower, more reproducible
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        cudnn.benchmark, cudnn.deterministic = True, False

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)

class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def load_checkpoint(model, filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    print('==> Loading parameters from checkpoint %s' % (filename))
    model.load_state_dict(torch.load(filename))

def inline_accuracy(y_true, y_prob, thr=0.5):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > thr
    return (y_true == y_prob).sum() / max(y_true.size(0), 1)

def get_accuracy(preds, targets):
    score_pred, gender_pred, age_pred, land_pred, glass_pred, smile_pred, hat_pred, mask_pred = preds
    # print(score_pred.shape, gender_pred.shape, age_pred.shape, land_pred.shape, glass_pred.shape, smile_pred.shape, hat_pred.shape, mask_pred.shape)
    
    score_label = targets[:, 0:1]
    gender_label = targets[:, 1:2]
    age_label = targets[:, 2:3]
    land_label = targets[:, 3:13]
    glass_label = targets[:, 13:14]
    smile_label = targets[:, 14:15]
    hat_label = targets[:, 15:16]
    mask_label = targets[:, 16:17]
    # print(score_label.shape, gender_label.shape, age_label.shape, land_label.shape, glass_label.shape, smile_label.shape, hat_label.shape, mask_label.shape)

    score_mask = (score_label != -1)
    gender_mask = (gender_label != -1)
    age_mask = (age_label != -1)
    land_mask = (land_label != -1)
    glass_mask = (glass_label != -1)
    smile_mask = (smile_label != -1)
    hat_mask = (hat_label != -1)
    mask_mask = (mask_label != -1)

    score_acc = inline_accuracy(score_label[score_mask], score_pred[score_mask], 0.5)
    gender_acc = inline_accuracy(gender_label[gender_mask], gender_pred[gender_mask], 0.5)
    glass_acc = inline_accuracy(glass_label[glass_mask], glass_pred[glass_mask], 0.5)
    smile_acc = inline_accuracy(smile_label[smile_mask], smile_pred[smile_mask], 0.5)
    hat_acc = inline_accuracy(hat_label[hat_mask], hat_pred[hat_mask], 0.5)
    mask_acc = inline_accuracy(mask_label[mask_mask], mask_pred[mask_mask], 0.5)
    age_acc = (age_label[age_mask] - age_pred[age_mask] * 100).abs().mean()
    land_acc = (land_label[land_mask] - land_pred[land_mask]).abs().mean() * 64

    # print(score_acc, smile_acc, age_acc, land_acc)
    acc = torch.stack((score_acc, gender_acc, age_acc, land_acc, glass_acc, smile_acc, hat_acc, mask_acc)).detach()
    acc = torch.nan_to_num(acc)

    return acc

if __name__ == '__main__':

    print("end process cv utils !!!")