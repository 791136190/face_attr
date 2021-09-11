import sys

from torch._C import device
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import yaml
import torch
import torch.nn as nn
import numpy as np

class BCELoss(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = 1e-12

    def forward(self, logits, target, mask=None):
        # logits: [N, *], target: [N, *]

        # logits = torch.sigmoid(logits)
        loss = -self.pos_weight * target * torch.log(logits + self.eps) - (1 - target) * torch.log(1 - logits + self.eps)
        # loss = torch.relu(logits) - logits * target + torch.log(1 + torch.exp(-torch.abs(logits)))
        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2, alpha=0.75):
        super(VFLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, mask=None):
 
        loss = self.loss_fcn(pred, true)
 
        pred_prob = pred # torch.sigmoid(pred)  # prob from logits

        pos_mask = (true > 0.0).float()

        focal_weight = pos_mask * true  + (1.0 - pos_mask) * self.alpha  * torch.abs(true - pred_prob) ** self.gamma
        
        loss *= focal_weight

        if mask is not None:
            loss = loss * mask
 
        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, mask=None):
        loss = self.loss_fcn(pred, true)

        pred_prob = pred # torch.sigmoid(pred)  # prob from logits
        # alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        # loss *= alpha_factor * modulating_factor
        loss *=  modulating_factor

        if mask is not None:
            loss = loss * mask
 
        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, (mask > 0).sum())
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2, reduction='mean'):
        super(WingLoss, self).__init__()
        self.reduction = reduction
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, pred, target, mask=None):
        abs_diff = torch.abs(pred - target)
        flag = (abs_diff < self.w).float()
        loss = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, mask.sum())
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.5, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target, mask=None):
        abs_diff = torch.abs(pred - target)
        cond = abs_diff < self.beta
        loss = torch.where(cond, 0.5 * abs_diff ** 2 / self.beta, abs_diff - 0.5 * self.beta)

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            # loss = loss.mean()
            loss = loss.sum() / max(1, (mask > 0).sum())
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class ComputeLoss:
    def __init__(self, model, cfg):
        super(ComputeLoss, self).__init__()
        self.device = next(model.parameters()).device  # get model device
        self.imgs = cfg['image_size']

        # self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        # self.BCEcls = nn.BCELoss(reduction='none')
        self.BCEcls = BCELoss()
        # self.BCEcls = QFocalLoss(BCELoss())
        
        self.smt_age = SmoothL1Loss(beta=3)
        # self.smt_age = BCELoss()

        self.smt_land = WingLoss(w=1.0 / self.imgs)

        # score_loss, gender_loss, age_loss, land_loss, glass_loss, smile_loss, hat_loss, mask_loss
        self.score_gain = cfg['score_gain']
        self.gender_gain = cfg['gender_gain']
        self.age_gain = cfg['age_gain']
        self.land_gain = cfg['land_gain']
        self.glass_gain = cfg['glass_gain']
        self.smile_gain = cfg['smile_gain']
        self.hat_gain = cfg['hat_gain']
        self.mask_gain = cfg['mask_gain']

        self.age_scale = torch.tensor([     10, 
        10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
        10,         10,         10,         10,        9.4,        6.3,        5.2,        3.1,        1.8,        1.4, 
       1.3,        1.1,        1.1,          1,          1,        1.1,        1.1,        1.2,        1.2,        1.2, 
       1.2,        1.3,        1.4,        1.5,        1.5,        1.5,        1.6,        1.6,        1.7,        2.1, 
       2.2,        2.2,        2.2,        2.3,        2.2,        2.3,        2.6,        2.6,        2.7,        2.9, 
       3.1,        3.3,        3.4,        3.6,        3.7,        4.4,        5.6,        6.3,        7.2,        7.1, 
       8.1,        9.3,         10,         10,         10,         10,         10,         10,         10,         10, 
        10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
        10,         10,         10,         10,         10,         10,         10,         10,         10,         10, 
        10,         10,         10,         10,         10,         10,         10,         10,         10,         10], device=self.device)

        # self.age_scale = torch.ones(101, device=self.device) * torch.mean(self.age_scale)
        
    def __call__(self, preds, targets):
        # print(len(preds), len(targets), targets.shape)

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
        # print(score_mask.shape, gender_mask.shape, age_mask.shape, land_mask.shape, glass_mask.shape, smile_mask.shape, hat_mask.shape, mask_mask.shape)

        score_loss = self.BCEcls(score_pred, score_label, score_mask) * self.score_gain
        gender_loss = self.BCEcls(gender_pred, gender_label, gender_mask) * self.gender_gain
        age_loss = self.smt_age(age_pred * 100, age_label, age_mask * self.age_scale[age_label.long()]) * self.age_gain
        land_loss = self.smt_land(land_pred, land_label, land_mask) * self.land_gain
        glass_loss = self.BCEcls(glass_pred, glass_label, glass_mask) * self.glass_gain
        smile_loss = self.BCEcls(smile_pred, smile_label, smile_mask) * self.smile_gain
        hat_loss = self.BCEcls(hat_pred, hat_label,  hat_mask) * self.hat_gain
        mask_loss = self.BCEcls(mask_pred, mask_label, mask_mask) * self.mask_gain

        # print(score_loss, gender_loss, age_loss, land_loss, glass_loss, smile_loss, hat_loss, mask_loss)
        loss = (score_loss + gender_loss + age_loss + land_loss + glass_loss + smile_loss + hat_loss + mask_loss)
        return loss, torch.stack((score_loss, gender_loss, age_loss, land_loss, glass_loss, smile_loss, hat_loss, mask_loss)).detach()

if __name__ == "__main__":
    config_file = "configs/face_attr.yaml"
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("end loss process !!!")