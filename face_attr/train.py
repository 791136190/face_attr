import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

from utils import cv_utils
from symbols import get_model
from utils import data_load
from utils import face_attr_loss

def train(config_file):
    print("into train func...")

    # read config
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    print("get cfg...")
    print(cfg)

    epochs = cfg['total_epochs']

    # make save dir
    train_model = config_file.split('/')[-1].split('.')[0]
    save_root = "runs/" + train_model + '/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    tb_i = 0
    tb_root = save_root + "sample"
    while os.path.exists(tb_root + str(tb_i)): # 递增
        tb_i += 1

    save_root = tb_root + str(tb_i) + '/'
    print('all train info will save in:', save_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    tb_writer = SummaryWriter(save_root)  # Tensorboard

    last_model = save_root + 'last.pt'
    best_model = save_root + 'best.pt'
    print(last_model)
    print(best_model)

    # select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set init seed
    cv_utils.init_seeds(seed=0)
    
    # get model
    model = get_model.build_model(cfg).to(device)

    # ema
    ModelEMA = cv_utils.ModelEMA(model)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for _, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if 'adam' == cfg['optimizer']:
        optimizer = optim.Adam(pg0, lr=cfg['lr_base'], betas=(cfg['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=cfg['lr_base'], momentum=cfg['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0
    del pg1
    del pg2

    # amp_scaler = torch.cuda.amp.GradScaler()

    if 'linear_lr' ==  cfg['scheduler']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg['lr_final']) + cfg['lr_final']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif 'one_cycle' ==  cfg['scheduler']:
        lf = cv_utils.one_cycle(1, cfg['lr_final'], epochs)  # cosine 1->hyp['lr_final']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif 'multi_step' ==  cfg['scheduler']:
        milestones = [int(epochs * 0.6), int(epochs * 0.9)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
    else:
        print('get', cfg['scheduler'], 'not support!!!')
    scheduler.last_epoch = - 1

    # data loader
    train_loader = data_load.create_dataloader(cfg=cfg, path=cfg['train'], imgsz=cfg['image_size'], batch_size=cfg['batch_size'], augment=True, workers=cfg['workers'])
    val_loader = data_load.create_dataloader(cfg=cfg, path=cfg['val'], imgsz=cfg['image_size'], batch_size=cfg['batch_size'], augment=False, workers=cfg['workers'])

    # loss func
    if 'face_attr' == train_model:
        compute_loss = face_attr_loss.ComputeLoss(model, cfg)

    # best acc
    best_acc = -1

    # start epoch
    for epoch in range(epochs):
        model.train()
        
        # Warmup
        nw = round(cfg['warmup_epochs'])
        if epoch < nw:
            for _, x in enumerate(optimizer.param_groups):
                x['lr'] = x['initial_lr'] * 0.1 + x['initial_lr'] * 0.9 * epoch / nw

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)

        mloss = torch.zeros(8, device=device)  # mean losses
        print(('%4s' + '%10s' * 12) % ('', 'epoch', 'mem/G', 'l/score', 'l/gender', 'l/age', 'l/land', 'l/glass', 'l/smile', 'l/hat', 'l/mask', 'batch', 'imgs'))
        
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), desc='train')
        for i_batch, (imgs, targets) in pbar:

            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device)
            
            if False:
                print(imgs.shape)
                # show target
                cls_names = cfg["names"]
                for i in range(imgs.shape[0]):
                    image = imgs[i]
                #     image = (image * 255)
                #     image = np.clip(image, 0, 255)
                    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
                    
                    image = image.astype('uint8')
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    attrs = targets[i]
                    
                    image = cv2.resize(image, dsize=(256, 256))

                    skip = 10
                    for idx, attr_name in enumerate(cls_names):
                        cv2.putText(image, '%s: %.2f' % (attr_name, attrs[idx]), (5, 10 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        if idx >=3 and idx <= 12 and idx % 2 == 0:
                            cv2.circle(image, (int(attrs[idx - 1] * image.shape[1]), int(attrs[idx] * image.shape[0])), 3, (0, 255, 0), -1)

                    b_show = True
                    if b_show:
                        cv2.imshow('line_split[0]', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            optimizer.zero_grad()

            # with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss, loss_items = compute_loss(preds, targets)

            loss.backward()
            # amp_scaler.scale(loss).backward()
            # total_norm = clip_grad_norm_(model.parameters(), optim_cfg['GRAD_NORM_CLIP'])
            
            optimizer.step()
            # amp_scaler.step(optimizer)
            # amp_scaler.update()
            ModelEMA.update(model)

            mloss = (mloss * i_batch + loss_items) / (i_batch + 1)  # update mean losses
            s = ('train: %3g/%3g' + '%10.3g' * 11) % (epoch + 1, epochs, mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            if 0 == i_batch and epoch < 3:
                img_grid = vutils.make_grid(imgs/255.0)
                tb_writer.add_image('imgs' + str(epoch), img_grid, epoch)

        # end epoch ----------------------------------------------------------------------------------------------------
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # start eval ----------------------------------------------------------------------------------------------------
        # model.eval()
        model_eval = ModelEMA.ema
        model_eval.eval()

        eloss = torch.zeros(8, device=device)  # mean losses
        eacc = torch.zeros(8, device=device)  # mean accs

        pbar = enumerate(val_loader)
        pbar = tqdm(pbar, total=len(val_loader), desc='val')
        for i_batch, (imgs, targets) in pbar:
            
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device)
            
            if False:
                print(imgs.shape)
                # show target
                cls_names = cfg["names"]
                for i in range(imgs.shape[0]):
                    image = imgs[i]
                #     image = (image * 255)
                #     image = np.clip(image, 0, 255)
                    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
                    
                    image = image.astype('uint8')
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    attrs = targets[i]
                    
                    image = cv2.resize(image, dsize=(256, 256))

                    skip = 10
                    for idx, attr_name in enumerate(cls_names):
                        cv2.putText(image, '%s: %.2f' % (attr_name, attrs[idx]), (5, 10 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        if idx >=3 and idx <= 12 and idx % 2 == 0:
                            cv2.circle(image, (int(attrs[idx - 1] * image.shape[1]), int(attrs[idx] * image.shape[0])), 3, (0, 255, 0), -1)

                    b_show = True
                    if b_show:
                        cv2.imshow('line_split[0]', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            with torch.no_grad():
                preds = model_eval(imgs)

                loss, loss_items = compute_loss(preds, targets)
                eloss = (eloss * i_batch + loss_items) / (i_batch + 1)  # update mean losses

                acc_items = cv_utils.get_accuracy(preds, targets)
                eacc = (eacc * i_batch + acc_items) / (i_batch + 1)  # update mean acc

                s = ('  val: %3g/%3g' + '%10.3g' * 11) % (epoch + 1, epochs, mem, *eloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
        
        # start log ----------------------------------------------------------------------------------------------------
        tags = ['train/score_loss', 'train/gender_loss', 'train/age_loss', 'train/land_loss', 
        'train/glass_loss', 'train/smile_loss', 'train/hat_loss', 'train/mask_loss', 
        'val/score_loss', 'val/gender_loss', 'val/age_loss', 'val/land_loss', 
        'val/glass_loss', 'val/smile_loss', 'val/hat_loss', 'val/mask_loss',
        'acc/score', 'acc/gender', 'acc/age', 'acc/land', 
        'acc/glass', 'acc/smile', 'acc/hat', 'acc/mask', 'lr/lr0', 'lr/lr1', 'lr/lr2']

        for x, tag in zip(list(mloss) + list(eloss) + list(eacc) + lr, tags):
            tb_writer.add_scalar(tag, x, epoch)

        print('%24s'%'' + '%10s' * 8 % ('a/score', 'a/gender', 'a/age', 'a/land', 'a/glass', 'a/smile', 'a/hat', 'a/mask'))
        print('%24s'%'' + '%10.3g' * 8 % (eacc[0], eacc[1], eacc[2], eacc[3], eacc[4], eacc[5], eacc[6], eacc[7]))
        print('')

        # start save ----------------------------------------------------------------------------------------------------
        cv_utils.save_checkpoint(model_eval, last_model)

        cur_acc = eacc[0] + eacc[1] +(100 - eacc[2]) / 100 + (64 - eacc[3]) / 64 + eacc[4] + eacc[5] + eacc[6] + eacc[7] # age land 应该更合理的归一化
        if cur_acc > best_acc:
            best_acc = cur_acc
            cv_utils.save_checkpoint(model_eval, best_model)

    # end epoch ----------------------------------------------------------------------------------------------------
    tb_writer.close()

    # end training
    print("end train func, all info saved in:", save_root)

if __name__ == "__main__":
    
    config_file = "configs/face_attr.yaml"

    train(config_file=config_file)

    print("end all train !!!")