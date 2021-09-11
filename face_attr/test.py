import sys

from numpy.lib.npyio import save

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import cv2
import glob
import yaml
import torch
import numpy as np

from utils import face_det 
from utils import cv_utils
from symbols import get_model

def test_face_attr(weight_path, config_file, image_path):
    print('22222')
    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv', 'flv']  # acceptable video suffixes

    FaceDet = face_det.FaceDet(conf_threshold=0.7)
    
    do_video = False
    if image_path.split('.')[-1] in vid_formats:
        do_video = True
    else:
        paths = glob.glob(os.path.join(image_path, '*.jpg'))
        paths.sort()

    # model
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    names = cfg['names']

    model = get_model.build_model(cfg=cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cv_utils.load_checkpoint(model, weight_path)
    model.to(device)
    model.eval()

    alpha = 8
    crop_image_size = 64

    if do_video:
        cap = cv2.VideoCapture(image_path)

        save_path = None
        # save_path = image_path.replace(image_path.split('/')[-1], image_path.split('/')[-1].split('.')[0]+'_save.mp4')
        save_path = image_path.split('/')[-1].split('.')[0]+'_save.mp4'

        if save_path is not None:
            # imgSize = (int(cap.get(3)), int(cap.get(4)))
            imgSize = (int(960), int(540))
            video_wr = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, imgSize)

        while(cap.isOpened()):
            ret, image_src = cap.read()
            if image_src is None:
                cap.release()
                if save_path is not None:
                    video_wr.release()
                break
            
            if save_path is not None:
                image_draw = cv2.resize(image_src, imgSize)
            else:
                image_draw = cv_utils.resize_image(image_src.copy())

            d_h, d_w, _ = image_draw.shape

            s_h, s_w, _ = image_src.shape

            # (x0, y0, x1, y1, s) in src image
            face_boxs = FaceDet(frame=image_src.copy())

            for box in face_boxs:
                w = box[2] - box[0]
                h = box[3] - box[1]

                x1 = int(box[0] / s_w * d_w)
                y1 = int(box[1] / s_h * d_h)
                x2 = int(box[2] / s_w * d_w)
                y2 = int(box[3] / s_h * d_h)
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
                
                with torch.no_grad():
                    resized_im = torch.tensor(resized_im, dtype=torch.float32)
                    resized_im = torch.unsqueeze(resized_im, 0)
                    resized_im = resized_im.permute(0, 3, 1, 2)
                    resized_im = resized_im.to(device)

                    face_attr = model(resized_im)

                score_pred, gender_pred, age_pred, land_pred, glass_pred, smile_pred, hat_pred, mask_pred = face_attr
                # print(score_pred, gender_pred, age_pred, land_pred, glass_pred, smile_pred, hat_pred, mask_pred)
                # cv2.imshow("image_draw", resized_im)
                # cv2.waitKey(0)

                x1 = int(box[0] / s_w * d_w)
                y1 = int(box[1] / s_h * d_h)
                x2 = int(box[2] / s_w * d_w)
                y2 = int(box[3] / s_h * d_h)
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_draw, "face:%.2f" % (box[4]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                attrs = torch.cat(face_attr, dim=1).squeeze()
                skip = 15
                for idx, attr_name in enumerate(names):
                    cv2.putText(image_draw, '%s: %.2f' % (attr_name, attrs[idx].cpu().item()), (x2 + 5, y1 + 5 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if idx >=3 and idx <= 12 and idx % 2 == 0:
                        if 4 == idx or 10 == idx:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (0, 255, 0), -1)
                        elif 8 == idx:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (0, 0, 255), -1)
                        else:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (255, 0, 0), -1)
            cv2.imshow("image_draw", image_draw)
            
            if save_path is not None:
                video_wr.write(image_draw)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                if save_path is not None:
                    video_wr.release()
                break
    else:
        for file_name in paths:
            print(file_name)
            image_src = cv_utils.decode_image(file_name)
            if image_src is None:
                print("get none image->", file_name)
                continue

            image_draw = cv_utils.resize_image(image_src.copy())
            d_h, d_w, _ = image_draw.shape

            s_h, s_w, _ = image_src.shape

            # (x0, y0, x1, y1, s) in src image
            face_boxs = FaceDet(frame=image_src.copy())

            for box in face_boxs:
                w = box[2] - box[0]
                h = box[3] - box[1]

                x1 = int(box[0] / s_w * d_w)
                y1 = int(box[1] / s_h * d_h)
                x2 = int(box[2] / s_w * d_w)
                y2 = int(box[3] / s_h * d_h)
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
                
                with torch.no_grad():
                    resized_im = torch.tensor(resized_im, dtype=torch.float32)
                    resized_im = torch.unsqueeze(resized_im, 0)
                    resized_im = resized_im.permute(0, 3, 1, 2)
                    resized_im = resized_im.to(device)

                    face_attr = model(resized_im)

                score_pred, gender_pred, age_pred, land_pred, glass_pred, smile_pred, hat_pred, mask_pred = face_attr
                print(score_pred, gender_pred, age_pred, land_pred, glass_pred, smile_pred, hat_pred, mask_pred)
                # cv2.imshow("image_draw", resized_im)
                # cv2.waitKey(0)

                x1 = int(box[0] / s_w * d_w)
                y1 = int(box[1] / s_h * d_h)
                x2 = int(box[2] / s_w * d_w)
                y2 = int(box[3] / s_h * d_h)
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_draw, "face:%.2f" % (box[4]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                attrs = torch.cat(face_attr, dim=1).squeeze()
                skip = 15
                for idx, attr_name in enumerate(names):
                    cv2.putText(image_draw, '%s: %.2f' % (attr_name, attrs[idx].cpu().item()), (x2 + 5, y1 + 5 + idx * skip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if idx >=3 and idx <= 12 and idx % 2 == 0:
                        if 4 == idx or 10 == idx:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (0, 255, 0), -1)
                        elif 8 == idx:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (0, 0, 255), -1)
                        else:
                            cv2.circle(image_draw, (x1 + int(attrs[idx - 1] * (x2 - x1)), y1 + int(attrs[idx] * (y2 - y1))), 3, (255, 0, 0), -1)


            print(image_draw.shape)
            cv2.imshow("image_draw", image_draw)
            cv2.waitKey(0)

if __name__ == "__main__":
    
    weight_path = 'runs/face_attr/sample0/last.pt'
    config_file = 'configs/face_attr.yaml'

    image_path = 'test_data/face_attr/'
    # image_path = 'test_data/face_attr/face.flv'
    # image_path = 'test_data/face_attr/Rompy.mp4'

    test_face_attr(weight_path, config_file, image_path)

    print("end all test !!!")