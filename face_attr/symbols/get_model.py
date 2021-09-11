import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import yaml
from symbols.face_attr_symbol import FaceAttr
import torch.nn as nn
import math

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
            # nn.init.uniform_(m.weight, b=0.01)
            # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-5
            m.momentum = 0.1
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
    return model

def build_model(cfg):
    m = cfg['backbone']
    w = cfg['width_multiple']
    
    print('backbone:%s, width_multiple:%.1f' % (m, w))
    if 'FaceAttr' == m:
        model = FaceAttr(training=True, w=w, use_dw=False)
    
    model = initialize_weights(model)

    return model

if __name__ == "__main__":
    
    config_file = "configs/face_attr.yaml"
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg=cfg)
    print(model)

    model.eval()
    
    from thop import profile
    from thop import clever_format
    import torch

    input = torch.randn((1, 3, 64, 64))

    # # to onnx
    input_names = ['input']
    output_names = ['score', 'gender', 'age', 'land', 'glass', 'smile', 'hat', 'mask']
    torch.onnx.export(model, input, 'runs/test.onnx', input_names=input_names, output_names=output_names,
                      verbose=True, training=torch._C._onnx.TrainingMode.EVAL, opset_version=12)
    
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops, 'params:', params)

    print("end all get model !!!")