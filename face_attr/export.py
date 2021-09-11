import sys
from typing import ClassVar
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

import torch
import numpy as np
import yaml
from symbols.face_attr_symbol import FaceAttr
import torch.nn as nn
from symbols import get_model
from utils.cv_utils import colorstr, file_size, model_info, load_checkpoint

def do_export(weight_path, model, device, imgs, simplify=True):
    print(f"\n{colorstr('PyTorch:')} starting from {weight_path} ({file_size(weight_path):.1f} MB)")
    
    # get model ------------------------------------------------------------------------------------------------------
    load_checkpoint(model, weight_path)
    model.to(device)
    model.eval()

    img = torch.rand(1, 3, imgs, imgs).to(device) * 255

    for _ in range(2):
        y = model(img)

    # get profile ------------------------------------------------------------------------------------------------------
    prefix = colorstr('get profile:')
    try:
        from thop import profile
        from thop import clever_format

        flops, params = profile(model, inputs=(img,))
        flops, params = clever_format([flops, params], "%.3f")
        print('flops:', flops, 'params:', params)

        model_info(model, verbose=False, img_size=imgs)
        
    except Exception as e:
        print(f'{prefix} get profile failure: {e}')

    # ONNX export ------------------------------------------------------------------------------------------------------
    prefix = colorstr('ONNX:')
    try:
        import onnx
        import onnxruntime
        print(f'{prefix} starting export with onnx {onnx.__version__}...')

        onnx_path = weight_path.replace('.pt', '.onnx')  # filename
        input_names = ['input']
        output_names = ['score', 'gender', 'age', 'land', 'glass', 'smile', 'hat', 'mask']
        torch.onnx.export(model, img, onnx_path, verbose=False, training=torch._C._onnx.TrainingMode.EVAL, opset_version=12, 
                                input_names=input_names, output_names=output_names)
        
        # Checks
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                # check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # CAFFE export ------------------------------------------------------------------------------------------------------
    prefix = colorstr('Caffe:')
    try:
        print(f'{prefix} starting export to caffe...')

        import utils.onnx2caffe.convertCaffe as convertCaffe

        onnx_path = weight_path.replace('.pt', '.onnx')  # filename
        
        if False:
            # onnx_path = 'weights/usp-on-des.onnx'
            onnx_path = 'weights/FaceRec/glint360k_cosface_r50_fp16_0.1_nopre.onnx'
            # print(onnx_path)
            onnx_model = onnx.load(onnx_path)
            graph = onnx_model.graph

            # print(onnx_model.graph.input)
            for input_node in onnx_model.graph.input:
                if 'input.1' == input_node.name:
                    print("data in", input_node.name)
                    input_node.name = 'data'

            # 插入sub
            sub_const_node = onnx.helper.make_tensor(name='const_sub',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[1],
                    # vals=np.array(-127.5).astype(np.float32).flatten().astype(float))
                    vals=[-127.5])
            graph.initializer.append(sub_const_node)

            sub_node = onnx.helper.make_node(
                    'Add',
                    name='pre_sub',
                    inputs=['data', 'const_sub'],
                    outputs=['pre_sub'],
                )
            graph.node.insert(0, sub_node)

            # 插入mul
            mul_const_node = onnx.helper.make_tensor(name='const_mul',
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[1],
                    vals=[1.0 / 127.5])
            
            graph.initializer.append(mul_const_node)

            sub_node = onnx.helper.make_node(
                    'Mul',
                    name='pre_mul',
                    inputs=['pre_sub', 'const_mul'],
                    outputs=['pre_mul'],
                )
            graph.node.insert(1, sub_node)

            # 第一层卷积的输入修改
            for id, node in enumerate(graph.node):
                # print(id, node.name, node.op_type, node.input, node.output)
                for i, input_node in enumerate(node.input):
                    if 'input.1' == input_node:
                        # node.input[i] = 'data'
                        node.input[i] = 'pre_mul'
            #     if id > 2:
            #         break
            
            # for id, node in enumerate(graph.node):
            #     print(id, node.name, node.op_type, node.input, node.output)
            #     if id > 2:
            #         break

            graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
            info_model = onnx.helper.make_model(graph)
            onnx_model = onnx.shape_inference.infer_shapes(info_model)
                
            onnx.checker.check_model(onnx_model)
            onnx.save(onnx_model, onnx_path.replace('nopre', 'fix'))
            # onnx_path = onnx_path.replace('nopre', 'fix')
            # print(onnx_path)
            # exit(0)

        if False:
            # onnx_path = 'weights/usp-on-des.onnx'
            onnx_path = 'weights/2d106det_nopre.onnx'
            # print(onnx_path)
            onnx_model = onnx.load(onnx_path)
            graph = onnx_model.graph

            graph.node.remove(graph.node[0])
            graph.node.remove(graph.node[0])

            # # print(onnx_model.graph.input)
            # for input_node in onnx_model.graph.input:
            #     if 'input.1' == input_node.name:
            #         print("data in", input_node.name)
            #         input_node.name = 'data'

            # # 插入sub
            # sub_const_node = onnx.helper.make_tensor(name='const_sub',
            #         data_type=onnx.TensorProto.FLOAT,
            #         dims=[1],
            #         # vals=np.array(-127.5).astype(np.float32).flatten().astype(float))
            #         vals=[-127.5])
            # graph.initializer.append(sub_const_node)

            # sub_node = onnx.helper.make_node(
            #         'Add',
            #         name='pre_sub',
            #         inputs=['data', 'const_sub'],
            #         outputs=['pre_sub'],
            #     )
            # graph.node.insert(0, sub_node)

            # # 插入mul
            # mul_const_node = onnx.helper.make_tensor(name='const_mul',
            #         data_type=onnx.TensorProto.FLOAT,
            #         dims=[1],
            #         vals=[1.0 / 127.5])
            
            # graph.initializer.append(mul_const_node)

            # sub_node = onnx.helper.make_node(
            #         'Mul',
            #         name='pre_mul',
            #         inputs=['pre_sub', 'const_mul'],
            #         outputs=['pre_mul'],
            #     )
            # graph.node.insert(1, sub_node)

            # 第一层卷积的输入修改
            # graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
            for id, node in enumerate(graph.node):
                # print(id, node.name, node.op_type, node.input, node.output)
                for i, input_node in enumerate(node.input):
                    if '_mulscalar0' == input_node:
                        node.input[i] = 'data'
            
            for id, node in enumerate(graph.node):
                # print(id, node.name, node.op_type, node.input, node.output)
                if 'BatchNormalization' == node.op_type:
                    # print(id, node.name, node.op_type, node.input, node.output)
                    # print(node.attribute, type(node.attribute))
                    # # del node.attribute['spatial']
                    # print(node.attribute[2], type(node.attribute[1]))
                    # print(node.attribute, type(node.attribute))
                    for i, attr in enumerate(node.attribute):
                        if attr.name == 'spatial':
                            print(id, node.name, node.op_type, node.input, node.output)
                            del node.attribute[i]
                    # exit(0)

            graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
            info_model = onnx.helper.make_model(graph)
            onnx_model = onnx.shape_inference.infer_shapes(info_model)
            print("44444")
            onnx.checker.check_model(onnx_model)
            onnx.save(onnx_model, onnx_path.replace('nopre', 'fix'))
            onnx_path = onnx_path.replace('nopre', 'fix')
            # print(onnx_path)
            # exit(0)

        # onnx_path = 'weights/face_attr/test.onnx'

        prototxt_path = onnx_path.replace('.onnx', '.prototxt')
        caffemodel_path = onnx_path.replace('.onnx', '.caffemodel')

        convertCaffe.do_convert(onnx_path, prototxt_path, caffemodel_path)

        print(f'{prefix} export success, saved as {caffemodel_path} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')
    
    # Model Compare ------------------------------------------------------------------------------------------------------
    prefix = colorstr('Model Compare:')
    try:
        print(f'{prefix} starting Model Compare...')
        import utils.onnx2caffe.modelComparator as modelComparator
        modelComparator.compareOnnxAndCaffe(onnx_path, prototxt_path, caffemodel_path)

        torch_out = model(img)
        
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # compare ONNX Runtime and PyTorch results
        for t_out, o_out in zip(torch_out, ort_outs):
            np.testing.assert_allclose(to_numpy(t_out), o_out, rtol=1e-05, atol=1e-05)
            # check if result are same by cosine distance
            print(t_out)
            dot_result = np.dot(to_numpy(t_out).flatten(), o_out.flatten())
            left_norm = np.sqrt(np.square(to_numpy(t_out)).sum())
            right_norm = np.sqrt(np.square(o_out).sum())
            cos_sim = dot_result / (left_norm * right_norm)
            print("cos sim between pytorch and onnx models: -> %f" % (cos_sim))

        print(f'{prefix} compar pytorch onnx caffe , and the results looks good!')

    except Exception as e:
        print(f'{prefix} model compare failure: {e}')
    
    print("end model export!!")

if __name__ == "__main__":
    
    config_file = "configs/face_attr.yaml"
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # weights
    train_model = config_file.split('/')[-1].split('.')[0]
    save_root = "runs/" + train_model + '/'
    weight_path = save_root + 'sample0/' +'best.pt'

    # model
    model = get_model.build_model(cfg=cfg)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # image size
    imgs = cfg['image_size']

    do_export(weight_path, model, device, imgs)
    exit(0)

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