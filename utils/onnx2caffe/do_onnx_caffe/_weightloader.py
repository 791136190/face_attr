from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# from caffe import params as P
import numpy as np
from ._graph import Node, Graph

USE_DECONV_AS_UPSAMPLE = False

def _convert_conv(net, node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name,))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    np.copyto(net.params[node_name][0].data,W,casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')

def _convert_relu(net, node, graph, err):
    pass

def _convert_prelu(net, node, graph, err):
 
    weight = node.input_tensors[node.inputs[1]]
   
    # copy weight to caffe model
    shape = weight.shape
    # 因为 onnx 中 prelu 是三维数组，如（64， 1， 1），而 caffe 中 prelu 是一维，如 (64, )
    # 故要 reshape ，不然会报错
    weight = weight.reshape((shape[0])) 
    np.copyto(net.params[node.name][0].data, weight, casting='same_kind') # 复制参数到 caffe 模型

def _convert_sigmoid(net, node, graph, err):
    pass

def _convert_BatchNorm(net, node, graph, err):
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name
    np.copyto(net.params[node_name + '_bn'][0].data, mean, casting='same_kind')
    np.copyto(net.params[node_name + '_bn'][1].data, var, casting='same_kind')
    net.params[node_name + '_bn'][2].data[...] = 1.0
    np.copyto(net.params[node_name][0].data, scale, casting='same_kind')
    np.copyto(net.params[node_name][1].data, bias, casting='same_kind')
    # net.params[node_name+'_bn'][1].data = var
    # net.params[node_name][0].data = scale
    # net.params[node_name][1].data = bias

def _convert_Add(net, node, graph, err):
    # print("weight load add")
    # weight = node.input_tensors[node.inputs[1]]
   
    # # copy weight to caffe model
    # shape = weight.shape
    # # 因为 onnx 中 prelu 是三维数组，如（64， 1， 1），而 caffe 中 prelu 是一维，如 (64, )
    # # 故要 reshape ，不然会报错
    # weight = weight.reshape((shape[0])) 
    # np.copyto(net.params[node.name][0].data, weight, casting='same_kind') # 复制参数到 caffe 模型

    pass

def _convert_Mul(net, node, graph, err):
    pass

def _convert_MatMul(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:  # 判断是否有参数，免得下面报错
        W = node.input_tensors[weight_name]  # 如果有，获得参数数组
    else:
        err.missing_initializer(node,
                                "MatMul weight tensor: {} not found in the graph initializer".format(weight_name, ))

    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")
    if b is not None:
        if W.shape[1] != b.shape[0]:
            return err.unsupported_op_configuration(node, "MatMul is supported only for inner_product layer")
    # net.params[node_name][0].data[...] = W  # 卖个关子，这样写并不完全对，下面有讲
    net.params[node_name][0].data[...] = W.transpose() # 这句才对 先做转置，caffe里再转置就回到原点

def _convert_Reshape(net, node, graph, err):
    pass

def _convert_Flatten(net, node, graph, err):
    pass

def _convert_pool(net, node, graph, err):
    pass

def _convert_dropout(net, node, graph, err):
    pass

def _convert_gemm(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    # if node.attrs["broadcast"] != 1 or node.attrs["transB"] != 1:
    if node.attrs["transB"] != 1:
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    net.params[node_name][0].data[...] = W
    net.params[node_name][1].data[...] = b

def _convert_upsample(net, node, graph, err):
    mode = node.attrs["mode"]
    node_name = node.name
    if mode == "nearest":
        caffe_params = net.params[node_name][0].data
        weights = np.ones(caffe_params.shape).astype("float32")
        np.copyto(net.params[node_name][0].data, weights, casting='same_kind')
        # net.params[node_name][0].data[]

def _convert_resize(net, node, graph, err):
    if USE_DECONV_AS_UPSAMPLE:
        print('add resize deconv param')
        node_name = node.name
        caffe_params = net.params[node_name][0].data
        weights = np.ones(caffe_params.shape).astype("float32")
        np.copyto(net.params[node_name][0].data, weights, casting='same_kind')

def _convert_transpose(net, node, graph, err):
    pass
def _convert_concat(net, node, graph, err):
    pass
def _convert_softmax(net, node, graph, err):
    pass

def _convert_conv_transpose(net, node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name,))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    np.copyto(net.params[node_name][0].data,W,casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')

_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    # "LeakyRelu": _convert_relu,
    "PRelu": _convert_prelu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Sum": _convert_Add,
    "Mul": _convert_Mul,
    "MatMul": _convert_MatMul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "Upsample": _convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
    "Resize": _convert_resize,
    "Transpose": _convert_transpose,
    "Softmax": _convert_softmax,
}


