import math
from .device.CIMA import *

from Model2IR.onnx2ir.converter import ConvertONNX
import numpy as np
from .self_defined_op.fused_op import *
from .self_defined_op.cima_op import *

from irtool.core.layer import make_layer, make_op, DataDef

import copy
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import warnings

def get_max_time_layer(layer_time):
    '''
    input:
        layer_time: dict like {'layer_name': layer_time}
    return:
        A dict containing the single layer with the maximum runtime.
        If multiple layers share the same max time, the first encountered is returned.
    '''

    a1 = sorted(layer_time.items(),key = lambda x:x[1],reverse = True)
    layer_name = a1[0][0]
    max_ = a1[0][1]
    return {layer_name : max_}

def split_node(node_shape,split_num):
    '''
    input:
        node_shape: dict like {'node_name': [w, h]}
        split_num: dict like {'node_name': [para_diff_array, para_same_array, w_splits, h_splits]}
            The list items indicate:
            - parallel copies across different arrays (para_diff_array)
            - parallel copies within the same array (para_same_array)
            - split counts along width/height
    return:
        A dict like {'node_name.new': [w_split, h_split]} where:
        node_name.new = node_name + '.repeat_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_shape.keys():
        h = []
        w = []
        [W,H] = node_shape[node_name]
        [pda,psa,w_split,h_split] = split_num[node_name]
        h_i = h_split
        w_i = w_split
        _h = math.floor(H/h_i)
        _w = math.floor(W/w_i)
        for i in range(h_i):
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        for k in range(pda):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]
    return node_shape_split

def split_node_window_duplicate(node_info,xb_size,split_num):
    '''
    input:
        node_info: dict keyed by node name.
        split_num: dict like {'node_name': [para_num, repeat_num, w_splits, h_splits]}.
    return:
        A dict like {'node_name.new': [w_split, h_split]} where:
        node_name.new = node_name + '.parallel_index' + '.h_index' + '.w_index'
    '''
    node_shape_split = {}
    for node_name in node_info.keys():

        h = []
        w = []
        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            [para_num,repeat_num,w_split,h_split] = split_num[node_name]
            kz = node_info[node_name]['kernel_size']
            stride = node_info[node_name]['stride']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            # cc = node_info[node_name]['copy_constraint']
            W = out_channel * repeat_num
            H = ( kz  + (repeat_num - 1) * stride ) * in_channel * kz
            h_i =  math.ceil(H /  xb_size[1])
            w_i =  math.ceil(W /  xb_size[0])

            _h = math.floor(H/h_i)
            _w = math.floor(W/w_i)
            split_num[node_name] = [para_num,repeat_num,w_i,h_i]

        elif node_info[node_name]['op_type'] in ['matmul','fc','linear', 'fused_fc']:
            [para_num, w_split, h_split] = split_num[node_name]
            _h = h_split
            _w = w_split

        for i in range(h_i):
            if H - _h > 0:
                h.append(_h)
                H = H - _h
            else:
                h.append(H)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)

        for k in range(para_num):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)+'_wd'] = [w[j],h[i]]

    return node_shape_split,split_num

def split_node_HWC(node_weight,node_info,para_num,XB_size,dmac_size=None,dmac_layer= None,device='cima'):
    '''
    input:
        node_weight: dict like {'node_name': [w, h]}
        node_info: dict like {'node_name': node_info}
        para_num: dict like {'node_name': para_num}
        XB_size: list like [w, h]
    return:
        A dict like {'node_name.new': [w_split, h_split]} where:
        node_name.new = node_name + '.repeat_index' + '.h_index' + '.w_index'
    '''

    node_shape_split = {}
    node_split_num = {}

    for node_name in node_weight.keys():

        h = []
        w = []
        [W, H] = node_weight[node_name]
        array_size = XB_size

        if dmac_layer!= None and node_name in dmac_layer:
            assert dmac_size != None
            array_size = dmac_size


        if node_info[node_name]['op_type'] in ['conv2d', 'fused_conv2d', 'conv_transpose2d']:
            kernel_size = node_info[node_name]['kernel_size']
            in_channel = node_info[node_name]['in_channel']
            out_channel = node_info[node_name]['out_channel']
            assert (H % (kernel_size**(2) * in_channel) == 0)
            row_repeat_avg = H / (kernel_size**(2) * in_channel)
            if H <= array_size[1]:
                h.append(H)
            else:
                h_temp = H
                # t = 1
                split_ic = []
                #         t += 1
                #     else:
                #         t *= 2
                #     max_split_channel_num, split_ic = get_max_channel_split_num(in_channel,t)
                #         warnings.warn(f"Non-uniform input-channel split for layer '{node_name}'. split_ic={split_ic}.")
                #     h_temp = max_split_channel_num * kernel_size * kernel_size * row_repeat_avg
                row_split_num = math.ceil(h_temp / array_size[1])
                while in_channel % row_split_num != 0:
                    row_split_num += 1
                max_split_channel_num, split_ic = get_max_channel_split_num(in_channel,row_split_num)
                if np.array(split_ic).mean() != max_split_channel_num:
                    warnings.warn(
                        f"Non-uniform input-channel split detected for layer '{node_name}'. "
                        f"Split channels: {split_ic}."
                    )

                assert (split_ic != [])
                for ic_ in split_ic:
                    h.append(ic_ * kernel_size * kernel_size * row_repeat_avg )
        else:

            h_split = math.ceil(H /  array_size[1])
            h_i = h_split
            _h = math.floor(H/h_i)
            for i in range(h_i):
                if H - _h > 0:
                    h.append(_h)
                    H = H - _h
                else:
                    h.append(H)

        w_split = math.ceil(W /  array_size[0])
        w_i = w_split
        _w = math.floor(W/w_i)
        for j in range(w_i):
            if W - _w > 0:
                w.append(_w)
                W = W - _w
            else:
                w.append(W)
        # Parallel repeat across different arrays.
        repeat = 1
        if para_num != None:
            repeat = para_num[node_name][0]
        diff_array_repeat = repeat
        same_array_repeat = 1
        node_split_num[node_name] = [diff_array_repeat, same_array_repeat, len(w), len(h)]
        for k in range(repeat):
            for i in range(len(h)):
                for j in range(len(w)):
                    node_shape_split[node_name+'.'+str(k)+'.'+str(i)+'.'+str(j)] = [w[j],h[i]]

    return node_shape_split, node_split_num

def get_max_channel_split_num(ic,split_num):
    '''
    input:
        ic: an integer
        split_num: number of partitions; each partition should be as even as possible
    return:
        max_num: the maximum partition size after splitting
    '''
    t = math.ceil(ic / split_num)
    w = []
    rest = ic
    for i in range(split_num):
        temp = rest - t
        if temp > 0:
            w.append(t)
            rest = temp
        else:
            w.append(rest)

    return np.array(w).max(), w

def get_layer_ref(inputs, layer, ref):
    '''
    input:
        inputs: operator inputs
        layer: layer metadata dict
    '''

    # MAX_count = 10
    for i in inputs:
        ref_name = i.ref
        # count = 0
        if ':' in ref_name:
            ref_name = ref_name.split(':')[0]
        if 'graph_input' in ref_name:
            ref.append(ref_name)
        elif layer[ref_name].type == 'reuse':
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d', 'linear','matmul', 'fc', 'fused_fc']:
            ref.append(ref_name)
        elif layer[ref_name].op.op_id in ['constant', 'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul']:
            ref.append(ref_name)
        else:
            get_layer_ref(layer[ref_name].inputs, layer, ref)

        # input()

def get_conv_shape(op_info):
    '''
    input:
        op_info: operator metadata (op object). Assumes square kernels by default.
    '''
    kernel_size = op_info.kernel
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    # bias=op_info.bias
    # Bias is assumed to be handled off-chip by default.
    bias = False
    if bias:
        unroll_shape_h = kernel_size * kernel_size * in_channel + 1
    else:
        unroll_shape_h = kernel_size * kernel_size * in_channel
    unroll_shape_w = out_channel

    return [unroll_shape_w,unroll_shape_h]

def get_linear_shape(op_info):
    '''
    input:
        op_info: operator metadata (op object)
    '''
    in_channel = op_info.in_channel
    out_channel = op_info.out_channel
    # bias=op_info.bias
    # Bias is assumed to be handled off-chip by default.
    bias = False
    if bias:
        unroll_shape_h = in_channel + 1
    else:
        unroll_shape_h = in_channel
    unroll_shape_w = out_channel

    return [unroll_shape_w,unroll_shape_h]

def get_conv_info(layer):
    '''
    input:
        layer: layer object
    return:
        dict with convolution attributes and derived metadata.
    '''

    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8

    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8

    kz = layer.op.kernel
    stride = layer.op.stride
    padding  = layer.op.padding

    out_height = layer.outputs[0].height
    out_width = layer.outputs[0].width

    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel

    copy_const = out_height
    calc_num = out_height * out_width
    op_type = layer.op.op_id


    in_data_len = (layer.inputs[0].height + 2 * padding) * (layer.inputs[0].width + 2 * padding) * layer.inputs[0].channel
    out_data_len = out_height * out_width * out_channel

    input_shape = [layer.inputs[0].height, layer.inputs[0].width]

    return dict(op_type=op_type,in_channel=in_channel,out_channel=out_channel,kernel_size=kz,
                stride=stride,calc_num=calc_num,in_precision=intype,
                out_precision=outtype,copy_constraint=copy_const, in_data_len=in_data_len,
                out_data_len = out_data_len, input_shape=input_shape)

def get_linear_info(layer):
    '''
    input:
        layer: layer object
    return:
        dict with linear/matmul attributes and derived metadata.
    '''
    if layer.inputs[0].dtype != None:
        intype = layer.inputs[0].dtype
    else:
        intype = 8

    if layer.outputs[0].dtype != None:
        outtype = layer.outputs[0].dtype
    else:
        outtype = 8
    in_channel = layer.op.in_channel
    out_channel = layer.op.out_channel

    calc_num = 1
    kz = 1
    stride = 1
    op_type = layer.op.op_id
    copy_const = 1

    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, kernel_size=kz,
                stride=stride, calc_num=calc_num, in_precision=intype, out_precision=outtype,
                copy_constraint=copy_const, in_data_len = in_channel, out_data_len = out_channel)

def get_split_concat_info(layer):
    '''
    input:
        layer: layer object
    return:
        dict with split/concat attributes and derived metadata.
    '''
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])

    out_channel = []
    if layer.outputs != None:
        for out_ in layer.outputs:
            out_channel.append(out_.channel)

    op_type = layer.op.op_id
    axis = layer.op.axis

    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, axis = axis, input_shape=input_shape)

def get_add_info(layer):
    '''
    input:
        layer: layer object
    return:
        dict with add attributes and derived metadata.
    '''
    in_channel = []
    input_shape = []
    for in_ in layer.inputs:
        in_channel.append(in_.channel)
        input_shape.append([in_.height, in_.width])

    out_channel = []
    for out_ in layer.outputs:
        out_channel.append(out_.channel)

    op_type = layer.op.op_id

    return dict(op_type=op_type, in_channel=in_channel, out_channel=out_channel, input_shape=input_shape)

def list_reverse(list_):
    '''
    input:
        list, e.g. [1, 2, 3, 4, 5]
    output:
        reversed list, e.g. [5, 4, 3, 2, 1]
    '''
    len_ = len(list_)
    reverse_list = []
    for i in range(len_-1,-1,-1):
        reverse_list.append(list_[i])
    return reverse_list

def make_mapped_ir(ir,split_info,place_info,copy_info=None,cpu_layer=None,
                   calc_info=None, device='cima', runtime = 'simulation', **kwargs):
    '''
    add mapping info into IR
    input:
        ir: ir object
        split_info: dict like {'node_name': [r, w, h], ...}
        place_info: dict like {node_name: [MappingInfo objects], ...}
    return:
        ir object with mapping info
    '''
    for name, layer in ir.iter_layers():
        if layer.type == 'op' :
            if cpu_layer != None and name in cpu_layer:
                #             Warning(f"Layer '{name}' has no runtime params; using defaults.")
                #             pass
                #         else:
                #             pass
                #     else:
                #         pass
                continue
            if layer.op.op_id in ['conv2d', 'fused_conv2d', 'conv_transpose2d'] or layer.op.op_id in ['linear', 'matmul', 'fc', 'fused_fc']:
                if 'cima' in device:
                    split_num = split_info[name]

                    if copy_info != None:
                        col_repeat_num = copy_info[name][1]
                        row_repeat_num = copy_info[name][0]
                    else:
                        col_repeat_num = 1
                        row_repeat_num = 1

                    in_line_buffer_addr = kwargs['in_line_buffer_addr'][name]
                    # output_tile_buffer_addr = kwargs['output_tile_buffer_addr'][name]
                    # in_buf_type = kwargs['in_buf_type'][name]
                    # out_buf_type = kwargs['out_buf_type'][name]
                    credit_len = kwargs['credit_len'][name]

                    layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=split_num[2],row_split_num=split_num[3],
                                                        col_repeat_num=col_repeat_num,row_repeat_num=row_repeat_num,
                                                        para_diff_array=split_num[0],in_line_buffer_addr = in_line_buffer_addr,
                                                        credit_len = credit_len,
                                                        mappings=place_info[name])
                    if calc_info == None:
                        layer.CIMA_calc_info = CIMACalcInfo().clone()
                    elif isinstance(calc_info,dict):
                        if name not in calc_info.keys():
                            Warning(f"Layer '{name}' has no calc_info configured; using defaults.")
                            layer.CIMA_calc_info = CIMACalcInfo().clone()
                        else:
                            layer.CIMA_calc_info = calc_info[name]
                    else:
                        layer.CIMA_calc_info = calc_info.clone()

                    if kwargs['dmac_layer'] != None and name in kwargs['dmac_layer']:
                        layer.CIMA_calc_info.data_type = '8bit'
                        continue
                    elif kwargs['layer_data_type_dict'] != None and name in kwargs['layer_data_type_dict'].keys():
                        layer.CIMA_calc_info.data_type = kwargs['layer_data_type_dict'][name]

                else:
                    raise ValueError(f"Unsupported device spec: {device!r}.")

            elif 'cima' in device and layer.op.op_id in ['add', 'maxpool2d', 'avgpool2d', 'concat', 'split', 'fused_add', 'fused_concat', 'identity',
                                                         'global_avg_pool2d', 'silu', 'resize', 'mul_add', 'pad', 'relu', 'type_conversion']:

                in_line_buffer_addr = kwargs['in_line_buffer_addr'][name]
                # output_tile_buffer_addr = kwargs['output_tile_buffer_addr'][name]
                # in_buf_type = kwargs['in_buf_type'][name]
                # out_buf_type = kwargs['out_buf_type'][name]
                credit_len = kwargs['credit_len'][name]

                layer.CIMA_mapping_info = CIMAMappingInfo(col_split_num=None,row_split_num=None,
                                                        col_repeat_num=None,row_repeat_num=None,
                                                        para_diff_array=None,in_line_buffer_addr = in_line_buffer_addr,
                                                        credit_len = credit_len,
                                                        mappings=place_info[name])

                if calc_info == None:
                    layer.CIMA_calc_info = CIMACalcInfo().clone()
                elif isinstance(calc_info,dict):
                    if name not in calc_info.keys():
                        Warning(f"Layer '{name}' has no calc_info configured; using defaults.")
                        layer.CIMA_calc_info = CIMACalcInfo().clone()
                    else:
                        layer.CIMA_calc_info = calc_info[name]
                else:
                    layer.CIMA_calc_info = calc_info.clone()

                # Type-conversion layers are handled separately.
                if layer.op.op_id == 'type_conversion':
                    layer.CIMA_calc_info.data_type = layer.op.out_dtype
                    continue

                # Derived layers of DMAC layers must keep the same numeric precision.
                if kwargs['dmac_layer'] != None :
                    IsDmacAppendLayer = False
                    for l in kwargs['dmac_layer']:
                        if l in name:
                            layer.CIMA_calc_info.data_type = ir.layers[name].CIMA_calc_info.data_type
                            IsDmacAppendLayer = True
                            break
                    if IsDmacAppendLayer:
                        continue

                # Use layer_data_type_dict to pin per-layer bit width if provided.
                if kwargs['layer_data_type_dict'] != None and name in kwargs['layer_data_type_dict'].keys():
                    layer.CIMA_calc_info.data_type = kwargs['layer_data_type_dict'][name]
                    continue


                # Otherwise, inherit the data type from the reference (previous) layer.
                pre_layer_dict = get_pre_layer(ir.layers)
                pre_layer = pre_layer_dict[name][0]
                if 'graph_input' in pre_layer:
                    continue
                pre_layer_data_type = ir.layers[pre_layer].CIMA_calc_info.data_type
                layer.CIMA_calc_info.data_type = pre_layer_data_type

    return ir

def make_device_ir(ir,device=None):
    '''
    add device info into IR
    input:
        ir: ir object
        device: device info dict(s) used to populate IR devices.
    return:
        ir object with device info

    '''
    if ir.devices != None:
        raise ValueError(f"IR already has devices: {list(ir.devices.keys())}.")

    if device != None:
        if isinstance(device,list):
            for dev_ in device:
                dev_copy = copy.deepcopy(dev_)
                if 'num' in dev_.keys():
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    dev_copy.pop('num')
                    ir.add_device(dev_['name'], dev_['kind'], number=dev_['num'], **dev_copy)
                else:
                    dev_copy.pop('name')
                    dev_copy.pop('kind')
                    ir.add_device(dev_['name'], dev_['kind'], **dev_copy)
        elif isinstance(device,dict):
            dev_copy = copy.deepcopy(device)
            dev_copy.pop('name')
            dev_copy.pop('kind')
            dev_copy.pop('num')
            ir.add_device(device['name'], device['kind'], number=device['num'], **dev_copy)
        else:
            raise TypeError(f'device type {type(device)} error!!!')
        return ir
    else:
        raise ValueError("Missing device info.")

def make_onnx_ir(onnx_file,return_weight=False, fix_layer_name=True):
    '''
    convert onnx into ir object
    input:
        onnx_file: onnx model name
        return_weight: boolean ;
    return:
        case1 : ir object and weight value when return_weight is True
        case2 : ir object
    '''
    t = ConvertONNX(onnx_file, fix_layer_name=fix_layer_name)
    if return_weight:
        return t.ir,t.model_parser.weight_numpy
    else:
        return t.ir

def make_node_id(split_nodes):
    node_id = {}
    for i in range(len(split_nodes)):
        if len(split_nodes[i]) != 1:
            raise ValueError(f"Multiple layers in a single XB are not supported: {split_nodes!r}")
        else:
            node_name = list(split_nodes[i][0].keys())[0]
            node_id[node_name] = i
    return node_id

def fuse_op(ir, relu_fuse =False, pool_fuse=False, split_fuse = False, silu_fuse = False, conv_bn_fuse = True, conv_mul_add_fuse = False):

    # Operator fusion.
    # Mapping from fused (removed) layer name -> fusion root layer name.
    fused_op_all = {}

    next_layer_dict = get_next_layer(ir.layers)
    layers_info = ir.layers

    for name, layer in layers_info.items():

        # Determine whether fusion is applicable.
        can_fuse_relu = False
        can_fuse_pool = False
        can_fuse_split = False
        can_fuse_silu = False
        can_fuse_mul_add = False
        can_fuse_conv_bn = False

        #
        if layer.type == 'op' and layer.op.op_id in ['conv2d']:
            next_layers = next_layer_dict[name]

            current_op_info = layer.op
            # First-level op-type check.
            if len(next_layers) == 1 and layers_info[next_layers[0]].type == 'op' and layers_info[next_layers[0]].op.op_id in ['mul']:
                nl = next_layers[0]
                nl_info = layers_info[nl]
                # First-level shape check.
                nl_input_2_info = layers_info[nl_info.inputs[1].ref]
                if len(nl_info.inputs) == 2 and nl_input_2_info.type == 'op' and nl_input_2_info.op.op_id in ['constant'] and \
                    nl_info.inputs[0].channel == nl_info.inputs[1].channel:

                    # Second-level op-type check.
                    next_layers_2 = next_layer_dict[nl]
                    if len(next_layers_2) == 1 and layers_info[next_layers_2[0]].type == 'op' and layers_info[next_layers_2[0]].op.op_id in ['add']:
                        nl_nl = next_layers_2[0]
                        nl_nl_info = layers_info[nl_nl]
                        # Second-level shape check.
                        nl_nl_input_2_info = layers_info[nl_nl_info.inputs[1].ref]
                        if len(nl_nl_info.inputs) == 2 and nl_nl_input_2_info.type == 'op' and nl_nl_input_2_info.op.op_id in ['constant'] and \
                            nl_nl_info.inputs[0].channel == nl_nl_info.inputs[1].channel:
                            can_fuse_mul_add = True

            if can_fuse_mul_add:
                #
                kernel = current_op_info.kernel
                in_channel = current_op_info.in_channel
                out_channel = current_op_info.out_channel
                stride = current_op_info.stride
                padding = current_op_info.padding
                bias = current_op_info.bias
                fused_op_obj = fused_conv2d(kernel=kernel, in_channel = in_channel,
                                        out_channel = out_channel, stride= stride,
                                        padding= padding, bias= bias).clone()

                # next_layer_op_info = layers_info[nl]
                fused_op_obj.mul = layers_info[nl]
                fused_op_obj.add = layers_info[nl_nl]
                #
                fused_layer_inputs = layer.inputs
                fused_layer_weights = layer.weights
                # Use the last fused layer outputs as the fused-layer outputs.
                fused_layer_outputs = layers_info[nl].outputs

                fused_layer = make_layer(op= fused_op_obj, inputs = fused_layer_inputs,
                                        weights = fused_layer_weights,
                                        outputs = fused_layer_outputs)
                ir.layers[name] = fused_layer
                #
                fused_op_all[nl] = name
                fused_op_all[nl_nl] = name

        if layer.type == 'op' and layer.op.op_id in ['conv2d', 'matmul', 'linear', 'fc', 'add', 'concat', 'fused_add', 'fused_concat']:
            next_layers = next_layer_dict[name]

            # Only support the case where there is exactly one next layer.
            if len(next_layers) == 1:
                nl = next_layers[0]
                if layer.op.op_id in ['conv2d', 'conv_transpose2d', 'matmul']:
                    if layers_info[nl].type == 'op':
                        if relu_fuse and layers_info[nl].op.op_id in ['relu']:
                            can_fuse_relu = True
                        if conv_bn_fuse and layers_info[nl].op.op_id in ['batch_norm2d']:
                            can_fuse_conv_bn = True
                else:
                    if layers_info[nl].type == 'op':
                        if relu_fuse and layers_info[nl].op.op_id in ['relu']:
                            can_fuse_relu = True
                        if pool_fuse and layers_info[nl].op.op_id in ['max_pool2d', 'maxpool2d','global_avg_pool2d']:
                            can_fuse_pool = True
                        if split_fuse and layers_info[nl].op.op_id in ['split'] :
                            if layer.op.op_id in ['fused_add', 'fused_concat'] and layer.op.split != None:
                                continue
                            can_fuse_split = True
                        if silu_fuse and layers_info[nl].op.op_id in ['silu']:
                            can_fuse_silu = True

            # Fusion is applicable.
            if can_fuse_pool or can_fuse_relu or can_fuse_split or can_fuse_silu or can_fuse_conv_bn:

                current_op_info = layer.op
                if current_op_info.op_id in ['conv2d', 'conv_transpose2d']:
                    kernel = current_op_info.kernel
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    stride = current_op_info.stride
                    padding = current_op_info.padding
                    bias = current_op_info.bias
                    fused_op_obj = fused_conv2d(kernel=kernel, in_channel = in_channel,
                                            out_channel = out_channel, stride= stride,
                                            padding= padding, bias= bias).clone()

                elif current_op_info.op_id in ['matmul']:
                    in_channel = current_op_info.in_channel
                    out_channel = current_op_info.out_channel
                    bias = current_op_info.bias
                    fused_op_obj = fused_fc(in_channel = in_channel, out_channel = out_channel,
                                        bias=bias).clone()

                if current_op_info.op_id in ['add']:
                    fused_op_obj = fused_add().clone()
                elif current_op_info.op_id in ['fused_add']:
                    fused_op_obj = layer.op.clone()

                elif current_op_info.op_id in ['concat']:
                    attr_ = dict(axis = current_op_info.axis)
                    fused_op_obj = fused_concat(**attr_).clone()
                elif current_op_info.op_id in ['fused_concat']:
                    fused_op_obj = layer.op.clone()

                for nl in next_layers:

                    if nl in fused_op_all.keys():
                        raise ValueError(f"Layer '{nl}' has already been fused into '{fused_op_all[nl]}'.")

                    # Attach the next-layer op into the fused op.
                    next_layer_op_info = layers_info[nl].op
                    if current_op_info.op_id in ['add', 'concat', 'fused_concat', 'fused_add'] and len(next_layers) == 1 and next_layer_op_info.op_id in ['split']:
                        # Fuse add/concat with split (CIMA architecture).
                        fused_op_obj.split = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['relu']:
                        fused_op_obj.relu = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['max_pool2d', 'maxpool2d','global_avg_pool2d']:
                        fused_op_obj.pool = next_layer_op_info
                        fused_op_all[nl] = name
                    # elif current_op_info.op_id in ['conv2d', 'matmul', 'linear', 'fc', 'add', 'concat'] and next_layer_op_info.op_id in ['silu']:
                    elif current_op_info.op_id in ['add', 'concat'] and next_layer_op_info.op_id in ['silu']:
                        fused_op_obj.silu = next_layer_op_info
                        fused_op_all[nl] = name
                    elif next_layer_op_info.op_id in ['batch_norm2d']:
                        fused_op_obj.with_bn = True
                        fused_op_all[nl] = name

                fused_layer_inputs = layer.inputs
                fused_layer_weights = layer.weights
                # Use the last fused layer outputs as the fused-layer outputs.
                fused_layer_outputs = layers_info[nl].outputs

                fused_layer = make_layer(op= fused_op_obj, inputs = fused_layer_inputs,
                                        weights = fused_layer_weights,
                                        outputs = fused_layer_outputs)
                ir.layers[name] = fused_layer
                # ir.add_layer(name=name, layer=fused_layer)

    # Remove all fused layers and rewrite downstream refs.
    for fused_op_name, replaced_op_name in fused_op_all.items():
        # Rewrite downstream refs.
        next_layers = next_layer_dict[fused_op_name]
        for nl in next_layers:
            for i in ir.layers[nl].inputs:
                if ":" in i.ref:
                    ref_name = i.ref.split(':')
                    if ref_name[0] == fused_op_name:
                        i.ref = f'{replaced_op_name}:{ref_name[1]}'
                elif i.ref == fused_op_name:
                    i.ref = replaced_op_name
        # Delete fused layer.
        ir.layers.pop(fused_op_name)

    # Re-sort layers.
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir, fused_op_all

def insert_mul_add_op(ir, mul_add_op = None):

    # Gather predecessor info.
    pre_layer_dict = get_pre_layer(ir.layers)
    layers_info = ir.layers

    # Record inserted layer name mapping.
    insert_op_name_dict = {}
    # Insert mul_add operator.
    if mul_add_op != None:
        for (name, index_list) in mul_add_op:
            insert_op_name_dict[name] = []

            if name not in pre_layer_dict.keys():
                raise ValueError(f"Layer '{name}' was not found in IR.")
            pre_layers = pre_layer_dict[name]

            for index in index_list:
                assert index <= len(pre_layers) - 1
                # Get predecessor layer.
                inserted_pre_layer_name = pre_layers[index]
                if 'graph_input' in inserted_pre_layer_name:
                    index = inserted_pre_layer_name.split(':')[-1]
                    inserted_pre_layer = layers_info['graph_input'].inputs[int(index)]
                    channel = inserted_pre_layer.channel
                    width = inserted_pre_layer.width
                    height = inserted_pre_layer.height
                    insert_op_obj = MulAddOp().clone()
                    inserted_layer_name = f'graph_input_{index}_mul_add'
                    inserted_layer_inputs = [dict(ref=inserted_pre_layer_name, channel=channel,height=height,width=width)]
                    inserted_layer_outputs = inserted_layer_inputs
                else:
                    inserted_pre_layer = layers_info[inserted_pre_layer_name]
                    channel = inserted_pre_layer.outputs[0].channel
                    width = inserted_pre_layer.outputs[0].width
                    height = inserted_pre_layer.outputs[0].height
                    # Inserted layer metadata.
                    insert_op_obj = MulAddOp().clone()
                    inserted_layer_name = f'{inserted_pre_layer_name}_mul_add'
                    inserted_layer_inputs = [dict(ref=inserted_pre_layer_name, channel=channel,height=height,width=width)]
                    inserted_layer_outputs = inserted_pre_layer.outputs

                inserted_layer = make_layer(op= insert_op_obj, inputs = inserted_layer_inputs,
                                        outputs = inserted_layer_outputs)
                ir.layers[inserted_layer_name] = inserted_layer

                # Rewrite refs.
                for i in ir.layers[name].inputs:
                    if ":" in i.ref:
                        ref_name = i.ref.split(':')
                        if ref_name[0] == inserted_pre_layer_name:
                            i.ref = f'{inserted_layer_name}:{ref_name[1]}'
                        if ref_name[0] == 'graph_input':
                            i.ref = inserted_layer_name
                    elif i.ref == inserted_pre_layer_name:
                        i.ref = inserted_layer_name

                # Record inserted layer name mapping.
                insert_op_name_dict[name].append((inserted_layer_name, index))
        # Re-sort layers.
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir, insert_op_name_dict

def get_power2_num_list_less_than_8(num, reference_num = 8):
    '''
    Given a positive integer `num`, return a list (or nested lists) of positive integers with:
      1) each element < reference_num
      2) each group (and subgroup) length <= 8
      3) each group (and subgroup) length is a power of two
      4) the sum equals `num`
    '''
    num_list = []
    while num > reference_num:
        num_list.append(reference_num)
        num -= reference_num
    if num > 0:
        bin_str = bin(num)[2:]
        for i in range(len(bin_str)):
            if bin_str[i] == '1':
                num_list.append(2**(len(bin_str)-i-1))
    # Group by max length <= 8.
    num_list_group = []
    while True:
        num_list_group_temp = []
        index = 0
        for i in num_list:
            if len(num_list_group_temp) < 8:
                num_list_group_temp.append(i)
            else:
                num_list_group.append(num_list_group_temp)
                num_list_group_temp = []
                num_list_group_temp.append(i)
            if index == len(num_list)-1:
                num_list_group.append(num_list_group_temp)
            index += 1
        if len(num_list_group) == 1:
            num_list_group = num_list_group[0]
        if len(num_list_group) <= 8:
            if len(num_list_group) > 1:
                # Group by power-of-two lengths.
                len_ = len(num_list_group)
                bin_str_len = bin(len_)[2:]
                len_list = []
                for i in range(len(bin_str_len)):
                    if bin_str_len[i] == '1':
                        len_list.append(2**(len(bin_str_len)-i-1))
                # num_list_group
                num_list_group_all = []
                index = 0
                for i in len_list:
                    num_list_group_all.append(num_list_group[index:index+i])
                    index += i
                if len(num_list_group_all) == 1:
                    num_list_group_all = num_list_group_all[0]
            else:
                num_list_group_all = num_list_group
            break
        else:
            num_list = num_list_group
            num_list_group = []

    return num_list_group_all

def insert_transition_op(ir):
    '''
    For the CIMA architecture, rdc/concat inputs and mcast/seg outputs have an upper bound
    on the number of sources (currently 8). Insert intermediate layers to reduce the
    fan-in/fan-out per node via hierarchical grouping.
    '''
    max_source_num = 8
    # Get successor info.
    next_layer_dict = get_next_layer(ir.layers)
    output_modified_layers = {}
    for k,v in ir.layers.items():
        if v.type == 'op' and len(v.outputs) > max_source_num:
            assert len(v.outputs) <= 64, f"Node '{k}' output fan-out exceeds 64: outputs={len(v.outputs)}."
            output_modified_layers[k] = len(v.outputs)

    if output_modified_layers != {}:
        for ln, ln_num in output_modified_layers.items():
            # Get current layer info.
            current_layer = ir.layers[ln]
            group_num = get_power2_num_list_less_than_8(ln_num, reference_num=max_source_num)
            # group_num = ln_num // max_source_num
            # group_num_list = [max_source_num] * group_num
            if current_layer.type == 'op':
            # seg case
                if (current_layer.op.op_id in ['fused_concat', 'fused_add'] and current_layer.op.split != None) \
                        or current_layer.op.op_id == 'split':
                    assert ln_num % max_source_num == 0, (
                        f"Node '{ln}' output fan-out ({ln_num}) must be a multiple of {max_source_num}."
                    )
                    # Due to hardware constraints, at most two hierarchy levels are used here.
                    assert isinstance(group_num[0], int)
                    # Insert first-level hierarchy layer.
                    level_0_seg_name = ln
                    level_0_seg_num = len(group_num)
                    current_layer_outputs_channel = current_layer.outputs[0].channel * len(current_layer.outputs)
                    assert current_layer_outputs_channel % level_0_seg_num == 0, f'{k}, {v}'
                    axis = 1
                    split = []
                    split_output = []
                    level_0_output_channel = current_layer_outputs_channel // level_0_seg_num
                    for i in range(level_0_seg_num):
                        # Assume uniform channel split by default.
                        split.append(level_0_output_channel)
                        split_output.append(to_cls_obj({'channel': level_0_output_channel,
                                             'width': current_layer.inputs[0].width,
                                             'height': current_layer.inputs[0].height}, DataDef))
                    # op_ = make_op('split', axis=axis, split=split)
                    # split_input = current_layer.inputs
                    # ir.add_layer(level_0_seg_name, op=op_, inputs=split_input, outputs=split_output)
                    if 'fused' in current_layer.op.op_id:
                        ir.layers[ln].op.split.split = split
                    else:
                        ir.layers[ln].op.split = split
                    ir.layers[ln].outputs = split_output
                    # Insert second-level hierarchy layers.
                    c_one = 0
                    #
                    two_level_seg_name = {}
                    c_two = 0
                    for level_1_seg_num in group_num:
                        split = []
                        split_output = []
                        level_1_output_channel = level_0_output_channel // level_1_seg_num
                        level_1_seg_name = f'{ln}_split_level_1_{c_one}'
                        for i in range(level_1_seg_num):
                            # Assume uniform channel split by default.
                            split.append(level_1_output_channel)
                            split_output.append({'channel': level_1_output_channel,
                                                'width': current_layer.inputs[0].width,
                                                'height': current_layer.inputs[0].height})

                        op_ = make_op('split', axis=axis, split=split)
                        split_input = [{'ref': f'{level_0_seg_name}:{c_one}',
                                        'channel': level_0_output_channel,
                                        'height': current_layer.inputs[0].height,
                                        'width': current_layer.inputs[0].width}]
                        ir.add_layer(level_1_seg_name, op=op_, inputs=split_input, outputs=split_output)
                        c_one += 1
                        #
                        c_three = 0
                        for seg_num in range(level_1_seg_num):
                            two_level_seg_name[f'{ln}:{c_two}'] = f'{level_1_seg_name}:{c_three}'
                            c_two += 1
                            c_three += 1
                    # Rewrite original downstream refs.
                    for nl in next_layer_dict[ln]:
                        for in_ in ir.layers[nl].inputs:
                            if in_.ref in two_level_seg_name.keys():
                                in_.ref = two_level_seg_name[in_.ref]

                # mcast case
                else:
                    pass

    # Gather predecessor info.
    # pre_layer_dict = get_pre_layer(ir.layers)
    input_modified_layers = {}
    for k,v in ir.layers.items():
        if v.type == 'op' and v.op.op_id not in ['constant'] and len(v.inputs) > max_source_num:
            assert len(v.inputs) <= 64, f"Node '{k}' input fan-in exceeds 64: inputs={len(v.inputs)}."
            input_modified_layers[k] = len(v.inputs)

    if input_modified_layers != {}:
        for ln, ln_num in input_modified_layers.items():
            # Get current layer info.
            current_layer = ir.layers[ln]
            group_num = get_power2_num_list_less_than_8(ln_num, reference_num=max_source_num)
            # group_num = ln_num // max_source_num
            # group_num_list = [max_source_num] * group_num
            if current_layer.type == 'op':
                # rdc case
                if current_layer.op.op_id in ['fused_add', 'add']:
                    assert ln_num % max_source_num == 0, (
                        f"Node '{ln}' input fan-in ({ln_num}) must be a multiple of {max_source_num}."
                    )
                    # Due to hardware constraints, at most two hierarchy levels are used here.
                    assert isinstance(group_num[0], int)
                    # Insert first-level hierarchy layers.
                    index = 0
                    c = 0
                    current_layer_inputs_list = current_layer.inputs
                    current_layer_outputs_list = current_layer.outputs
                    #
                    level_1_add_input_list = []
                    for level_0_add_num in group_num:
                        level_0_add_name = f'{ln}_add_level_0_{c}'
                        op_ = make_op('add')
                        add_inputs_list = current_layer_inputs_list[index:index+level_0_add_num]
                        ir.add_layer(level_0_add_name, op=op_, inputs=add_inputs_list, outputs=current_layer_outputs_list)
                        index += level_0_add_num
                        c += 1
                        # Record the second-level hierarchy layer names.
                        level_1_add_input_list.append(to_cls_obj({'ref': level_0_add_name,
                                                   'channel': current_layer_inputs_list[0].channel,
                                                   'height': current_layer_inputs_list[0].height,
                                                   'width': current_layer_inputs_list[0].width}, DataDef))
                    # Insert second-level hierarchy layer.
                    ir.layers[ln].inputs = level_1_add_input_list

                # concat case
                elif current_layer.op.op_id in ['fused_concat', 'concat']:
                    assert ln_num % max_source_num == 0, (
                        f"Node '{ln}' input fan-in ({ln_num}) must be a multiple of {max_source_num}."
                    )
                    # Due to hardware constraints, at most two hierarchy levels are used here.
                    assert isinstance(group_num[0], int)
                    # Insert first-level hierarchy layers.
                    index = 0
                    c = 0
                    current_layer_inputs_list = current_layer.inputs
                    current_layer_outputs_list = current_layer.outputs
                    concat_axis = current_layer.op.axis
                    # Record the first-level hierarchy layer names.
                    level_1_concat_input_list = []
                    for level_0_concat_num in group_num:
                        # Compute the first-level output channel count.
                        level_0_concat_output_channel = current_layer_inputs_list[0].channel * level_0_concat_num
                        level_0_concat_name = f'{ln}_concat_level_0_{c}'
                        op_ = make_op('concat', axis=concat_axis)
                        add_inputs_list = current_layer_inputs_list[index:index+level_0_concat_num]
                        current_layer_outputs_list[0].channel = level_0_concat_output_channel
                        ir.add_layer(level_0_concat_name, op=op_, inputs=add_inputs_list, outputs=current_layer_outputs_list)
                        #
                        index += level_0_concat_num
                        c += 1
                        # Record the second-level hierarchy layer names.
                        level_1_concat_input_list.append(to_cls_obj({'ref': level_0_concat_name,
                                                   'channel': current_layer_outputs_list[0].channel,
                                                   'height': current_layer_outputs_list[0].height,
                                                   'width': current_layer_outputs_list[0].width}, DataDef))
                    # Insert second-level hierarchy layer.
                    ir.layers[ln].inputs = level_1_concat_input_list

    ir.dump_json(file=f'test.yaml')

    # Re-sort layers.
    if output_modified_layers != {} or input_modified_layers != {} :
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir


def insert_identity_op(ir):

    # Gather predecessor info.
    next_layer_dict = get_next_layer(ir.layers)
    layers_info = ir.layers

    # Identify layers that require identity insertion.
    need_insert_layer = {}
    for k,v in next_layer_dict.items():
        if not math.log2(len(v)).is_integer():
            need_insert_layer[k] = len(v)

    # Insert identity ops.
    if need_insert_layer != {}:
        for layer_name, output_node_num in need_insert_layer.items():
            # Determine how many downstream nodes each output feeds.
            layer_output_num = len(layers_info[layer_name].outputs)
            assert output_node_num % layer_output_num == 0, (
                f"output_node_num={output_node_num} is not divisible by layer_output_num={layer_output_num}."
            )

            # Split node count.
            split_num = output_node_num // layer_output_num
            # Decompose split_num into a sum of powers of two.
            split_num_list = bin(split_num)[2:]
            #
            insert_node_num_list = []
            for i in range(len(split_num_list)):
                if split_num_list[i] == '1':
                    insert_node_num_list.append(2**(len(split_num_list)-i-1))
            #
            if  layer_output_num != 1:
                assert math.log2(layer_output_num).is_integer(), "layer_output_num must be a power of two."
                # Group next layers by original ref 'layer_name:X'.
                next_layer_sort_dict = {}
                for nl in next_layer_dict[layer_name]:
                    for i in ir.layers[nl].inputs:
                        if layer_name in i.ref:
                            ref_name = i.ref.split(':')
                            if ref_name[1] not in next_layer_sort_dict.keys():
                                next_layer_sort_dict[ref_name[1]] = []
                            next_layer_sort_dict[ref_name[1]].append(nl)

                # Insert identity ops based on insert_node_num_list.
                index = 0
                for i in range(len(insert_node_num_list)):
                    # Rewrite refs that pointed to the original output.
                    for k,v in next_layer_sort_dict.items():
                        # Inserted layer metadata.
                        identity_name = f'{layer_name}_identity_mcast_{i}_seg_{k}'
                        op_ = make_op('identity')
                        ref_layer_name = f'{layer_name}:{i}'
                        input_shape = layers_info[layer_name].outputs[0]
                        inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                        outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                        ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
                        for l in v[index:index+insert_node_num_list[i]]:
                            for j in ir.layers[l].inputs:
                                if layer_name in j.ref:
                                    j.ref = identity_name
                    index += insert_node_num_list[i]
            else:
                # Insert identity ops based on insert_node_num_list.
                index = 0
                for i in range(len(insert_node_num_list)):
                    # Inserted layer metadata.
                    identity_name = f'{layer_name}_identity_{i}'
                    op_ = make_op('identity')
                    ref_layer_name = f'{layer_name}'
                    input_shape = layers_info[layer_name].inputs[0]
                    inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                    outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                    ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)
                    # Rewrite refs that pointed to the original output.
                    nl_list = next_layer_dict[layer_name]
                    for l in nl_list[index:index+insert_node_num_list[i]]:
                        for j in ir.layers[l].inputs:
                            if layer_name in j.ref:
                                j.ref = identity_name
                    index += insert_node_num_list[i]

        # Re-sort layers.
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir


def insert_type_conversion_op(ir, type_conversion_list = None):

    # Gather predecessor info.
    layers_info = ir.layers

    for (name, in_index, conversion_type) in type_conversion_list:

        assert len(in_index) == len(conversion_type)

        # Get current layer info.
        inserted_original_layer = layers_info[name]
        c = 0
        for id in in_index:
            assert conversion_type[c][0] != conversion_type[c][1], (
                f"Type conversion for layer '{name}' must change dtype, got {conversion_type[c]!r}."
            )
            in_info = inserted_original_layer.inputs[id]

            # Inserted layer metadata.
            insert_op_obj = TypeConversionOp().clone()
            insert_op_obj.in_dtype = conversion_type[c][0]
            insert_op_obj.out_dtype = conversion_type[c][1]
            inserted_layer_name = f'{name}_in_type_conversion_{id}'
            inserted_layer_inputs = [copy.deepcopy(in_info)]
            inserted_layer_outputs = [dict(channel=in_info.channel, width=in_info.width, height=in_info.height)]

            inserted_layer = make_layer(op= insert_op_obj, inputs = inserted_layer_inputs,
                                    outputs = inserted_layer_outputs)
            ir.layers[inserted_layer_name] = inserted_layer

            # Rewrite current layer input ref.
            ir.layers[name].inputs[id].ref = inserted_layer_name

            c += 1

    # Re-sort layers.
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir

def get_device_ip(device):
    device_ip = {}
    if isinstance(device, list):
        for dev_ in device:
            assert isinstance(dev_, dict)
            if 'ip' in dev_.keys():
                device_ip[dev_['name']] = dev_['ip']
    elif isinstance(device, dict):
        if 'ip' in dev_.keys():
            device_ip[dev_['name']] = dev_['ip']
    else:
        raise ValueError(f"Unsupported device type: {type(device)}")

    return device_ip

def gen_cima_mapping_info(placed_nodes, hardware_name, window_copy=False):
    node_mapping_info = {}
    for index in range(len(placed_nodes)):
        device_ref = hardware_name[index]
        for node_addr in placed_nodes[index]:
            key = list(node_addr.keys())[0]
            value = list(node_addr.values())[0]
            name_ = key.split('.')
            node_name = name_[0]
            if window_copy:
                index_ = [int(name_[1]),int(name_[2]),int(name_[3].split('_')[0])]
            else:
                index_ = [int(name_[1]),int(name_[2]),int(name_[3])]
            if node_name not in node_mapping_info.keys():
                node_mapping_info[node_name] = []
            mapping_info = CIMADeviceMappingInfo(index = index_, device=device_ref, address=value)
            node_mapping_info[node_name].append(mapping_info)


def get_pre_layer(layers):
    """
    Build a mapping from each layer name to its predecessor layer names.

    input: {layer_name: layer_object}
    return: {current_layer_name: [pre_layer_name, ...]}
    """
    prefix_layer = {}
    for name, layer in layers.items():
        if layer.type not in ['input']:
            prefix_layer[name] =  []
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            # get_layer_ref(layers[name].inputs, layers, prefix_layer[name])
            for i in layer.inputs:
                if 'graph_input' not in i.ref:
                    ref = i.ref
                    if ':' in ref:
                        ref = ref.split(':')[0]
                    pre_layer = layers[ref]
                    if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten','reshape']:
                        for j in pre_layer.inputs:
                            prefix_layer[name].append(j.ref)
                    else:
                        prefix_layer[name].append(ref)
                else:
                    prefix_layer[name].append(i.ref)
    # input()
    return prefix_layer


def get_next_layer(layers):
    '''
    Build a mapping from each layer name to its successor layer names.
    input: {layer_name: layer_object}
    return: {current_layer_name: [next_layer_name, ...]}
    '''
    next_layer = {}
    pre_layer = get_pre_layer(layers)

    for k,v in pre_layer.items():
        #             next_layer[name] = []
        #         next_layer[name].append(k)
        if layers[k].type == 'op' and layers[k].op.op_id in ['flatten']:
            continue

        for name in v:
            if name not in next_layer.keys():
                next_layer[name] = []
            next_layer[name].append(k)

    return next_layer

def draw_square_mesh(grid_size, square_size, arrow_size, text_dict, value_dict, dmac_info,
                     save_fig = None):
    # Create a canvas without ticks.
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Draw text at the specified position (with size/color).
    def draw_text_upon_dash(text, position, x, y, square_size, arrow_size):
        # text_x = x + square_size / 2
        # text_y = y + square_size / 2
        if position == 'top':
            text_x = x + square_size / 2
            text_y = y - arrow_size / 2
        elif position == 'bottom':
            text_x = x + square_size / 2
            text_y = y + square_size + arrow_size / 2
        elif position == 'left':
            text_x = x - arrow_size / 2
            text_y = y + square_size / 2
        elif position == 'right':
            text_x = x + square_size + arrow_size / 2
            text_y = y + square_size / 2

        if position == 'left':
            text_y = text_y - 10
        elif position == 'right' :
            text_y = text_y + 10
        elif position == 'top':
            text_x = text_x - 10
        elif position == 'bottom':
            text_x = text_x + 10

        ax.text(text_x, text_y, text, fontsize = 8, ha='center', va='center', color='purple')

    # Draw text inside a square (with size).
    def draw_text_in_square(text, position, x, y, square_size):
        small_square_size = square_size / 4
        text_x = None
        text_y = None

        if position == 'top':
            text_x = x + square_size / 2
            text_y = y + square_size - small_square_size / 2
        elif position == 'bottom':
            text_x = x + square_size / 2
            text_y = y + small_square_size / 2
        elif position == 'left':
            text_x = x + small_square_size / 2
            text_y = y + square_size / 2
        elif position == 'right':
            text_x = x + square_size - small_square_size / 2
            text_y = y + square_size / 2

        if text_x is not None and text_y is not None:
            ax.text(text_x, text_y, text, fontsize = 6, ha='center', va='center', color='red', fontdict={'weight':'bold'})

    # Draw square grid and arrows.
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = j * (square_size + arrow_size)
            y = i * (square_size + arrow_size)
            ax.text(x  + square_size / 2, y + square_size / 2, f'Node\n[{i}][{j}]', fontsize=7, ha='center', va='center')

            # Write DMAC layer name.
            if (i, j) in dmac_info.keys():
                ln = dmac_info[(i ,j)]
                ax.text(x + 25 , y + square_size / 2 + 10, f'{ln}(DMAC)', fontsize=5, ha='center',
                        va='center', color='red', fontdict={'weight':'bold'})

            # Create the main square.
            square = plt.Rectangle((x, y), square_size, square_size, linewidth=2, edgecolor='black', facecolor='none')

            # Add the square to the figure.
            ax.add_patch(square)

            # Add mini squares.
            small_square_size = square_size / 4
            small_x = x + square_size / 2 - small_square_size / 2
            small_y_top = y + square_size - small_square_size
            small_y_bottom = y
            small_x_left = x
            small_x_right = x + square_size - small_square_size

            # Add the top mini square.
            small_square_top = plt.Rectangle((small_x, small_y_top), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_top)

            # Add the bottom mini square.
            small_square_bottom = plt.Rectangle((small_x, small_y_bottom), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_bottom)

            # Add the left mini square.
            small_square_left = plt.Rectangle((small_x_left, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_left)

            # Add the right mini square.
            small_square_right = plt.Rectangle((small_x_right, y + square_size / 2 - small_square_size / 2), small_square_size, small_square_size, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(small_square_right)

            # Add arrows.
            if j < grid_size[1] - 1:
                # Right arrow.
                right_arrow = plt.Arrow(x + square_size, y + square_size / 2 + 2, arrow_size, 0, color='black', width=arrow_size/2)
                ax.add_patch(right_arrow)
                # Left arrow.
                left_arrow = plt.Arrow(x + square_size + arrow_size, y + square_size / 2 - 2, -arrow_size, 0, color='black', width=arrow_size/2)
                ax.add_patch(left_arrow)
            if i < grid_size[0] - 1:
                # Up arrow.
                up_arrow = plt.Arrow(x + square_size / 2 + 2, y + square_size, 0, arrow_size, color='black', width=arrow_size/2)
                ax.add_patch(up_arrow)
                # Down arrow.
                down_arrow = plt.Arrow(x + square_size / 2 - 2, y + square_size + arrow_size, 0, -arrow_size, color='black', width=arrow_size/2)
                ax.add_patch(down_arrow)

            # Draw text in mini squares if present.
            if (i, j) in text_dict:
                text_position = text_dict[(i, j)]
                for value in text_position:
                    text, position = value
                    draw_text_in_square(text, position, x, y, square_size)

            # Draw text across two adjacent squares if present.
            if (i, j) in value_dict:
                text_position = value_dict[(i,j)]
                for value in text_position:
                    text, position = value
                    draw_text_upon_dash(text, position, x, y, square_size, arrow_size)


    # Set axis limits.
    ax.set_xlim(0, grid_size[1] * (square_size + arrow_size))
    ax.set_ylim(grid_size[0] * (square_size + arrow_size), 0)

    plt.tight_layout()
    # Render figure.
    if save_fig:
        import os
        if '\\' in save_fig:
            path = save_fig.split('\\')
            path = '\\'.join(path[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
        plt.savefig(save_fig)

    plt.close()
    # plt.show()


def draw_mesh_fig(record_io_workload, node_mapping_info, mesh = [6, 6], save_fig_path = '1.svg'):

    # record_io_workload = map.place.record_io_workload

    draw_info = {} # RRAM xb layer
    dash_info = {} # Data communication
    dmac_info = {} # DMAC layer

    draw_loc = {0:'bottom', 1:'right', 2:'top', 3:'left'}

    # node_mapping_info = map.place.node_mapping_info_list

    for name, addr in node_mapping_info.items():
        name_ = name.split('.')[0]

        node_id = int(addr.split('.')[1].split(':')[1])
        figure_id = (node_id // mesh[1], node_id % mesh[1])

        if 'dmac' in addr:
            dmac_info[figure_id] = name_
            continue

        if figure_id not in draw_info.keys():
            draw_info[figure_id] = []
        location = draw_loc[int(addr.split('.')[2].split(':')[1])]
        if '[' in addr.split('.')[-1] and ',' in addr.split('.')[-1]:
            name_ = 'PE_TASK'
            pe_number = addr.split('.')[3].split(':')[1]
            #     p1 = int(pe_number.split('-')[0])
            #     p2 = int(pe_number.split('-')[1])
            #     pe_num = p2 - p1 + 1
            # else:
            #     pe_num = 1
            pe_num = 1
        else:
            name_ = 'Others'
            pe_num = 1
        draw_info[figure_id].append((name_, location, pe_num))

    # Count PE tasks per direction.
    draw_info_pe_task_num = {}
    pe_num_list = []
    for k,v in draw_info.items():
        draw_info_pe_task_num[k] = []
        count_loc = {'bottom':0, 'right':0, 'top':0, 'left':0}
        for n, l, pe_num in v:
            if n == 'PE_TASK':
                count_loc[l] += pe_num
        for d, val in count_loc.items():
            draw_info_pe_task_num[k].append((val, d))
            if val > 2:
                print(k)
                print(l)
                print(val)

            pe_num_list.append(val)

    print(f'Max PE Thread Num: {np.array(pe_num_list).max()}')

    value_list = []

    for loc, value in record_io_workload.items():
        t1 = list(loc.split('-')[0])

        node1_id = [int(t1[1]), int(t1[4])]
        t2 = list(loc.split('-')[1])
        node2_id = [int(t2[1]), int(t2[4])]
        figure_id = (int(t1[1]), int(t1[4]))
        position = None

        if node2_id[1] - node1_id[1] > 0:
            position = 'right'
        elif node2_id[1] - node1_id[1] < 0:
            position = 'left'
        elif node2_id[0] - node1_id[0] > 0:
            position = 'bottom'
        elif node2_id[0] - node1_id[0] < 0:
            position = 'top'
        if figure_id not in dash_info.keys():
            dash_info[figure_id] = []
        dash_info[figure_id].append((str(value), position))
        value_list.append(value)

    draw_square_mesh(mesh, 50, 10, draw_info_pe_task_num, dash_info, dmac_info, save_fig_path)

def replace_op(ir):

    # ================================================================================
    # Replace adjacent operators with equivalent ones to reduce compute cost for a specific target.
    # ================================================================================

    # pattern 1: Conv/FC --> sigmoid -->mul --> y    ==>   Conv/FC --> Silu ---> y
    #               |                    |
    #               --------------------->

    layers = ir.layers
    next_layers_dict = get_next_layer(layers)

    layers_recurrent = copy.deepcopy(layers)

    for layer_name, layer_info in layers_recurrent.items():

        current_layer_info = layer_info
        # Pattern matching:
        # 1) current op is Conv/FC/Add/Concat/etc.
        # 2) next ops include sigmoid and mul
        # 3) sigmoid output feeds mul
        next_sigmoid = False
        next_mul = False
        sigmoid_out_mul = False

        if current_layer_info.type == 'op' and current_layer_info.op.op_id in ['conv2d','fc','linear','matmul', 'fused_conv2d', 'fused_fc',
                                                                               'fused_add', 'add', 'concat', 'fused_concat', 'batch_norm2d']:
            next_layers = next_layers_dict[layer_name]

            for i in next_layers:
                if layers[i].type == 'op':
                    if layers[i].op.op_id == 'sigmoid':
                        next_sigmoid = True
                        sigmoid_next_layer = next_layers_dict[i]
                        # Sigmoid output must only feed mul to be mergeable.
                        if len(sigmoid_next_layer) == 1:
                            snl = sigmoid_next_layer[0]
                            if layers[snl].type == 'op' and layers[snl].op.op_id == 'mul':
                                sigmoid_out_mul = True
                    if layers[i].op.op_id == 'mul':
                        next_mul = True

            if next_sigmoid and next_mul and sigmoid_out_mul:
                # Insert silu.
                op_ = make_op('silu')
                in_height = current_layer_info.outputs[0].height
                in_width = current_layer_info.outputs[0].width
                in_channel = current_layer_info.outputs[0].channel
                input_ = [dict(ref=layer_name, channel= in_channel, width=in_width, height=in_height)]
                silu_name = 'Silu_' + layer_name
                # Activation does not change tensor shape.
                output_ = [dict(channel= in_channel, width=in_width, height=in_height)]
                ir.add_layer(silu_name,op=op_,inputs=input_,outputs=output_)

                # Delete mul/sigmoid and rewrite downstream refs.
                for i in next_layers:
                    if layers[i].type == 'op':
                        if layers[i].op.op_id == 'sigmoid':
                            ir.layers.pop(i)
                        elif layers[i].op.op_id == 'mul':
                            # Rewrite next-layer ref from mul.
                            mul_next_layer = next_layers_dict[i]
                            for mnl in mul_next_layer:
                                for in_ in ir.layers[mnl].inputs:
                                    if in_.ref == i:
                                        in_.ref = silu_name
                            ir.layers.pop(i)

    # Re-sort IR layers.
    ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

    return ir

def remove_flatten_op(ir):
    layers = ir.layers
    layers_recurrent = copy.deepcopy(layers)

    # Remove flatten ops.
    for n, l in layers_recurrent.items():
        if l.type == 'op' and l.op.op_id == 'flatten':
            current_layer_ref = l.inputs[0].ref
            # Rewrite all refs that pointed to flatten.
            for n_,l_ in ir.layers.items():
                for in_ in l_.inputs:
                    if in_.ref == n:
                        in_.ref = current_layer_ref
            ir.layers.pop(n)
    # ir.layers = layers
    return ir

