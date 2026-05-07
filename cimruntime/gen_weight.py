from irtool.core.ir import load_ir, BaseIR
from irtool.tools import flatten_layers  # noqa
import pickle
import numpy as np
from .quant import *
import warnings

def pickle_load(file, **kwargs):
    with open(file, 'rb') as f:
        return pickle.load(f, **kwargs)

def get_addr_record(mappings):
    row_start_addr_record = {}
    col_start_addr_record = {}
    for k,v in mappings.items():
        r_index, h_index, w_index = v.index
        if h_index in row_start_addr_record.keys():
            if h_index != 0:
                assert ((v.address[2] + row_start_addr_record[h_index-1]) == row_start_addr_record[h_index])
        else:
            if h_index == 0:
                row_start_addr_record[h_index] = 0
            else:
                row_start_addr_record[h_index] = v.address[2] + row_start_addr_record[h_index-1]
        if w_index in col_start_addr_record.keys():
            if w_index != 0:
                assert ((v.address[3] + col_start_addr_record[w_index-1]) == col_start_addr_record[w_index])
        else:
            if w_index == 0:
                col_start_addr_record[w_index] = 0
            else:
                col_start_addr_record[w_index] = v.address[3] + col_start_addr_record[w_index-1]
    return row_start_addr_record,col_start_addr_record

def gen_array_weight(ir,weight_file=None,format='CHW', device_shape=(576,128), device='CIMA', **kwargs):
    # Validate IR input type.
    if isinstance(ir,str):
        ir = load_ir(ir)
    elif isinstance(ir, BaseIR):
        ir = ir
    else:
        raise ValueError(f"Unsupported IR type: {type(ir)}. Expected BaseIR or path string.")
    # Validate weights input type (dict or pickle path).
    # {'layer_name':tensor, }
    weight = None
    if isinstance(weight_file,str):
        weight = pickle_load(weight_file)
    elif isinstance(weight_file, dict):
        weight = weight_file
    else:
        raise ValueError(f"Unsupported weight input type: {type(weight_file)}. Expected dict or path string.")

    array_data = {}
    systemc_weight_data = {}

    # onnx_weight_HWC = {}

    # layer info
    layers = ir.flatten_layers()

    for name, layer in layers.items():
        if layer.type in ['input', 'output', 'reuse']:
            continue

        # Read mapping_info from the IR layer.
        mapping_info = None
        if device.lower() == 'cima':
            mapping_info = layer.CIMA_mapping_info
        else:
            raise ValueError(f"Unsupported device {device!r}. This project only supports CIMA(A280).")

        op_id = layer.op.op_id
            # Only mapped layers are materialized into on-chip arrays.
        if op_id in ['matmul','fc','linear','conv2d', 'conv_transpose2d'] and mapping_info != None  :
            weight_name = name +'.weight'
            assert weight_name in weight.keys(), f"Missing weight {weight_name!r}. Available keys: {list(weight.keys())[:10]}..."
            col_repeat_num =  mapping_info.col_repeat_num
            row_repeat_num = mapping_info.row_repeat_num
            wd = weight[weight_name]
            # Conv weights are typically 4D tensors.
            if op_id in ['conv2d', 'conv_transpose2d']:

                if len(wd.shape) == 2:
                    # By convention, output channels are first; transpose to put output channels last.
                    wd = wd.transpose(1,0)

                elif len(wd.shape) == 4:
                    if op_id == 'conv_transpose2d':
                        # Flip horizontally and vertically for transposed convolution.
                        wd = np.flip(wd, axis=(-1, -2))
                        wd = wd.transpose(1, 0, 2, 3)

                    # Default weight layout is CHW.
                    if format == 'HWC':
                        wd = wd.transpose(0,2,3,1)
                        # # (debug) record channel-last transformed weights
                        # onnx_weight_HWC[weight_name] = wd

                        wd = wd.reshape(wd.shape[0],-1,wd.shape[3])
                    elif format == 'CHW':

                        wd = wd.reshape(wd.shape[0],-1)
                        wd = wd.transpose(1,0)
                        wd = np.tile(wd,[row_repeat_num,col_repeat_num])
                    else:
                        raise ValueError(f"Unsupported weight format: {format!r}. Expected 'HWC' or 'CHW'.")
                    # # (legacy) bias handling
                    #     bias_name = name +'.bias'
                    #     bias = weight[bias_name]
                    #         bias = bias.reshape(1,bias.shape[0])
                    #     wd = np.concatenate([wd,bias],axis=0)
                else:
                    raise ValueError(f"Unsupported weight rank/shape: {wd.shape}.")
            elif op_id in ['matmul','fc','linear']:

                if format == 'HWC':
                    # re-order weights when format='HWC' to match the runtime layout.
                    if (len(layer.inputs) != 1):
                        raise ValueError("FC currently supports exactly one dynamic input; weights must be static.")
                    former_layer_name = layer.inputs[0].ref
                    former_layer =  layers[former_layer_name]
                    if former_layer.op.op_id in ['reshape','flatten']:
                        in_channel = former_layer.inputs[0].channel
                        in_h = former_layer.inputs[0].height
                        in_w = former_layer.inputs[0].width
                        assert(wd.shape[1] == in_channel * in_h * in_w)
                        out_d = wd.shape[0]
                        wd = wd.reshape(out_d,in_channel,in_h,in_w)
                        wd = wd.transpose(0,2,3,1)
                        wd = wd.reshape(out_d,-1)
                        # # (debug) record channel-last transformed weights
                        # onnx_weight_HWC[weight_name] = wd
                    # split weights per-branch, reorder, then concatenate back.
                    elif former_layer.op.op_id == 'concat':
                        current_input_row_start = 0
                        transformed_fc_weight = []
                        for in_ in former_layer.inputs:
                            former_former_layer = layers[in_.ref]
                            if former_former_layer.op.op_id in ['reshape','flatten']:
                                # reshape/flatten is assumed to have a single input.
                                assert (len(former_former_layer.inputs) == 1)
                                in_channel = former_former_layer.inputs[0].channel
                                in_h = former_former_layer.inputs[0].height
                                in_w = former_former_layer.inputs[0].width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:,current_input_row_start:current_input_row_end] + 0
                                out_d = current_layer_fc_weight.shape[0]
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d,in_channel,in_h,in_w)
                                current_layer_fc_weight = current_layer_fc_weight.transpose(0,2,3,1)
                                current_layer_fc_weight = current_layer_fc_weight.reshape(out_d,-1)
                                transformed_fc_weight.append(current_layer_fc_weight)
                                # Advance input offset.
                                current_input_row_start = current_input_row_end
                            else:
                                in_channel = in_.channel
                                in_h = in_.height
                                in_w = in_.width
                                row_num = in_channel * in_h * in_w
                                current_input_row_end = current_input_row_start + row_num
                                current_layer_fc_weight = wd[:,current_input_row_start:current_input_row_end] + 0
                                transformed_fc_weight.append(current_layer_fc_weight)
                                # Advance input offset.
                                current_input_row_start = current_input_row_end

                        # Concatenate all branch weights.
                        transformed_fc_weight = np.concatenate(transformed_fc_weight,axis=1)
                        assert (transformed_fc_weight.shape == wd.shape)
                        wd = transformed_fc_weight
                # Transpose so output channels become columns in the array layout.
                wd = wd.transpose(1,0)
                wd = np.tile(wd,[row_repeat_num,col_repeat_num])

            row_record,col_record = get_addr_record(mapping_info.mappings)

            systemc_id = 0
            for k,v in mapping_info.mappings.items():
                r_index, h_index, w_index = v.index
                if h_index == 0:
                    input_row_start = 0
                else:
                    input_row_start = row_record[h_index]
                if w_index == 0:
                    input_col_start = 0
                else:
                    input_col_start = col_record[w_index]
                # input()
                # array_id = int(v.device.split(":")[-1])
                array_id = v.device
                current_row_num = v.address[2]
                current_col_num = v.address[3]
                array_row_start = v.address[0]
                array_col_start = v.address[1]
                array_row_end = array_row_start + current_row_num
                array_col_end = array_col_start + current_col_num
                input_row_end = input_row_start + current_row_num
                input_col_end = input_col_start + current_col_num
                if array_id not in array_data.keys():
                    array_data[array_id] = np.zeros(shape=device_shape)
                if op_id == 'conv2d' and format == 'HWC':
                    # conv2d is split by input channel; row sizes must be multiples of k^2 * row_repeat_num.
                    kernel_size = layer.op.kernel
                    assert(current_row_num % ((kernel_size**2) * row_repeat_num) == 0)
                    assert(input_row_start % ((kernel_size**2) * row_repeat_num) == 0)
                    assert((kernel_size**2) == wd.shape[1])
                    current_channel_num = int(current_row_num / ((kernel_size**2) * row_repeat_num))
                    input_channel_start = int(input_row_start / ((kernel_size**2) * row_repeat_num))
                    input_channel_end = input_channel_start + current_channel_num
                    # input()

                    # Extract the channel slice from the original tensor.
                    current_wd = wd[:,:,input_channel_start:input_channel_end] + 0
                    # Reshape to 2D.
                    current_wd = current_wd.reshape(current_wd.shape[0],-1)
                    # Transpose to [rows, cols].
                    current_wd = current_wd.transpose(1,0)
                    # Tile by [row_repeat_num, col_repeat_num].
                    current_wd = np.tile(current_wd,[row_repeat_num,col_repeat_num])
                    array_data[array_id][array_row_start:array_row_end,array_col_start:array_col_end] = current_wd[:,input_col_start:input_col_end]
                    # systemc_weight_data uses transpose for convenient compute.
                    systemc_weight_data[name+f":{systemc_id}"] = current_wd[:,input_col_start:input_col_end].transpose(1,0)

                else:
                    array_data[array_id][array_row_start:array_row_end,array_col_start:array_col_end] = wd[input_row_start:input_row_end,input_col_start:input_col_end]
                    # systemc_weight_data uses transpose for convenient compute.
                    systemc_weight_data[name+f":{systemc_id}"] = wd[input_row_start:input_row_end,input_col_start:input_col_end].transpose(1,0)
                systemc_id += 1

    return array_data,systemc_weight_data

def Hardware_adaptive_split_weight(onnx_weight, array_size=[576, 128], bn_split_layer_dict=None, split_method='uniform'):

    array_data = {}
    split_layer = []
    for k,v in onnx_weight.items():

        layer_name = k.split('.')[0]
        data_shape = v.shape
        IsNeedSplit = False

        # Skip batchnorm layers.
        if 'bn' in k or 'BatchNormalization' in k:
            if f'{layer_name}_bn.weight' not in onnx_weight.keys():
                array_data[k] = v
            if layer_name in bn_split_layer_dict.keys():
                output_split_num = 0
                input_split_num = 0
                for name in bn_split_layer_dict[layer_name]:
                    osn = int(name.split('_')[-1])
                    isn = int(name.split('_')[-2])
                    if osn > output_split_num:
                        output_split_num = osn
                    if isn > input_split_num:
                        input_split_num = isn
                output_split_num += 1
                input_split_num += 1
                # Split weights.
                bn_weight = onnx_weight[f'{layer_name}.weight']
                bn_bias = onnx_weight[f'{layer_name}.bias']
                bn_mean = onnx_weight[f'{layer_name}.running_mean']
                bn_var = onnx_weight[f'{layer_name}.running_var']
                assert bn_weight.shape[0] % output_split_num == 0
                oc = bn_weight.shape[0] // output_split_num
                for name in bn_split_layer_dict[layer_name]:
                    osn = int(name.split('_')[-1])
                    array_data[f'{name}.weight'] = bn_weight[osn*oc:(osn+1)*oc]
                    array_data[f'{name}.bias'] = bn_bias[osn*oc:(osn+1)*oc]/input_split_num
                    array_data[f'{name}.running_mean'] = bn_mean[osn*oc:(osn+1)*oc]/input_split_num
                    array_data[f'{name}.running_var'] = bn_var[osn*oc:(osn+1)*oc]

            continue

        # constant
        if 'x_Constant' in k:
            array_data[k] = v
            continue

        # LayerNorm layers.
        if 'LayerNormalization' in k:
            array_data[k] = v
            continue

        # Skip activation layers.
        if 'quantizer' in k and 'Silu' not in k:
            continue

        if 'weight' in k:
            in_row = 0
            if len(data_shape) == 4:
                [oc, ic, h1, h2] = data_shape
                in_row = ic * h1 * h2
                if in_row > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            elif len(data_shape) == 2:
                [oc, ic] = data_shape
                in_row = ic
                if ic > array_size[0] or oc > array_size[1]:
                    IsNeedSplit = True
            else:
                raise ValueError(f"Unsupported weight shape: {data_shape}.")

            if IsNeedSplit:
                if split_method == 'uniform':
                    row_split_num = math.ceil(in_row / array_size[0])
                    col_split_num = math.ceil(oc / array_size[1])

                    _, row_value = get_split_num(ic, row_split_num)

                    _, col_value = get_split_num(oc, col_split_num)
                    max_row = np.array(row_value).max()
                    max_col = np.array(col_value).max()

                    for rn in range(row_split_num):
                        for cn in range(col_split_num):
                            start_row = int(np.sum(np.array(row_value[:rn])))
                            end_row =  int(start_row + row_value[rn])
                            start_col =  int(np.sum(np.array(col_value[:cn])))
                            end_col =  int(start_col + col_value[cn])

                            if len(data_shape) == 4:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row,:,:]
                            elif len(data_shape) == 2:
                                array_data[f'{layer_name}_{rn}_{cn}.weight'] = onnx_weight[k][start_col:end_col,start_row:end_row]

                            if rn == row_split_num - 1:
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = onnx_weight[f'{layer_name}.bias'][start_col:end_col]
                            else:
                                if f'{layer_name}.bias' in onnx_weight.keys():
                                    array_data[f'{layer_name}_{rn}_{cn}.bias'] = np.zeros(col_value[cn])

                            if len(data_shape) == 4:
                                if (end_row - start_row) < max_row:
                                    w_oc, w_ic , w_h, w_w = array_data[f'{layer_name}_{rn}_{cn}.weight'].shape
                                    diff_row = max_row - (end_row - start_row)
                                    diff_data = np.zeros((w_oc, diff_row, w_h, w_w))
                                    array_data[f'{layer_name}_{rn}_{cn}.weight'] = np.concatenate([array_data[f'{layer_name}_{rn}_{cn}.weight'], diff_data], axis=1)

                                if (end_col - start_col) < max_col:
                                    w_oc, w_ic , w_h, w_w = array_data[f'{layer_name}_{rn}_{cn}.weight'].shape
                                    diff_col = max_col - (end_col - start_col)
                                    diff_data = np.zeros((diff_col, w_ic, w_h, w_w))
                                    array_data[f'{layer_name}_{rn}_{cn}.weight'] = np.concatenate([array_data[f'{layer_name}_{rn}_{cn}.weight'], diff_data], axis=0)
                            else:
                                raise ValueError(f"2D weight padding is not supported for shape {data_shape}.")

                            # Split batchnorm parameters.
                            if f'{layer_name}_bn.weight' in onnx_weight.keys():
                                for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                                    if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                                        warnings.warn(f"Missing BN param: {layer_name}_bn.{ln}")
                                        continue
                                    if ln == 'num_batches_tracked':
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']
                                    elif ln in ['bias', 'running_mean']:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col] / row_split_num
                                    else:
                                        array_data[f'{layer_name}_{rn}_{cn}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}'][start_col:end_col]

                            # Split scale parameters.
                            if f'{layer_name}.a_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.a_quantizer.s'] = onnx_weight[f'{layer_name}.a_quantizer.s']
                            if f'{layer_name}.w_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.w_quantizer.s'] = onnx_weight[f'{layer_name}.w_quantizer.s']
                            if f'{layer_name}.a_out_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}.a_out_quantizer.s'] = np.array(onnx_weight[f'{layer_name}.a_out_quantizer.s'] / row_split_num)
                            if f'{layer_name}_bn.a_out_quantizer.s' in onnx_weight.keys():
                                array_data[f'{layer_name}_{rn}_{cn}_bn.a_out_quantizer.s'] = np.array(onnx_weight[f'{layer_name}_bn.a_out_quantizer.s'] / row_split_num)

                else:
                    raise ValueError(f"Unsupported split_method: {split_method!r}.")
                split_layer.append(layer_name)
            else:
                array_data[k] = v
                # Do not split batchnorm parameters.
                if f'{layer_name}_bn.weight' in onnx_weight.keys():
                    for ln in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        if f'{layer_name}_bn.{ln}' not in onnx_weight.keys():
                            warnings.warn(f"Missing BN param: {layer_name}_bn.{ln}")
                            continue
                        array_data[f'{layer_name}_bn.{ln}'] = onnx_weight[f'{layer_name}_bn.{ln}']

                # Do not split scale parameters.
                if f'{layer_name}.a_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.a_quantizer.s'] = onnx_weight[f'{layer_name}.a_quantizer.s']
                if f'{layer_name}.w_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.w_quantizer.s'] = onnx_weight[f'{layer_name}.w_quantizer.s']
                if f'{layer_name}.a_out_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}.a_out_quantizer.s'] = onnx_weight[f'{layer_name}.a_out_quantizer.s']
                if f'{layer_name}_bn.a_out_quantizer.s' in onnx_weight.keys():
                    array_data[f'{layer_name}_bn.a_out_quantizer.s'] = onnx_weight[f'{layer_name}_bn.a_out_quantizer.s']

        elif 'bias' in k:
            if layer_name not in split_layer:
                array_data[k] = v
        elif 'Silu' in k:
            array_data[k] = v
        else:
            raise ValueError(f"Unrecognized data key: {k!r}.")
    return array_data

def get_split_num(ic, split_num):
    '''
    input:
        ic: integer
        split_num: number of partitions; each partition should be as even as possible
    return:
        max_num: max partition size after splitting
        num: list of partition sizes
    '''
    t = int(math.ceil(ic / split_num))
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
