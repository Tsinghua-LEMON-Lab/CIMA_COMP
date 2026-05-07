from ..helper import *
from ..placement import *
from ..esti_model import *
import warnings
from scipy.spatial.distance import cdist
from irtool.core import make_op
from ..parser import IrParser
from ..self_defined_op.fused_op import *
import re
import torch

class   Base(object):

    def __init__(self, node_info, node_weight, hardware_config, weight_format='CHW',
                 average_copy=None, specify_para_num=None, specify_split_num=None,
                 place_strategy=OneOnOne, window_copy=False, ir=None,
                 adaptive_split_ir = False, dmac_layer = None, insert_mul_add_op=None, BN_adaptive_split = False, layer_data_type_dict=None):
        '''
        node_info:
            Dict keyed by node name, e.g.:
            {'node_name': {'op_type': str, 'kernel_size': int, 'stride': int, 'calc_num': int,
                           'in_precision': int, 'shape': [w, h], 'in_channel': int, 'out_channel': int,
                           'out_precision': int, 'copy_constraint': int}, ...}
        node_weight:
            Dict keyed by node name. Values are [w, h] before any hardware-aware splitting.
            The convolution weights should already be unrolled in the on-chip layout convention.
        hardware_config:
            Dict describing the target hardware, e.g.:
            {'name': [str], 'xb_number': int, 'xb_shape': [w, h], 'adc_num': int, 'dac_num': int,
             'dac_precision': int}
            'name' encodes the device hierarchy from coarse to fine.
        weight_format:
            'CHW' (default) or 'HWC'.
        specify_para_num:
            Dict specifying parallelism by weight duplication:
            {'node_name': [para_diff_array, para_same_array], ...}.
            If window_copy is False, same-array duplication defaults to a diagonal scheme.
        specify_split_num:
            Dict specifying split factors after duplication: [row_splits, col_splits].
        average_copy:
            Dict specifying averaging-based replication: {'node_name': [repeat_h, repeat_w], ...}.
        place_strategy:
            Placement strategy class, default is OneOnOne.
        window_copy:
            Whether to use window-copy style replication (see the cited paper below):
            Zhang, Y., et al. (2021). Efficient and Robust RRAM-Based Convolutional Weight Mapping
            With Shifted and Duplicated Kernel. IEEE TCAD 40(2): 287-300.
        '''
        self.node_info = node_info
        self.node_weight = node_weight
        self.weight_format = weight_format
        self.average_copy = average_copy
        self.specify_para_num = specify_para_num
        self.specify_split_num = specify_split_num
        self.hardware_config = hardware_config
        self.place_strategy = place_strategy
        self.window_copy = window_copy
        self.ir = ir
        self.adaptive_split_ir = adaptive_split_ir
        self.dmac_layer = dmac_layer
        self.insert_mul_add_op = insert_mul_add_op
        # self.run()
        self.BN_adaptive_split = BN_adaptive_split

    def get_hardware_info(self):
        '''
        '''
        self.XB_num = self.hardware_config['xb_number']
        self.XB_size = self.hardware_config['xb_shape']
        self.hd_name = self.hardware_config['name']
        self.dac_num = self.hardware_config['dac_num']
        self.adc_num = self.hardware_config['adc_num']
        self.dac_precision = self.hardware_config['dac_precision']

        # This project is CIMA(A280)-only; we keep window_copy validation generic.
        if self.window_copy not in (False, True):
            raise TypeError(f"window_copy must be a boolean, got {type(self.window_copy)}.")

        # Device kind string (must be CIMA in this project).
        self.device_field = self.hd_name[0]
        if 'cima' not in self.device_field:
            raise ValueError(
                f"Unsupported device kind for this project: {self.device_field!r}. Expected a CIMA device."
            )


    def split_average(self, CIMA_datawidth = 8):

        '''
        '''
        if self.average_copy != None:
            for i in self.average_copy.keys():
                if i in self.node_weight.keys():
                    w, h = self.node_weight[i]
                    w_ = w * self.average_copy[i][1]
                    h_ = h * self.average_copy[i][0]
                    self.node_weight[i] = [w_, h_]
                else:
                    warnings.warn(f'Message translated to English.')

        if self.weight_format == 'HWC':
            XB_size = self.XB_size
            DMAC_size = None
            if 'cima' in self.hd_name[0]:
                # packaged array size
                if CIMA_datawidth == 8:
                    XB_size = [self.XB_size[0] * 2, self.XB_size[1]]
                elif CIMA_datawidth == 4:
                    XB_size = [self.XB_size[0] * 4, self.XB_size[1]]

                DMAC_size =  self.hardware_config['dmac_shape']

            self.split_node_weight, self.split_num = split_node_HWC(self.node_weight,self.node_info,self.specify_para_num,
                                                                    XB_size, DMAC_size, self.dmac_layer, device=self.device_field)

        elif self.weight_format == 'CHW':

            self.split_num = {}
            for i in self.node_weight.keys():

                if self.specify_para_num != None and i in self.specify_para_num.keys():
                    p_diff_array,p_same_array = self.specify_para_num[i]
                else:
                    p_diff_array,p_same_array = 1, 1

                if self.window_copy and self.node_info[i]['op_type'] in ['conv2d', 'conv_transpose2d']:
                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0]
                        _w = self.specify_split_num[i][1]
                        self.split_num[i] = [p_diff_array, p_same_array, _w, _h] #
                    else:
                        self.split_num[i] = [p_diff_array, p_same_array, 1, 1] #
                else:

                    self.node_weight[i][1] = self.node_weight[i][1] * p_same_array
                    self.node_weight[i][0] = self.node_weight[i][0] * p_same_array

                    if self.specify_split_num != None and i in self.specify_split_num.keys():
                        _h = self.specify_split_num[i][0]
                        _w = self.specify_split_num[i][1]
                    else:
                        _h = math.ceil(self.node_weight[i][1] /  self.XB_size[1])
                        _w = math.ceil(self.node_weight[i][0] /  self.XB_size[0])
                    self.split_num[i] = [p_diff_array,p_same_array, _w, _h]

            if self.window_copy:
                self.split_node_weight, self.split_num = split_node_window_duplicate(self.node_info,self.XB_size,self.split_num)
            else:
                self.split_node_weight = split_node(self.node_weight,self.split_num)

        else:
            raise ValueError(f"Message translated to English.")

        # torch.save(self.split_node_weight, 'split_node_weight.pth')


    def run(self, CIMA_alpha = 0, CIMA_method = 'random_search', CIMA_datawidth = 8, masked_xb = None):
        '''
        self.placed_nodes:
        '''
        self.get_hardware_info()

        self.split_average(CIMA_datawidth = CIMA_datawidth)

        self.split_weight_layer_dict = {}

        self.split_bn_layer_dict = {}

        if self.adaptive_split_ir:

            layers_info = self.ir.layers

            next_layer_dict = get_next_layer(self.ir.layers)
            split_layer_name = []
            new_split_num = {}

            for k, v in self.split_num.items():

                if v[2] * v[3] != 1:
                    current_layer = layers_info[k]
                    # ref_name = current_layer.inputs[0].ref
                    #     ref_name = ref_name.split(':')[0]
                    # former_layer = layers_info[ref_name]
                    split_layer_name.append(k)
                    if v[3] > 1:
                        insert_split_node_name = f'{k}_Split'
                        assert current_layer.op.in_channel % v[3] == 0, f'{k}, {v}'
                        axis = 1
                        split = []
                        split_output = []
                        for i in range(v[3]):
                            split.append(current_layer.op.in_channel // v[3])
                            split_output.append({'channel': int(current_layer.op.in_channel // v[3]),
                                                 'width': current_layer.inputs[0].width,
                                                 'height': current_layer.inputs[0].height})
                        op_ = make_op('split', axis=axis, split=split)
                        split_input = current_layer.inputs
                        self.ir.add_layer(insert_split_node_name, op=op_, inputs=split_input, outputs=split_output)

                    split_in_channel = current_layer.op.in_channel // v[3]
                    if current_layer.op.out_channel % v[2] != 0:
                        warnings.warn(f"Message translated to English.")
                        current_layer.op.out_channel += 1
                    split_out_channel = int(math.ceil(current_layer.op.out_channel // v[2]))

                    in_width = current_layer.inputs[0].width
                    in_height = current_layer.inputs[0].height

                    out_width = current_layer.outputs[0].width
                    out_height = current_layer.outputs[0].height

                    IsSplitBN = False
                    if self.BN_adaptive_split:
                        if len(next_layer_dict[k]) == 1 and hasattr(layers_info[next_layer_dict[k][0]], 'op') and layers_info[next_layer_dict[k][0]].op.op_id == 'batch_norm2d':
                            IsSplitBN = True
                            self.split_bn_layer_dict[next_layer_dict[k][0]] = []
                    ConcatFirst = False

                    if ConcatFirst:
                        for h_ in range(v[3]):
                            for w_ in range(v[2]):
                                new_insert_layer = current_layer.clone()
                                if v[3] > 1:
                                    new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'

                                new_node_name = k + f'_{h_}_{w_}'

                                new_insert_layer.inputs[0].channel = split_in_channel
                                new_insert_layer.outputs[0].channel = split_out_channel

                                original_weight_shape = current_layer.weights['weight'].shape
                                if len(original_weight_shape) == 4:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                                elif len(original_weight_shape) == 2:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                                else:
                                    raise ValueError(f'Message translated to English.')
                                new_insert_layer.op.in_channel = split_in_channel
                                new_insert_layer.op.out_channel = split_out_channel

                                if new_insert_layer.op.op_id in ['fused_conv2d', 'fused_fc'] and v[3] > 1:
                                    if new_insert_layer.op.silu != None:
                                        new_insert_layer.op.silu = None

                                    if new_insert_layer.op.relu != None:
                                        new_insert_layer.op.relu = None

                                if 'bias' in new_insert_layer.weights.keys():
                                    new_insert_layer.weights['bias'].shape = (split_out_channel)

                                self.ir.layers[new_node_name] = new_insert_layer

                                new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]

                                if IsSplitBN:
                                    insert_bn_node_name = f'{k}_BN_{h_}_{w_}'
                                    #
                                    BN_layer = layers_info[next_layer_dict[k][0]].clone()
                                    BN_layer.inputs[0].ref = k + f'_{h_}_{w_}'
                                    BN_layer.op.in_channel = split_out_channel
                                    BN_layer.op.out_channel = split_out_channel
                                    BN_layer.inputs[0].channel = split_out_channel
                                    BN_layer.outputs[0].channel = split_out_channel
                                    BN_layer.op.scale = layers_info[next_layer_dict[k][0]].op.scale[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.bias = layers_info[next_layer_dict[k][0]].op.bias[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_mean = layers_info[next_layer_dict[k][0]].op.input_mean[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_var = layers_info[next_layer_dict[k][0]].op.input_var[w_ * split_out_channel : (w_+1) * split_out_channel]

                                    self.ir.layers[insert_bn_node_name] = BN_layer
                                    self.split_bn_layer_dict[next_layer_dict[k][0]].append(insert_bn_node_name)

                            if v[2] > 1:
                                insert_concat_node_name = f'{k}_Concat_{h_}'
                                op_ = make_op('concat', axis=1)
                                concat_input = []
                                for w_ in range(v[2]):
                                    if IsSplitBN:
                                        ref_name = f'{k}_BN_{h_}_{w_}'
                                    else:
                                        ref_name = k + f'_{h_}_{w_}'
                                    concat_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                                concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                                self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)


                        if v[3] > 1:
                            insert_add_node_name = f'{k}_Add'
                            op_ = make_op('add')
                            if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                if current_layer.op.silu != None or current_layer.op.relu != None:
                                    op_ = make_op('fused_add')

                            add_input = []
                            for h_ in range(v[3]):
                                ref_name = k + f'_{h_}_0'
                                if v[2] > 1:
                                    ref_name = f'{k}_Concat_{h_}'
                                add_input.append(dict(ref=ref_name, channel=current_layer.op.out_channel, width=out_width, height=out_height))
                            add_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]

                            if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                if current_layer.op.silu != None:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                    self.ir.layers[insert_add_node_name].op.silu = {'op_id': 'silu'}
                                elif current_layer.op.relu != None:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                    self.ir.layers[insert_add_node_name].op.relu = {'op_id': 'relu'}
                                else:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                            else:
                                self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)

                        if v[3] > 1:
                            self.split_weight_layer_dict[k] = insert_add_node_name
                        else:
                            self.split_weight_layer_dict[k] = insert_concat_node_name

                    else:
                        for w_ in range(v[2]):
                            for h_ in range(v[3]):
                                new_insert_layer = current_layer.clone()
                                if v[3] > 1:
                                    new_insert_layer.inputs[0].ref = insert_split_node_name + f':{h_}'

                                new_node_name = k + f'_{h_}_{w_}'

                                new_insert_layer.inputs[0].channel = split_in_channel
                                new_insert_layer.outputs[0].channel = split_out_channel

                                original_weight_shape = current_layer.weights['weight'].shape
                                if len(original_weight_shape) == 4:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel, original_weight_shape[2], original_weight_shape[3])
                                elif len(original_weight_shape) == 2:
                                    new_insert_layer.weights['weight'].shape = (split_out_channel, split_in_channel)
                                else:
                                    raise ValueError(f'Message translated to English.')
                                new_insert_layer.op.in_channel = split_in_channel
                                new_insert_layer.op.out_channel = split_out_channel

                                if new_insert_layer.op.op_id in ['fused_conv2d', 'fused_fc'] and v[3] > 1:
                                    if new_insert_layer.op.silu != None:
                                        new_insert_layer.op.silu = None

                                    if new_insert_layer.op.relu != None:
                                        new_insert_layer.op.relu = None

                                if 'bias' in new_insert_layer.weights.keys():
                                    new_insert_layer.weights['bias'].shape = (split_out_channel)

                                self.ir.layers[new_node_name] = new_insert_layer

                                new_split_num[new_node_name] = [self.split_num[k][0], self.split_num[k][1], 1, 1]

                                if IsSplitBN:
                                    insert_bn_node_name = f'{k}_BN_{h_}_{w_}'
                                    #
                                    BN_layer = layers_info[next_layer_dict[k][0]].clone()
                                    BN_layer.inputs[0].ref = k + f'_{h_}_{w_}'
                                    BN_layer.op.in_channel = split_out_channel
                                    BN_layer.op.out_channel = split_out_channel
                                    BN_layer.inputs[0].channel = split_out_channel
                                    BN_layer.outputs[0].channel = split_out_channel
                                    BN_layer.op.scale = layers_info[next_layer_dict[k][0]].op.scale[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.bias = layers_info[next_layer_dict[k][0]].op.bias[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_mean = layers_info[next_layer_dict[k][0]].op.input_mean[w_ * split_out_channel : (w_+1) * split_out_channel]
                                    BN_layer.op.input_var = layers_info[next_layer_dict[k][0]].op.input_var[w_ * split_out_channel : (w_+1) * split_out_channel]

                                    self.ir.layers[insert_bn_node_name] = BN_layer
                                    self.split_bn_layer_dict[next_layer_dict[k][0]].append(insert_bn_node_name)
                            if v[3] > 1:
                                insert_add_node_name = f'{k}_Add_{w_}'
                                op_ = make_op('add')
                                if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                    if current_layer.op.silu != None or current_layer.op.relu != None:
                                        op_ = make_op('fused_add')
                                add_input = []
                                for h_ in range(v[3]):
                                    if IsSplitBN:
                                        ref_name = f'{k}_BN_{h_}_{w_}'
                                    else:
                                        ref_name = k + f'_{h_}_{w_}'
                                    add_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                                add_output = [dict(channel=split_out_channel, width=out_width, height=out_height)]

                                if current_layer.op.op_id in ['fused_conv2d', 'fused_fc']:
                                    if current_layer.op.silu != None:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                        self.ir.layers[insert_add_node_name].op.silu = {'op_id': 'silu'}
                                    elif current_layer.op.relu != None:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                        self.ir.layers[insert_add_node_name].op.relu = {'op_id': 'relu'}
                                    else:
                                        self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)
                                else:
                                    self.ir.add_layer(insert_add_node_name, op=op_, inputs=add_input, outputs=add_output)


                        if v[2] > 1:
                            insert_concat_node_name = f'{k}_Concat'
                            op_ = make_op('concat', axis=1)
                            concat_input = []
                            for w_ in range(v[2]):
                                ref_name = k + f'_0_{w_}'
                                if IsSplitBN:
                                    ref_name = k + f'_BN_0_{w_}'
                                if v[3] > 1:
                                    ref_name = f'{k}_Add_{w_}'
                                concat_input.append(dict(ref=ref_name, channel=split_out_channel, width=out_width, height=out_height))
                            concat_output = [dict(channel=current_layer.op.out_channel, width=out_width, height=out_height)]
                            self.ir.add_layer(insert_concat_node_name, op=op_, inputs=concat_input, outputs=concat_output)

                        if v[2] > 1:
                            self.split_weight_layer_dict[k] = insert_concat_node_name
                        else:
                            self.split_weight_layer_dict[k] = insert_add_node_name

                    next_layer_list = next_layer_dict[k]
                    replace_ref_name = k
                    if IsSplitBN:
                        next_layer_list = next_layer_dict[next_layer_dict[k][0]]
                        replace_ref_name = next_layer_dict[k][0]
                    # input()
                    for nl in next_layer_list:
                        c = 0
                        if ConcatFirst:
                            if v[3] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                    c += 1
                            elif v[2] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                    c += 1
                        else:
                            if v[2] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_concat_node_name
                                    c += 1
                            elif v[3] > 1:
                                for i in self.ir.layers[nl].inputs:
                                    if i.ref == replace_ref_name:
                                        self.ir.layers[nl].inputs[c].ref = insert_add_node_name
                                    c += 1

                    self.ir.layers.pop(k)

                    if IsSplitBN:
                        bn_name = next_layer_dict[k][0]
                        self.ir.layers.pop(bn_name)

                else:
                    new_split_num[k] = v

            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
            if 'cima' in self.hd_name[0]:

                self.ir, _ = fuse_op(self.ir, split_fuse=True)
                self.ir = insert_identity_op(self.ir)

                self.ir = insert_transition_op(self.ir)

            # self.ir.dump_json(file=f'Hardware_adaptive_ir_torch_0509.yaml')
            # exit()
            new_split_node_weight = {}
            for k,v in self.split_node_weight.items():
                k_ = k.split('.')
                if k_[0] in split_layer_name:
                    new_split_node_weight[f'{k_[0]}_{k_[2]}_{k_[3]}.0.0.0'] = v
                else:
                    new_split_node_weight[k] = v
            self.split_node_weight = new_split_node_weight

            ir_parser = IrParser(ir = self.ir)
            self.node_info = ir_parser.node_info

            self.split_num = new_split_num

        self.placed_nodes = self.place_strategy(self.split_node_weight, self.XB_size).run()

        assert isinstance(self.placed_nodes, dict)
        sum_ = 0
        for i in self.placed_nodes:
            v = self.placed_nodes[i]
            ori_lname = ('_').join(i.split('_')[:2])
            if self.dmac_layer != None and ori_lname in self.dmac_layer:
                continue
            sum_ += len(v)
        rest_xb = self.XB_num - sum_
        if rest_xb < 0 :
            raise ValueError(f'Message translated to English.')
        self.ref_to_device( CIMA_alpha = CIMA_alpha, CIMA_method = CIMA_method, CIMA_datawidth = CIMA_datawidth, masked_xb = masked_xb)

    def ref_to_device(self, CIMA_alpha = 0, CIMA_method ='random', CIMA_datawidth = 8, masked_xb = None):
        '''
        return:
        '''
        self.node_mapping_info = {}
        assert len(self.placed_nodes) <= len(self.hd_name)
        if 'cima' not in self.hd_name[0]:
            raise ValueError(
                f"Unsupported device kind for this project: {self.hd_name[0]!r}. Only CIMA(A280) is supported."
            )

        if 'cima' in self.hd_name[0]:

            if self.insert_mul_add_op != None:
                self.ir, self.insert_op_name_dict = insert_mul_add_op(self.ir, mul_add_op=self.insert_mul_add_op)
            else:
                self.insert_op_name_dict = None
            # remove flatten op
            self.ir = remove_flatten_op(self.ir)

            # self.ir.dump_json(file='insert_mul_add_op.yaml')
            # exit(1)
            layer_ref = {}
            for k,v  in self.node_info.items():
                layer_ref[k] = v['ref']

            available_nodes_xb = copy.deepcopy(self.hd_name)
            if masked_xb != None:
                for item in masked_xb:
                    if item in available_nodes_xb:
                        available_nodes_xb.remove(item)

            device_name = list(self.ir.devices.keys())[0]
            mesh_height = self.ir.devices[device_name].height
            mesh_width = self.ir.devices[device_name].width

            alpha = CIMA_alpha
            # limit_child_num = self.limit_child_num
            if CIMA_method.lower() == 'workload_balance':
                # Workload balance search
                self.node_mapping_info_list, self.record_io_workload, self.transfer_thread_num  = packaged_Workload_balance_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                self.node_info,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True,
                                                                                dmac_layer = self.dmac_layer)

            elif CIMA_method.lower() in ['lru_search', 'a_search']:
                # LRU search (Least Recently Used)
                self.node_mapping_info_list, self.record_io_workload = packaged_LRU_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True,
                                                                                dmac_layer = self.dmac_layer)
            elif CIMA_method.lower() == 'random_search':
                # random search
                self.node_mapping_info_list, self.record_io_workload, self.transfer_thread_num  = packaged_random_search(layer_ref, self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True,
                                                                                dmac_layer = self.dmac_layer)
            elif CIMA_method.lower() == 'onebyone_search':
                # onebyone search
                self.node_mapping_info_list, self.record_io_workload = onebyone_search(layer_ref,self.placed_nodes,
                                                                                available_nodes_xb,
                                                                                mesh_height=mesh_height,
                                                                                mesh_width=mesh_width,
                                                                                alpha=alpha,
                                                                                pe_bind_direction=True)
            else:
                raise ValueError(f"Unsupported CIMA_method: {CIMA_method!r}.")

            self.in_line_buffer_addr = {}
            self.credit_len = {}

            # device_linebuffer_assigned
            linebuf_assigned = {}

            assert self.ir != None
            next_layer_dict = get_next_layer(self.ir.layers)
            pre_layer_dict = get_pre_layer(self.ir.layers)

            layers = self.ir.layers
            layers_name = list(layers.keys())

            layers_name.reverse()

            mapping_mesh_node = {}

            # hosti_core = [(3,0)]
            # ddr_core = [(3,5)]
            # none_core = [(1,5), (2,5), (4,5)]
            # "Empty_Core":["Core0_5", "Core3_5", "Core3_6"],
            # "Hosti_Core":"Core0_4",
            # "DDRI_Core":"Core3_4"

            hosti_core = [(0,3), (0, 4), (0,5)]
            ddr_core = [(3, 4), (3,5)]

            self.ddr_core_id = (3, 4)
            self.hosti_core_id = (0, 4)
            none_core = []
            cant_mapped_core = hosti_core + ddr_core + none_core


            all_points = []
            for i in range(mesh_height):
                for j in range(mesh_width):
                    if (i, j) not in cant_mapped_core:
                        all_points.append((i,j))

            self.Max_Memory_Size = 0x100000 // 2

            self.Max_Transfer_Thread_Num = 32

            self.Thread_Dmem_Base_Addr = 640

            self.Max_MFOP_Num = 1
            self.Mapped_MFOP_Core_Full = []
            self.Mapped_MFOP_Num = {}

            for name in layers_name:

                if layers[name].type != 'op':
                    continue

                if layers[name].op.op_id in ['flatten', 'reshape', 'constant']:
                    continue

                if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'silu', 'resize', 'relu',
                                             'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul_add', 'pad',
                                             'type_conversion', 'identity'] \
                        and name not in self.node_mapping_info_list.keys():

                    in_channel = layers[name].inputs[0].channel
                    if in_channel == 255:
                        in_channel += 1
                    if in_channel % 16 != 0:
                        t = 0
                        while (in_channel + t) % 16 != 0:
                            t += 1
                        in_channel = in_channel + t

                    #
                    height = layers[name].inputs[0].height
                    width = layers[name].inputs[0].width
                    if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', ]:
                        kernel_size = layers[name].op.kernel
                        len_ = in_channel * width * kernel_size
                    elif layers[name].op.op_id in ['global_avg_pool2d']:
                        kernel_size = height
                        len_ = in_channel * width * kernel_size
                    elif layers[name].op.op_id in ['concat', 'fused_concat']:
                        len_ = in_channel * width * 4
                        #     len_ *= 4
                    else:
                        len_ = in_channel * width
                        if layers[name].op.op_id in ['add', 'fused_add'] and CIMA_datawidth == 8:
                            len_ *= 4
                    #
                    len_ *= CIMA_datawidth
                    # else:
                    len_ = math.ceil(len_ / 8)


                    nl = []
                    if name in next_layer_dict.keys():
                        nl = next_layer_dict[name]
                    pl = []
                    if name in pre_layer_dict.keys():
                        pl = pre_layer_dict[name]
                    relative_name = nl + pl
                    occupied_core = []
                    for n in relative_name:
                        addr_ = None
                        if n in self.node_mapping_info_list.keys():
                            addr_ = self.node_mapping_info_list[n]
                        elif n + '.0.0.0' in self.node_mapping_info_list.keys():
                            n_ = n + '.0.0.0'
                            if n_ in self.node_mapping_info_list.keys():
                                addr_ = self.node_mapping_info_list[n_]
                        elif n in  mapping_mesh_node.keys():
                            addr_ = mapping_mesh_node[n]

                        if addr_ != None:
                            core_id = int(addr_.split('.')[1].split(':')[1])
                            core = (core_id//mesh_width,core_id%mesh_width)
                            if core not in occupied_core:
                                occupied_core.append(core)

                    rest_possible_nodes = []
                    for x in all_points:
                        # index_rpn = x[0] * mesh_width + x[1]
                        # device_ref_rpn = f'{device_name}.cima-node:{index_rpn}'
                        #     mem_occupied_size = linebuf_assigned[device_ref_rpn][1]
                        # else:
                        #     mem_occupied_size = 0
                        if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'resize']:
                            occupied_core.extend(self.Mapped_MFOP_Core_Full)

                        if x not in occupied_core:
                            rest_possible_nodes.append(x)

                    if rest_possible_nodes == []:
                        raise ValueError(f'Message translated to English.')

                    try:
                        if occupied_core != []:
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                        else:
                            closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[(3,0)])

                    except (np.AxisError) :
                        print(f'Message translated to English.')
                        print(f'Message translated to English.')
                        print(f'Message translated to English.')
                        print(f'Message translated to English.')
                        print(f'Message translated to English.')
                        print(f'Allocate Non-PE Thread Error !!!')
                        exit(1)

                    index_ = closest_point[0] * mesh_width + closest_point[1]

                    device_ref = f'{device_name}.cima-node:{index_}'
                    current_node = device_ref


                    if layers[name].op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', 'resize']:
                        if device_ref not in self.Mapped_MFOP_Num.keys():
                            self.Mapped_MFOP_Num[device_ref] = 1
                        elif self.Mapped_MFOP_Num[device_ref] < self.Max_MFOP_Num:
                            self.Mapped_MFOP_Num[device_ref] += 1
                        #
                        if self.Mapped_MFOP_Num[device_ref] == self.Max_MFOP_Num:
                            self.Mapped_MFOP_Core_Full.append(closest_point)

                    mapping_info = CIMADeviceMappingInfo(index = [0,0,0], device=device_ref, address=0)
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []
                    self.node_mapping_info[name].append(mapping_info)

                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []

                    self.credit_len[name].append( width)
                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr]

                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []

                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])

                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1

                    mapping_mesh_node[name] = current_node

                elif layers[name].op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc',
                                               'split', 'add', 'fused_add', 'fused_concat', 'concat', 'mul_add',
                                               'silu', 'pad', 'relu', 'type_conversion']:
                    # nl_ = name + f'.0.{h1}.{w1}'
                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        nl_ = name
                    else:
                        nl_ = name + f'.0.0.0'

                    assert nl_ in self.node_mapping_info_list.keys()
                    addr = self.node_mapping_info_list[nl_]

                    index_ = [0, 0, 0]
                    if name not in self.node_mapping_info.keys():
                        self.node_mapping_info[name] = []

                    # device_ref = ".".join(addr.split('.')[:2])

                    if layers[name].op.op_id in ['split', 'add', 'fused_add', 'fused_concat', 'concat']:
                        value = 0
                        device_ref = ".".join(addr.split('.')[:2])
                    else:
                        if 'cima-dmac' in addr:
                            value = 0
                            device_ref = addr
                        else:
                            device_ref = ".".join(addr.split('.')[:-1])
                            value_ = addr.split('.')[-1].split(' ')
                            value = []
                            for v in range(len(value_)):
                                if v == 0:
                                    value.append(int(value_[v].split('[')[1].split(',')[0]))
                                elif v == 3:
                                    value.append(int(value_[v].split(']')[0]))
                                else:
                                    value.append(int(value_[v].split(',')[0]))

                    mapping_info = CIMADeviceMappingInfo(index = index_, device=device_ref, address=value)
                    self.node_mapping_info[name].append(mapping_info)
                    current_node = ".".join(addr.split('.')[:2])

                    if current_node not in linebuf_assigned.keys():
                        linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr]
                    if name not in self.credit_len.keys():
                        self.credit_len[name] = []

                    node_info = self.node_info[name]
                    if node_info['in_channel'] == 255:
                        warnings.warn(f'Message translated to English.')
                        node_info['in_channel'] += 1
                    if isinstance(node_info['in_channel'], int):
                        if node_info['in_channel'] % 16 != 0 and 'cima-dmac' not in addr:
                            t = 0
                            while (node_info['in_channel'] + t) % 16 != 0:
                                t += 1
                            warnings.warn(f"Message translated to English.")
                            node_info['in_channel'] = node_info['in_channel'] + t

                    if node_info['op_type'] in ['matmul', 'fused_fc']:
                        len_ = node_info['in_channel']
                        self.credit_len[name].append(1)
                    elif node_info['op_type'] in ['conv2d', 'fused_conv2d']:
                        len_ = node_info['input_shape'][1] * node_info['in_channel'] * max(node_info['kernel_size'], node_info['stride'])
                        self.credit_len[name].append(node_info['input_shape'][1])
                    elif node_info['op_type'] in ['fused_concat', 'concat']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1] * len(layers[name].inputs)

                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    elif node_info['op_type'] in ['split', 'add', 'fused_add', 'mul_add']:
                        len_ = node_info['in_channel'][0] * node_info['input_shape'][0][1]
                        if CIMA_datawidth == 8:
                            if node_info['op_type'] in ['add', 'fused_add']:
                                len_ *= 4
                                #     len_ *= 2
                        # len_ *= 4
                        #     len_ *= 4
                        self.credit_len[name].append(node_info['input_shape'][0][1])
                    else:
                        raise ValueError(f"Message translated to English.")
                    if name not in self.in_line_buffer_addr.keys():
                        self.in_line_buffer_addr[name] = []
                    # self.in_line_buffer_addr[name].append([linebuf_assigned[current_node][1], len_])
                    if 'cima-dmac' in addr:
                        len_ *= 8
                    else: len_ *= CIMA_datawidth

                    # else:
                    len_ = int(math.ceil(len_ / 8))
                    self.in_line_buffer_addr[name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
                    linebuf_assigned[current_node][1] += len_
                    while True:
                        if linebuf_assigned[current_node][1] % 32 == 0:
                            break
                        linebuf_assigned[current_node][1] += 1

                    mapping_mesh_node[name] = current_node

                else:
                    raise ValueError(f'Message translated to English.')

            count = 0


            for layer_name in next_layer_dict.keys():
                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]
                if current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:
                    distance_thr = 1000
                    current_index_num = int(layer_name.split('_')[-1])
                    for pl_ in pre_layer_dict[layer_name]:
                        nums = re.findall(r'\d+', pl_)
                        pre_index_num = int(nums[0])
                        # LastOpDistance.append(current_index_num - pre_index_num)
                        IsInsertDram = False
                        #     IsInsertDram = True
                        #     IsInsertDram = True
                        if (current_index_num - pre_index_num) > distance_thr or IsInsertDram:
                            pl = pre_layer_dict[pl_]
                            nl = [layer_name]
                            #
                            pre_layer = self.ir.layers[pl_]
                            pre_node = int(mapping_mesh_node[pl_].split('.')[1].split(':')[1])
                            pre_node_coor = (pre_node//mesh_width, pre_node%mesh_width)

                            all_possible_nodes = ['DDR']
                            count, linebuf_assigned = self.make_CIMA_transfer_thread(count, pre_layer, pl_, mapping_mesh_node,
                                                                                    pre_node_coor, [pre_node_coor], mesh_width,
                                                                                    pl, nl, linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)
            next_layer_dict = get_next_layer(self.ir.layers)

            for layer_name in next_layer_dict.keys():

                if 'graph_input' in layer_name:
                    continue
                current_layer = self.ir.layers[layer_name]

                if current_layer.type == 'op' and current_layer.op.op_id in ['conv2d','fc','linear', 'matmul', 'fused_conv2d', 'fused_fc',
                                                                             'maxpool2d', 'avgpool2d', 'global_avg_pool2d']:

                    mapped_name = layer_name + '.0.0.0'
                    if current_layer.op.op_id in ['maxpool2d', 'avgpool2d', 'global_avg_pool2d', ]:
                        current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    else:
                        current_node = int(self.node_mapping_info_list[mapped_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node//mesh_width, current_node%mesh_width]

                    nl = next_layer_dict[layer_name]

                    pl = []
                    if layer_name in pre_layer_dict.keys():
                        pl = pre_layer_dict[layer_name]

                    if current_layer.op.op_id in ['conv2d', 'fc', 'linear', 'matmul', 'fused_conv2d', 'fused_fc'] and len(next_layer_dict[layer_name]) == 1:


                        if 'cima-dmac' in self.node_mapping_info_list[mapped_name]:
                            continue
                        pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]

                        pe_number = int(pe_relative.split(':')[-1])


                        if current_node_coor[0] == 0 and pe_number == 0:
                            continue
                        if current_node_coor[0] == mesh_height - 1 and pe_number == 2:
                            continue
                        if current_node_coor[1] == mesh_width - 1 and pe_number == 1:
                            continue
                        if current_node_coor[1] == 0 and pe_number == 3:
                            continue

                        if 'graph_output' in nl:
                            next_node = self.hosti_core_id[0] * mesh_width + self.hosti_core_id[1]
                        else:
                            next_node = int(mapping_mesh_node[nl[0]].split('.')[1].split(':')[1])

                        next_node_coor = (next_node//mesh_width, next_node%mesh_width)

                        if pe_number == 0:
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] < current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node,
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl,
                                                                                linebuf_assigned, all_possible_nodes)

                        elif pe_number == 2:
                            if not (next_node_coor[1] == current_node_coor[1] and next_node_coor[0] > current_node_coor[0]):
                                all_possible_nodes = []
                                for i in range(current_node_coor[0]+1, mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core

                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node,
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl,
                                                                                linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)

                        elif pe_number == 1:
                            if next_node_coor[1] <= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(current_node_coor[1]+1, mesh_width):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core

                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node,
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl,
                                                                                linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)

                        elif pe_number == 3:
                            if next_node_coor[1] >= current_node_coor[1]:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[1]):
                                    if (current_node_coor[0], i) not in none_core:
                                        all_possible_nodes.append((current_node_coor[0], i))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core
                                count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node,
                                                                                current_node_coor, [next_node_coor], mesh_width, pl, nl,
                                                                                linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)

                        else:
                            raise ValueError(f'Message translated to English.')

                    else:

                        next_node_coor = []
                        for nl_ in nl:

                            if 'graph_output' in nl_:
                                next_node_coor.append(self.hosti_core_id)
                            elif 'dram' in mapping_mesh_node[nl_]:
                                for n_ddr in ddr_core:
                                    next_node_coor.append(n_ddr)
                            else:
                                next_node = int(mapping_mesh_node[nl_].split('.')[1].split(':')[1])
                                next_node_coor.append((next_node//mesh_width, next_node%mesh_width))

                        all_possible_nodes = all_points + hosti_core + ddr_core

                        if current_layer.op.op_id in ['conv2d','fc','linear', 'matmul', 'fused_conv2d', 'fused_fc']:

                            pe_relative = self.node_mapping_info_list[mapped_name].split('.')[2]

                            pe_number = int(pe_relative.split(':')[-1])
                            if pe_number == 0 and current_node_coor[0] != 0:
                                all_possible_nodes = []
                                for i in range(0, current_node_coor[0]):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core

                            elif pe_number == 2 and current_node_coor[0] != mesh_height - 1:
                                all_possible_nodes = []
                                for i in range(current_node_coor[0], mesh_height):
                                    if (i, current_node_coor[1]) not in none_core:
                                        all_possible_nodes.append((i, current_node_coor[1]))
                                if all_possible_nodes == []:
                                    all_possible_nodes += ddr_core

                            elif pe_number == 1:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] > current_node_coor[1] and node not in none_core:
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core

                                all_possible_nodes = all_possible_nodes_new

                            elif pe_number == 3:
                                all_possible_nodes_new = []
                                for node in all_possible_nodes:
                                    if node[1] < current_node_coor[1] and node not in none_core:
                                        all_possible_nodes_new.append(node)
                                if all_possible_nodes_new == []:
                                    all_possible_nodes_new += ddr_core
                                all_possible_nodes = all_possible_nodes_new

                        count, linebuf_assigned = self.make_CIMA_transfer_thread(count, current_layer, layer_name, mapping_mesh_node,
                                                                                current_node_coor, next_node_coor, mesh_width, pl, nl,
                                                                                linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)

                elif current_layer.type == 'op' and current_layer.op.op_id in ['fused_concat', 'concat', 'fused_add', 'add']:

                    current_node = int(mapping_mesh_node[layer_name].split('.')[1].split(':')[1])
                    current_node_coor = [current_node//mesh_width, current_node%mesh_width]

                    for il in current_layer.inputs:

                        all_possible_nodes = all_points + hosti_core + ddr_core

                        # last_node_coor = []
                        ref_name = il.ref
                        last_layer_name = ref_name
                        if ':' in ref_name:
                            last_layer_name = ref_name.split(':')[0]


                        last_node = int(mapping_mesh_node[last_layer_name].split('.')[1].split(':')[1])
                        last_node_coor = (last_node//mesh_width, last_node%mesh_width)

                        if tuple(current_node_coor) == last_node_coor:
                            # input()
                            last_layer = self.ir.layers[last_layer_name]

                            nl = [layer_name]

                            pl = []
                            for pln in last_layer.inputs:
                                pl.append(pln.ref)

                            if last_node_coor in all_possible_nodes:
                                all_possible_nodes.remove(last_node_coor)

                            count, linebuf_assigned = self.make_CIMA_transfer_thread(count, last_layer, ref_name, mapping_mesh_node,
                                                                                    last_node_coor, [current_node_coor], mesh_width,
                                                                                    pl, nl, linebuf_assigned, all_possible_nodes, data_width=CIMA_datawidth)

            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))

        else:
            raise ValueError(f'Message translated to English.')

    def make_CIMA_transfer_thread(self, count, current_layer, layer_name, mapping_mesh_node, current_node_coor, next_node_coor,
                                        mesh_width, pl, nl, linebuf_assigned, all_possible_nodes, data_width=8):

        identity_name = f'identity_{count}'
        count += 1
        input_shape = current_layer.outputs[0]

        in_channel = input_shape.channel
        if in_channel == 255:
            warnings.warn(f'Message translated to English.')
            in_channel += 1
        if in_channel % 32 != 0:
            t = 0
            while (in_channel + t) % 32 != 0:
                t += 1
            warnings.warn(f'Message translated to English.')
            in_channel = in_channel + t

        height = input_shape.height
        width = input_shape.width
        #
        len_ = math.ceil(in_channel * width)
        len_ *= data_width
        # else:

        self.make_identity_op(input_shape, layer_name, identity_name)

        device_name = mapping_mesh_node[layer_name].split('.')[0]

        if all_possible_nodes == ['DDR']:
            closest_point = self.ddr_core_id
            device_ref = f'{device_name}.cima-dram'

            index_ = closest_point[0] * mesh_width + closest_point[1]
            current_node = f'{device_name}.cima-node:{index_}'

            len_ = int(math.ceil(len_ / (16 * 256))) * 16 * 256 / 8 # unit: byte
            len_ *= 2

        else:
            # all_possible_nodes = []
            #     all_possible_nodes.append((i, current_node_coor[1]))
            rest_possible_nodes = []

            for x in all_possible_nodes:
                index_rpn = x[0] * mesh_width + x[1]
                device_ref_rpn = f'{device_name}.cima-node:{index_rpn}'

                if device_ref_rpn in linebuf_assigned.keys():
                    mem_occupied_size = linebuf_assigned[device_ref_rpn][1]
                else:
                    mem_occupied_size = 0

                if list(x) != current_node_coor and x not in next_node_coor:
                    rest_possible_nodes.append(x)

                if mem_occupied_size > self.Max_Memory_Size:
                    warnings.warn(f'Message translated to English.')


            # occupied_core = [tuple(current_node_coor), tuple(next_node_coor)]
            occupied_core = [tuple(current_node_coor)] + next_node_coor

            try:
                if occupied_core != []:
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=occupied_core)
                else:
                    closest_point = self.find_closest_point(rest_possible_nodes, linebuf_assigned, len_, mesh_width=mesh_width, exclude_points=[self.hosti_core_id])

            except (np.AxisError) :
                print(f'Message translated to English.')
                print(f'Message translated to English.')
                print(f'Message translated to English.')
                print(f'Message translated to English.')
                print(f'Message translated to English.')
                print(f'Make Transfer Thread Error !!!')
                exit(1)

            index_ = closest_point[0] * mesh_width + closest_point[1]
            device_ref = f'{device_name}.cima-node:{index_}'
            current_node = device_ref
            #
            len_ = int(math.ceil(len_ / 8)) # unit: byte

        if current_node not in linebuf_assigned.keys():
            linebuf_assigned[current_node] = [self.Thread_Dmem_Base_Addr, self.Thread_Dmem_Base_Addr]

        mapping_info = CIMADeviceMappingInfo(index = [0,0,0], device=device_ref, address=0)
        if identity_name not in self.node_mapping_info.keys():
            self.node_mapping_info[identity_name] = []
        self.node_mapping_info[identity_name].append(mapping_info)

        mapping_mesh_node[identity_name] = current_node

        if identity_name not in self.credit_len.keys():
            self.credit_len[identity_name] = []


        if current_layer.op.op_id in ['matmul', 'fused_fc']:
            self.credit_len[identity_name].append(1)
        else:
            self.credit_len[identity_name].append(width)

        if identity_name not in self.in_line_buffer_addr.keys():
            self.in_line_buffer_addr[identity_name] = []
        # self.in_line_buffer_addr[identity_name].append([linebuf_assigned[current_node][1], len_])
        self.in_line_buffer_addr[identity_name].append([hex(linebuf_assigned[current_node][1]), hex(len_)])
        linebuf_assigned[current_node][1] += len_
        while True:
            if linebuf_assigned[current_node][1] % 32 == 0:
                break
            linebuf_assigned[current_node][1] += 1

        for nl_ in nl:
            layer_inputs = self.ir.layers[nl_].inputs
            for li in layer_inputs:
                if li.ref == layer_name:
                    li.ref = identity_name

        return count, linebuf_assigned

    def find_closest_point(self, points, linebuf_assigned, data_len, mesh_width=6,  exclude_points=[]):
        points = np.array(points)
        exclude_points = np.array(exclude_points)

        try:
            distances = cdist(points, exclude_points, metric='cityblock')
        except ValueError:
            print(points)
            print(exclude_points)
            raise ValueError(f'Error!!!')
        distances = np.sum(distances,axis=1)

        # min_distance_index = np.argmin(distances) if distances is not None else None

        sorted_index_list = sorted(range(len(list(distances))), key=lambda k:distances[k])

        # index = np.where(distances == distances[min_distance_index])

        if len(linebuf_assigned.keys()) != 0:

            device_name = list(linebuf_assigned.keys())[0].split('.')[0]

            mem_size = []

            for p_id in sorted_index_list:

                core_id = points[p_id]
                index_ = core_id[0] * mesh_width + core_id[1]
                device_ref = f'{device_name}.cima-node:{index_}'

                if device_ref in self.transfer_thread_num.keys() and self.transfer_thread_num[device_ref] >= self.Max_Transfer_Thread_Num:
                    mem_size.append(0x100000)
                    continue

                #         min_mem_size_id = p_id
                #         CanMap = True
                #         break
                # else:
                #     min_mem_size_id = p_id
                #     CanMap = True
                #     break

                if device_ref in linebuf_assigned.keys():
                    mem_size.append(linebuf_assigned[device_ref][1])
                else:
                    mem_size.append(0)

            sorted_mem_size_id = sorted(range(len(list(mem_size))), key=lambda k:mem_size[k])

            CanMap = False
            for id in sorted_mem_size_id:

                min_mem_size_id = sorted_index_list[id]
                core_id = points[min_mem_size_id]
                index_ = core_id[0] * mesh_width + core_id[1]
                device_ref = f'{device_name}.cima-node:{index_}'
                # device reference
                if device_ref in linebuf_assigned.keys():
                    if (linebuf_assigned[device_ref][1] + data_len) <= self.Max_Memory_Size:
                        CanMap = True
                        break
                else:
                    CanMap = True
                    break
            #
            if not CanMap:
                for p in points:
                    index_ = p[0] * mesh_width + p[1]
                    core_name = f'{device_name}.cima-node:{index_}'
                    print(core_name)
                print(self.transfer_thread_num)
                raise ValueError(f'Message translated to English.')

            if device_ref not in self.transfer_thread_num.keys():
                self.transfer_thread_num[device_ref] = 1
            else:
                self.transfer_thread_num[device_ref] += 1

            # min_mem_size_id = sorted_index_list[np.argmin(np.array(mem_size))]
        else:

            min_mem_size_id = sorted_index_list[0]


        closest_point = tuple(points[min_mem_size_id]) if min_mem_size_id is not None else None

        return closest_point

    def Is_CIMA_mapped_layer(self, layer):
        if layer.type == 'op' and layer.op.op_id in ['conv2d','fc','linear','matmul','fused_conv2d','fused_fc']:
            return True
        return False

    def make_identity_op(self, input_shape, ref_layer_name, identity_name):
        op_ = make_op('identity')
        inputs_ = [dict(ref=ref_layer_name,channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
        outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
        self.ir.add_layer(identity_name,op=op_,inputs=inputs_,outputs=outputs_)

    def update_info(self):
        '''
        return:
        '''
        node_info = {}
        out_loop = 0

        for i in self.node_info.keys():
            if self.window_copy:
                [p,r,w,h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / (p*r))
                out_loop = p
            else:
                [r,w,h] = self.split_num[i]
                calc_num = math.ceil(self.node_info[i]['calc_num'] / r)
                out_loop = r
            for j in range(out_loop):
                for k in range(h):
                    for l in range(w):
                        if self.window_copy:
                            new_name = i+'.'+str(j)+'.'+str(k)+'.'+str(l)+'_wd'
                        else:
                            new_name = i+'.'+str(j)+'.'+str(k)+'.'+str(l)
                        shape = self.split_node_weight[new_name]
                        in_pre = self.node_info[i]['in_precision']
                        out_pre = self.node_info[i]['out_precision']
                        node_info[new_name] = dict(shape=shape,calc_num=calc_num,in_precision=in_pre,out_precision=out_pre)
        return node_info

