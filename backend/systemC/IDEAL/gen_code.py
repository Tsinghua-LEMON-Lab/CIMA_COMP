from .module import *
from irtool.core.ir import BaseIR, load_ir
from irtool.tools import flatten_layers  # noqa
import math
import numpy as np

class CodeGen:

    def __init__(self, ir, input_layer = None, output_layer=None):
        self.ir = ir
        self.input_layer = input_layer
        self.output_layer= output_layer

    def run(self, output_dir=None):
        # self.output_file = output_file
        self.layers = None
        if isinstance(self.ir, BaseIR):
            self.layers = self.ir.flatten_layers()
        elif isinstance(self.ir, str):
            self.ir = load_ir(self.ir)
            self.layers = self.ir.flatten_layers()
        else:
            raise ValueError(f"Message translated to English.")
        output_file_name = 'cfg_compiler_net.h'
        if output_dir != None:
            output_file = output_dir +'\\' + output_file_name
        else:
            output_file = output_file_name

        assert (self.layers != None)
        pre_layer = self.get_pre_layer(self.layers)

        next_layer_mode, next_layer_sequence = self.get_next_layer_mode_sequence(self.layers)

        layer_id = 0

        module = []
        self.layer_id_dict = {}

        self.layer_id_final = {}

        if self.input_layer != None:
            assert isinstance(self.input_layer,str)
            name_ = self.input_layer
            self.layers['graph_input'].inputs[0].height = self.layers[name_].inputs[0].height
            self.layers['graph_input'].inputs[0].width = self.layers[name_].inputs[0].width
            self.layers['graph_input'].inputs[0].channel = self.layers[name_].inputs[0].channel
            self.layers[name_].inputs[0].ref = 'graph_input:0'

        if  self.output_layer != None:
            assert isinstance(self.output_layer,str)
            name_ = self.output_layer
            self.layers['graph_output'].inputs[0].height = self.layers[name_].inputs[0].height
            self.layers['graph_output'].inputs[0].width = self.layers[name_].inputs[0].width
            self.layers['graph_output'].inputs[0].channel = self.layers[name_].inputs[0].channel
            self.layers['graph_output'].inputs[0].ref = self.output_layer

        for name,layer in self.layers.items():
            if layer.type == 'input':
                for i in range(len(layer.inputs)):
                    self.layer_id_dict[name+ f':{i}'] = []
                    self.layer_id_final[name+ f':{i}'] = []
            else:
                self.layer_id_dict[name] = []
                self.layer_id_final[name] = []

            if layer.type == 'input':
                for i in range(len(layer.inputs)):
                    module.append(InBuffer(layer,i))
                    layer_id += 1
                    self.layer_id_dict[name + f':{i}'].append(layer_id - 1)

                    self.layer_id_final[name + f':{i}'].append(layer_id - 1)

            elif layer.type == 'op' :

                source_list = []
                for name_ in pre_layer[name]:
                    for s in self.layer_id_final[name_]:
                        source_list.append(s)

                if layer.op.op_id in ['conv2d','matmul','fc','linear', 'fused_conv2d', 'fused_fc']:
                    if layer.c200_mapping_info == None:
                        continue
                    assert (layer.c200_calc_info != None)

                    if layer.op.op_id in ['conv2d', 'fused_conv2d']:
                        kernel_size = layer.op.kernel
                        stride = layer.op.stride
                        padding = layer.op.padding
                        ifm_row = layer.inputs[0].height
                        ifm_col = layer.inputs[0].width
                    else:
                        kernel_size = 1
                        stride = 1
                        padding = 0
                        ifm_row = 1
                        ifm_col = 1

                    ofm_channel = layer.op.out_channel

                    mapping_info = layer.c200_mapping_info
                    assert(mapping_info.row_split_num % mapping_info.row_repeat_num == 0)
                    assert(mapping_info.col_split_num % mapping_info.col_repeat_num == 0)

                    maps = layer.c200_mapping_info.mappings

                    ofm_row = layer.outputs[0].height
                    ofm_col = layer.outputs[0].width

                    calc_info = layer.c200_calc_info
                    if calc_info.shift_expansion_mode == 'bit_shift':
                        bitwise_mode = 8
                    else:

                    shift = int(math.log2(calc_info.assigned_output_quant_scale))
                    adc_gear = calc_info.it_time

                    avg_id = []
                    for para_diff in range(mapping_info.para_diff_array):

                        add_id = {}

                        for h_index in range(mapping_info.row_split_num):
                            concat_id = {}
                            col_concat_category =  int(mapping_info.col_split_num / mapping_info.col_repeat_num)
                            for w_index in range(mapping_info.col_split_num):

                                index_ = (para_diff,h_index,w_index)
                                maps_info = maps[index_]
                                if layer.op.op_id in ['conv2d', 'fused_conv2d']:
                                    assert (maps_info.address[2] % kernel_size **(2) == 0)
                                    current_in_channel = int(maps_info.address[2] / kernel_size **(2))
                                else:
                                    current_in_channel = maps_info.address[2]
                                current_out_channel = maps_info.address[3]
                                if layer.op.op_id in ['conv2d', 'fused_conv2d']:
                                    module.append(Conv2d(node_id=layer_id,ifm_row=ifm_row,
                                                        ifm_col=ifm_col,ifm_channel=current_in_channel,
                                                        ofm_channel=current_out_channel,stride=stride,padding=padding,
                                                        kernel_size=kernel_size,bitwise_mode=bitwise_mode,
                                                        source_list=source_list,shift=shift,
                                                        adc_gear=adc_gear))

                                else:
                                    module.append(FC(node_id=layer_id,ifm_row=ifm_row,
                                                        ifm_col=ifm_col,ifm_channel=current_in_channel,
                                                        ofm_channel=current_out_channel,stride=stride,padding=padding,
                                                        kernel_size=kernel_size,bitwise_mode=bitwise_mode,
                                                        source_list=source_list,shift=shift,
                                                        adc_gear=adc_gear))

                                layer_id += 1

                                self.layer_id_dict[name].append(layer_id - 1)

                                concat_quotient = math.floor(w_index / col_concat_category)
                                if (concat_quotient) not in concat_id.keys():
                                    concat_id[concat_quotient] = []
                                concat_id[concat_quotient].append(layer_id - 1)

                            after_cat_id = {}
                            for cat in concat_id.keys():
                                if len(concat_id[cat]) > 1:
                                    module.append(Concat(node_id=layer_id, ifm_row=ofm_row, ifm_col=ofm_col,
                                                         ifm_channel=current_out_channel, ofm_channel=ofm_channel,
                                                         source_list=concat_id[cat]))
                                    layer_id += 1
                                after_cat_id[cat] = layer_id - 1

                            for ac in after_cat_id:
                                if (ac) not in add_id.keys():
                                    add_id[ac] = []
                                add_id[ac].append(after_cat_id[ac])

                        for add in add_id.keys():
                            if len(add_id[add]) > 1:
                                module.append(Add(node_id=layer_id, ifm_row=ofm_row, ifm_col=ofm_col,
                                                        ifm_channel=ofm_channel, ofm_channel=ofm_channel,
                                                        source_list=add_id[add]))
                                layer_id += 1
                                avg_id.append(layer_id - 1)
                            else:
                                avg_id.append(add_id[add])

                        if len(avg_id) > 1:
                            module.append(Avg(node_id=layer_id,ifm_row=ofm_row,ifm_col=ofm_col,
                                            ifm_channel=ofm_channel,ofm_channel=ofm_channel,
                                            source_list=avg_id))
                            layer_id += 1

                    # self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)

                elif layer.op.op_id == 'relu':

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        # out_sequence = np.array(next_layer_sequence[name]) + layer_id
                        # out_sequence = out_sequence.tolist()
                        out_sequence = []
                        for nl in next_layer_sequence[name]:
                            sub_sequence = []
                            for t in nl:
                                if isinstance(t,int):
                                    sub_sequence.append(t + layer_id)
                                else:
                                    sub_sequence.append(t)
                            out_sequence.append(sub_sequence)
                    module.append(Relu(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))

                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)

                elif layer.op.op_id == 'add':

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        out_sequence = np.array(next_layer_sequence[name]) + layer_id
                        out_sequence = out_sequence.tolist()
                    module.append(Add(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))

                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)
                elif layer.op.op_id == 'avg':
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        out_sequence = np.array(next_layer_sequence[name]) + layer_id
                        out_sequence = out_sequence.tolist()

                    module.append(Avg(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))

                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)
                elif layer.op.op_id in ['avgpool2d','avg_pool2d','max_pool2d','maxpool2d','global_avg_pool2d']:
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    if 'global' in layer.op.op_id:
                        kernel_size = ifm_row
                        stride = 1
                        padding = 0
                    else:
                        kernel_size = layer.op.kernel
                        stride = layer.op.stride
                        padding = layer.op.padding

                    assert (in_channel == out_channel)
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        out_sequence = np.array(next_layer_sequence[name]) + layer_id
                        out_sequence = out_sequence.tolist()
                    if layer.op.op_id in ['max_pool2d','maxpool2d']:
                        module.append(MaxPool(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            stride=stride, padding=padding, kernel_size=kernel_size,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))
                    else:
                        module.append(AvgPool(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            stride=stride, padding=padding, kernel_size=kernel_size,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))
                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)

                    self.layer_id_final[name].append(layer_id - 1)

                elif layer.op.op_id == 'concat':

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        out_sequence = np.array(next_layer_sequence[name]) + layer_id
                        out_sequence = out_sequence.tolist()
                    module.append(Concat(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                        ifm_channel=in_channel,ofm_channel=out_channel,
                                        source_list=source_list,out_mode=out_mode,
                                        out_sequence=out_sequence))
                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)

                elif layer.op.op_id == 'resize':

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)
                    out_mode = 0
                    out_sequence = None
                    if name in next_layer_mode.keys():
                        out_mode = next_layer_mode[name]
                        out_sequence = []
                        for nl in next_layer_sequence[name]:
                            sub_sequence = []
                            for t in nl:
                                if isinstance(t,int):
                                    sub_sequence.append(t + layer_id)
                                else:
                                    sub_sequence.append(t)
                            out_sequence.append(sub_sequence)
                    module.append(Resize(node_id=layer_id,ifm_row=ifm_row,ifm_col=ifm_col,
                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                            source_list=source_list,out_mode=out_mode,
                                            out_sequence=out_sequence))

                    layer_id += 1
                    self.layer_id_dict[name].append(layer_id - 1)
                    self.layer_id_final[name].append(layer_id - 1)

                elif layer.op.op_id in ['flatten','reshape']:
                    assert len(pre_layer[name]) == 1
                    pre_layer_id_dict = self.layer_id_dict[pre_layer[name][0]][0]
                    pre_layer_id_final = self.layer_id_final[pre_layer[name][0]][0]
                    self.layer_id_dict[name].append(pre_layer_id_dict)
                    self.layer_id_final[name].append(pre_layer_id_final)

                else:

            elif layer.type == 'output':
                source_list = []
                for name_ in pre_layer[name]:
                    for s in self.layer_id_final[name_]:
                        source_list.append(s)
                module.append(OutBuffer(node_id=layer_id,layer_obj=layer,source_list=source_list))
                layer_id += 1
        # input()
        with open(output_file,'w',encoding="utf8") as f:
            f.write("/* **************************************** */\n")
            f.write("/* This file is automatically generated !!! */\n")
            f.write("/*        Please do not modify it !!!      */\n")
            f.write("/* **************************************** */\n")
            f.write("#pragma once\n")
            f.write(f"constexpr auto OP_SIZE = {layer_id};\n")
            f.write("\n")
            f.write("\n")
            f.write(f"static Top_Cfg top_cfg_compiler[OP_SIZE] = \n")
            f.write("{\n")
            f.write("   //|ID  |op_code  |IFM_Col   |IFM_ROW   |IFM_CH  |OFM_CH   |K_Size   |Stride  |Padding   |bitwise_mode  |SourceList  |OutMode   |OutSequence  |Shift  |ADC_GEAR\n")
            for m in module:
                if m.out_sequence != None:
                    new_sequence = []
                    len_ = len(m.out_sequence)
                    start_point = 0
                    for n_1 in m.out_sequence:
                        new_sequence_inner = []
                        for n_2 in n_1:
                            if n_2 in self.layer_id_dict.keys():
                                if len(self.layer_id_dict[n_2]) > 1:
                                    assert len(self.layer_id_dict[n_2]) % len_ == 0
                                    current_len = int(len(self.layer_id_dict[n_2]) / len_)

                                    for cl in range(current_len):
                                        new_sequence_inner.append(self.layer_id_dict[n_2][start_point+cl])
                                    start_point = start_point + current_len
                                else:
                                    new_sequence_inner.append(self.layer_id_dict[n_2][0])
                            else:
                                new_sequence_inner.append(n_2)
                        new_sequence.append(new_sequence_inner)
                    m.out_sequence = new_sequence
                    m.match_mode_sequence()
                f.write(m.gen_code())
                f.write('\n')
            f.write("};\n")

    def get_pre_layer(self,layers):
        ''''
        input: {layer_name:layer_object}
        return: {current_layer_name: pre_layer_name}
        '''
        prefix_layer = {}
        for name,layer in layers.items():
            if layer.type not in ['input']:
                prefix_layer[name] =  []
                for i in layer.inputs:
                    if 'graph_input' not in i.ref:
                        ref_name = i.ref
                        if ":" in ref_name:
                            ref_name = ref_name.split(':')[0]
                        pre_layer = layers[ref_name]
                        if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten','reshape']:
                            for j in pre_layer.inputs:
                                prefix_layer[name].append(j.ref)
                        else:
                            prefix_layer[name].append(i.ref)
                    else:
                        prefix_layer[name].append(i.ref)
        return prefix_layer


    def get_next_layer(self,layer):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_name}
        '''
        next_layer = {}
        pre_layer = self.get_pre_layer(layer)
        for k,v in pre_layer.items():
            for name in v:
                if ':' in name:
                    name = name.split(':')[0]
                if name not in next_layer.keys():
                    next_layer[name] = []
                next_layer[name].append(k)
        return next_layer

    def get_mode_sequence(self,next_layer_, name, out_mode, out_sequence, next_layer_name):

        op_id = 0

        if next_layer_.op.op_id in ['fc','linear','matmul','conv2d'] and next_layer_.c200_mapping_info != None:
            assert (next_layer_.c200_mapping_info != None)
            assert (next_layer_.c200_calc_info != None)

            if next_layer_.op.op_id == 'conv2d':
                kernel_size = next_layer_.op.kernel
            else:
                kernel_size = 0

            mapping_info = next_layer_.c200_mapping_info
            para_diff_array = mapping_info.para_diff_array
            row_repeat_num = mapping_info.row_repeat_num
            row_split_num = mapping_info.row_split_num
            col_repeat_num = mapping_info.col_repeat_num
            col_split_num = mapping_info.col_split_num
            mappings = mapping_info.mappings
            row_category = int(row_split_num / row_repeat_num )

            assert (para_diff_array == 1)
            if name in out_sequence.keys():
                for i in out_sequence[name].keys():
                    out_sequence[name][i].append(next_layer_name)
            else:
                out_mode[name] = {}
                out_sequence[name] = {}
                for i in range(row_split_num):

                    for j in range(col_split_num):
                        maps_info = mappings[(0,i,j)]
                        if next_layer_.op.op_id == 'conv2d':
                            assert (maps_info.address[2] % kernel_size **(2) == 0)
                            current_in_channel = int(maps_info.address[2] / kernel_size **(2))
                        else:
                            current_in_channel = maps_info.address[2]
                        quotient = i % row_category
                        if quotient in out_mode[name].keys():

                            assert(current_in_channel == out_mode[name][quotient])
                        else:
                            out_sequence[name][quotient] = []
                            out_mode[name][quotient] = current_in_channel

                        op_id += 1
                        out_sequence[name][quotient].append(op_id)

                    if col_split_num > 1:
                        op_id += col_repeat_num
        elif next_layer_.op.op_id in ['add']:
            if name in out_sequence.keys():
                for i in out_sequence[name].keys():
                    out_sequence[name][i].append(next_layer_name)

        return out_mode, out_sequence

    def get_next_layer_mode_sequence(self,layer):
        '''
        input: {layer_name:layer_object}
        return:{layer_name:[out_mode]}, {layer_name:[[out_squence]]}
        '''
        next_layer = self.get_next_layer(layer)
        out_mode = {}
        out_sequence = {}

        for name in list(layer.keys()):
            if layer[name].type == 'output':
                continue
            if layer[name].type == 'op' and layer[name].op.op_id in ['reshape','flatten']:
                continue
            current_layer = layer[name]
            if current_layer.type == 'input':
                for n in next_layer[name]:
                    next_layer_ = layer[n]
                    if next_layer_.type != 'op':
                        continue
                    else:
                        out_mode, out_sequence = self.get_mode_sequence(next_layer_,name,out_mode,out_sequence,n)
            else:
                for n in next_layer[name]:
                    next_layer_ = layer[n]
                    if next_layer_.type != 'op':
                        continue
                    out_mode, out_sequence = self.get_mode_sequence(next_layer_,name,out_mode,out_sequence,n)

        assert len(out_mode.keys()) == len(out_sequence.keys())
        out_mode_list = {}
        out_sequence_list = {}
        for name in list(out_mode.keys()):
            if len(out_mode[name].keys()) > 1:
                assert len(out_mode[name].keys()) ==  len(out_sequence[name].keys())
                out_mode_list[name] = []
                out_sequence_list[name] = []
                for mode in list(out_mode[name].keys()):
                    out_mode_list[name].append((out_mode[name][mode]))
                for sq in list(out_sequence[name].keys()):
                    out_sequence_list[name].append(out_sequence[name][sq])
            else:
                continue
        return out_mode_list, out_sequence_list

    @staticmethod
    def gen_weight(quant_weight,layer_id_dict,weight_file=None):
        assert isinstance(quant_weight,dict)
        if weight_file == None:
            weight_file = 'weight.txt'
        with open(weight_file,'w+',encoding='utf-8') as f:
            f.write("/* **************************************** */\n")
            f.write("/* This file is automatically generated !!! */\n")
            f.write("/*        Please do not modify it !!!      */\n")
            f.write("/* **************************************** */\n")
            for weight_name_id in list(quant_weight.keys()):
                weight_name = weight_name_id.split(':')[0]
                index = int(weight_name_id.split(':')[-1])
                Node_ID = layer_id_dict[weight_name][index]
                f.write(f"Node_ID:{Node_ID}\n")
                weight_data = quant_weight[weight_name_id]
                for t in range(weight_data.shape[0]):
                    for z in range(weight_data.shape[1]):
                        f.write(f'{int(weight_data[t][z])},  ')
                    f.write("\n")
                f.write("\n")



