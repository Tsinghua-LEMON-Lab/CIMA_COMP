from .module import *
from irtool.core.ir import BaseIR, load_ir
from irtool.tools import flatten_layers  # noqa
import math
import numpy as np
import copy

class CodeGen:

    def __init__(self, ir, input_layer = None, output_layer=None):
        self.ir = ir
        self.input_layer = input_layer
        self.output_layer= output_layer

    def run(self,output_file=None):
        self.output_file = output_file
        self.layers = None
        if isinstance(self.ir, BaseIR):
            self.layers = self.ir.flatten_layers()
        elif isinstance(self.ir, str):
            self.ir = load_ir(self.ir)
            self.layers = self.ir.flatten_layers()
        else:
            raise ValueError(f"Message translated to English.")
        first_device_key = list(self.ir.devices.keys())[0]
        MESH_HEIGHT = self.ir.devices[first_device_key].height
        MESH_WIDTH = self.ir.devices[first_device_key].width
        TASK_NUM = self.ir.devices[first_device_key].task_num

        output_file = 'rcas_task_cfg.h'
        if self.output_file != None:
            output_file = self.output_file +'\\' + 'rcas_task_cfg.h'

        assert (self.layers != None)
        pre_layer = self.get_pre_layer(self.layers)

        next_layer = self.get_next_layer(self.layers)

        # next_layer_mode, next_layer_sequence = self.get_next_layer_mode_sequence(self.layers)

        module = []
        for i in range(MESH_HEIGHT):
            core_width = []
            for j in range(MESH_WIDTH):
                task = []
                for k in range(TASK_NUM):
                    task.append(BaseModule(valid=0))
                core_width.append(task)
            module.append(core_width)

        self.core_task_id = {}
        for i in range(MESH_HEIGHT):
            for j in range(MESH_WIDTH):
                self.core_task_id[f'[{i}][{j}]'] = 0

        self.layer_core_dict = {}

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

        layers_name = list(self.layers.keys())
        # layes_name.reverse()

        LAYER_LEN = len(layers_name)


        for name in layers_name:
            layer = self.layers[name]

            if layer.type == 'input':
                #     self.layer_id_dict[name+ f':{i}'] = []
                #     self.layer_id_final[name+ f':{i}'] = []
                assert len(layer.inputs) == 1
                self.layer_core_dict[name + ':0'] = [0,  0 , 0]
                self.core_task_id[f'[0][0]'] += 1

            elif layer.type == 'output':
                #     self.layer_id_dict[name+ f':{i}'] = []
                #     self.layer_id_final[name+ f':{i}'] = []
                assert len(layer.inputs) == 1
                self.layer_core_dict[name] = [MESH_HEIGHT-1, 0 , 0]
                self.core_task_id[f'[{MESH_HEIGHT-1}][0]'] += 1

            elif layer.type == 'op' :

                if layer.CIMA_mapping_info == None:
                    continue

                mapping_info = layer.CIMA_mapping_info
                assert(mapping_info.row_split_num % mapping_info.row_repeat_num == 0)
                assert(mapping_info.col_split_num % mapping_info.col_repeat_num == 0)

                maps = layer.CIMA_mapping_info.mappings

                assert mapping_info.para_diff_array == 1
                assert mapping_info.row_split_num == 1
                assert mapping_info.col_split_num == 1

                index_ = (0,0,0)
                maps_info = maps[index_]

                device = maps_info.device.split('.')

                index_ = int(device[1].split(':')[1])
                current_mesh_height = index_ // MESH_WIDTH
                current_mesh_width = index_ % MESH_WIDTH

                current_task_id = self.core_task_id[f'[{current_mesh_height}][{current_mesh_width}]']

                self.layer_core_dict[name] = [current_mesh_height, current_mesh_width, current_task_id]
                assert self.core_task_id[f'[{current_mesh_height}][{current_mesh_width}]'] <= TASK_NUM - 1
                self.core_task_id[f'[{current_mesh_height}][{current_mesh_width}]'] += 1

            else:

        memout = BaseModule(valid=0)

        for name in layers_name:
            layer = self.layers[name]

            if layer.type in ['input']:
                continue

            elif layer.type in ['output']:
                source_list = []
                assert len(pre_layer[name]) == 1

                for name_ in pre_layer[name]:
                    assert name_ in self.layer_core_dict.keys()
                    source_list.append(self.layer_core_dict[name_])
                pre_layer_obj = self.layers[pre_layer[name][0]]

                in_linebuffer_width = pre_layer_obj.CIMA_mapping_info.in_line_buffer_addr[0]
                credit_len = pre_layer_obj.CIMA_mapping_info.credit_len[0]
                memout = OutBuffer(task_id=0, layer_obj=layer, source_list=source_list,
                                   in_linebuffer_width=in_linebuffer_width, credit_len=credit_len)

            elif layer.type == 'op' :

                if layer.CIMA_mapping_info == None:
                    continue

                source_list = []
                for name_ in pre_layer[name]:
                    assert name_ in self.layer_core_dict.keys()
                    source_list.append(self.layer_core_dict[name_])

                out_sequence = []
                for name_ in next_layer[name]:
                    core_dict = copy.deepcopy(self.layer_core_dict[name_])
                    channel = self.layers[name_].inputs[0].channel
                    core_dict.append(channel)
                    out_sequence.append(core_dict)

                mapping_info = layer.CIMA_mapping_info
                assert(mapping_info.row_split_num % mapping_info.row_repeat_num == 0)
                assert(mapping_info.col_split_num % mapping_info.col_repeat_num == 0)

                maps = layer.CIMA_mapping_info.mappings

                assert mapping_info.para_diff_array == 1
                assert mapping_info.row_split_num == 1
                assert mapping_info.col_split_num == 1

                index_ = (0,0,0)
                maps_info = maps[index_]

                device = maps_info.device.split('.')

                index_ = int(device[1].split(':')[1])
                current_mesh_height = index_ // MESH_WIDTH
                current_mesh_width = index_ % MESH_WIDTH

                in_linebuffer_width = mapping_info.in_line_buffer_addr[0]
                credit_len = mapping_info.credit_len[0]
                current_task_id = self.layer_core_dict[name][-1]
                # input()
                if layer.op.op_id in ['conv2d','matmul','fc','linear','fused_conv2d','fused_fc']:

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

                    if layer.op.op_id == 'conv2d':
                        assert (maps_info.address[2] % kernel_size **(2) == 0)
                        current_in_channel = int(maps_info.address[2] / kernel_size **(2))
                    else:
                        current_in_channel = maps_info.address[2]
                    current_out_channel = maps_info.address[3]

                    direction = int(device[-1].split(':')[1]) # N:0,  E:1, S:2, W:3
                    direction = (direction + 1) % 4 # W:0, N:1,  E:2, S:3

                    relu = 0
                    if layer.op.op_id in ['fused_conv2d', 'fused_fc'] and layer.op.relu != None:
                        relu = 1

                    if layer.op.op_id in ['conv2d', 'fused_conv2d']:
                        module[current_mesh_height][current_mesh_width][current_task_id] = Conv2d(task_id=current_task_id,ifm_row=ifm_row,
                                                                                                ifm_col=ifm_col,ifm_channel=current_in_channel,
                                                                                                ofm_channel=current_out_channel,stride=stride,padding=padding,
                                                                                                kernel_size=kernel_size,source_list=source_list, out_sequence = out_sequence,
                                                                                                in_linebuffer_width = in_linebuffer_width,
                                                                                                credit_len = credit_len, pe_index=direction,relu=relu, len=LAYER_LEN)
                    else:
                        module[current_mesh_height][current_mesh_width][current_task_id] = FC(task_id=current_task_id,ifm_row=ifm_row,
                                                                                                ifm_col=ifm_col,ifm_channel=current_in_channel,
                                                                                                ofm_channel=current_out_channel,stride=stride,padding=padding,
                                                                                                kernel_size=kernel_size,source_list=source_list,out_sequence = out_sequence,
                                                                                                in_linebuffer_width = in_linebuffer_width,
                                                                                                credit_len = credit_len, pe_index=direction,relu=relu, len=LAYER_LEN)

                elif layer.op.op_id in ['add', 'fused_add']:

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)
                    relu = 0
                    if layer.op.op_id == 'fused_add':
                        relu = 1
                    module[current_mesh_height][current_mesh_width][current_task_id] = Add(task_id=current_task_id,
                                                                                           ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len,relu=relu, len=LAYER_LEN)

                elif layer.op.op_id == 'avg':
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    assert (in_channel == out_channel)

                    module[current_mesh_height][current_mesh_width][current_task_id] = Add(task_id=current_task_id,
                                                                                           ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, len=LAYER_LEN)

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

                    if layer.op.op_id in ['max_pool2d','maxpool2d']:
                        module[current_mesh_height][current_mesh_width][current_task_id] = MaxPool(task_id=current_task_id,
                                                                                            ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, len=LAYER_LEN)
                    else:
                        module[current_mesh_height][current_mesh_width][current_task_id] = AvgPool(task_id=current_task_id,
                                                                                            ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, len=LAYER_LEN)


                elif layer.op.op_id in ['concat', 'fused_concat']:

                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    relu = 0
                    if layer.op.op_id == 'fused_concat':
                        relu = 1
                    module[current_mesh_height][current_mesh_width][current_task_id] = Concat(task_id=current_task_id,
                                                                                            ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, relu=relu, len=LAYER_LEN)

                elif layer.op.op_id in ['flatten','reshape']:
                    continue

                elif layer.op.op_id in ['identity']:
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel

                    module[current_mesh_height][current_mesh_width][current_task_id] = Identity(task_id=current_task_id,
                                                                                            ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, len=LAYER_LEN)

                elif layer.op.op_id in ['split']:
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel

                    module[current_mesh_height][current_mesh_width][current_task_id] = Split(task_id=current_task_id,
                                                                                            ifm_row=ifm_row,ifm_col=ifm_col,
                                                                                            ifm_channel=in_channel,ofm_channel=out_channel,
                                                                                            source_list=source_list, out_sequence = out_sequence,
                                                                                            in_linebuffer_width = in_linebuffer_width,
                                                                                            credit_len = credit_len, len=LAYER_LEN)


                else:

        with open(output_file,'w',encoding="utf8") as f:
            f.write("/* **************************************** */\n")
            f.write("/* This file is automatically generated !!! */\n")
            f.write("/*        Please do not modify it !!!      */\n")
            f.write("/* **************************************** */\n")
            # f.write("#pragma once\n")
            # f.write(f"constexpr auto OP_SIZE = {layer_id};\n")
            f.write(self.gen_const_str())
            f.write("\n")
            f.write("\n")
            f.write(f"static Task_Cfg_struct task_cfg_list[{MESH_HEIGHT * MESH_WIDTH}][{TASK_NUM}] = \n")
            f.write("{\n")
            # f.write("   //|ID  |op_code  |IFM_Col   |IFM_ROW   |IFM_CH  |OFM_CH   |K_Size   |Stride  |Padding   |bitwise_mode  |SourceList  |OutMode   |OutSequence  |Shift  |ADC_GEAR\n")
            count_h = 0
            for m_height in module:
                count_w = 0
                for m_width in m_height:
                    f.write(f'   //Core{count_h}_{count_w} \n')
                    f.write("   {\n")
                    for task in m_width:
                        f.write("     {\n")
                        f.write(task.gen_code())
                        f.write("     },")
                        f.write('\n')
                    f.write("   },\n")
                    count_w += 1
                count_h += 1
            f.write("};\n")
            # generate memout code

            f.write('static Task_Cfg_struct task_cfg_memout_new = { \n')
            f.write(memout.gen_code())
            f.write("};")

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
                        pre_layer = layers[i.ref]
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
        # input()
        for k,v in pre_layer.items():
            if  self.layers[k].type == 'op' and self.layers[k].op.op_id in ['flatten']:
                continue
            for name in v:
                if name not in next_layer.keys():
                    next_layer[name] = []
                next_layer[name].append(k)
        return next_layer

    def get_mode_sequence(self,next_layer_, name, out_mode, out_sequence, next_layer_name):

        op_id = 0

        if next_layer_.op.op_id in ['fc','linear','matmul','conv2d'] and next_layer_.CIMA_mapping_info != None:
            assert (next_layer_.CIMA_mapping_info != None)
            assert (next_layer_.CIMA_calc_info != None)

            if next_layer_.op.op_id == 'conv2d':
                kernel_size = next_layer_.op.kernel
            else:
                kernel_size = 0

            mapping_info = next_layer_.CIMA_mapping_info
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
                for i in range(len(current_layer.inputs)):
                    for n in next_layer[name+f':{i}']:
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

    def gen_const_str(self):
        return '''
#pragma once
#include <vector>
#include "rcas_arch_cfg.h"
using namespace std;

typedef pair<int, int> node_id;

typedef struct {
	int		src_merge;
	node_id	dst_id;
	int		dst_task;
}Addr_encode;



enum Op_code { R_MVM, D_MVM, MaxPool, AvgPool, Add, Concat, Split, Identity };
enum DIR { W, B, E, S, NA };

typedef struct {
	int src_list;//(row*meshcol + col)*16+task
	pair<int, int> src_write_init_sride;//not used, for IO magic
	bool src_add_flag;//not used, for IO magic
}Src_struct;

typedef struct {
	int fin_size;
	int cin;
	int cout;
	int k_size;
	int stride;
	int padding;
	int rela_pe;//0~3
	int bitwise;
	int relu;
}Conv_struct;

typedef struct {
	bool vld;
	int task_id;
	pair<int,int> addr_base_offset;//global LB for assertion
	int credit_pix_len;
	vector<Src_struct> src_struct_list;
	int op_code;
	Conv_struct conv_attribute;
	vector<pair<int,int>> dst_chcast_list; //(row*meshcol + col)*16+task
										   //channel split within same pix

}Task_Cfg_struct;
'''

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



