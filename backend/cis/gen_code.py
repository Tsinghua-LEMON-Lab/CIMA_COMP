from irtool.core.ir import BaseIR, load_ir
from irtool.tools import flatten_layers  # noqa
from .module import *

class CodeGen:

    def __init__(self, ir):
        self.ir = ir

    def run(self, output_file=None):

        if isinstance(self.ir, BaseIR):
            self.layers = self.ir.flatten_layers()
        elif isinstance(self.ir, str):
            self.ir = load_ir(self.ir)
            self.layers = self.ir.flatten_layers()
        else:
            raise ValueError(f"Message translated to English.")

        if output_file == None:
           output_file = 'cis_code.txt'

        mvm_relu_layer = self.get_mvm_relu_layer(self.layers)

        layer_bias_addr_offset = self.get_bias_addr_offset(self.layers)

        ordered_layer,cross_index_layer = self.get_ordered_layer(self.layers)

        next_layer_id = self.get_next_layer_id(self.layers)

        module = []
        # gen code
        for name, layer in self.layers.items():
            if layer.type == 'op':
                if layer.op.op_id  == 'conv2d':
                    op_id = ordered_layer[name]
                    target_id = next_layer_id[name]
                    kernel_size = layer.op.kernel
                    stride = layer.op.stride
                    padding = layer.op.padding
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    ofm_row = layer.outputs[0].height
                    ofm_col = layer.outputs[0].width
                    maps = layer.mapping_info.mappings
                    if len(maps) > 1:
                    index_ = (0,0,0)
                    array_row_start = maps[index_].address[0]
                    array_id = maps[index_].device.split(':')[-1]
                    func_mode = cross_index_layer[name]
                    bias_addr_offset = layer_bias_addr_offset[name]
                    array_col_start = maps[index_].address[1]
                    array_col_num = maps[index_].address[3]
                    adc_select = self.gen_adc_index(array_col_start,array_col_num)
                    relu = 0
                    if name in mvm_relu_layer:
                        relu = 1
                    bias_en = 0
                    if layer.op.bias == True:
                        bias_en = 1
                    in_channel = layer.op.in_channel
                    out_channel = layer.op.out_channel
                    module.append(Conv2d(op_id=op_id, target_id=target_id, array_id=array_id, adc_select=adc_select,
                                    array_row_start=array_row_start, relu=relu, bias_addr_offset=bias_addr_offset,
                                    bias_en=bias_en, stride=stride, padding=padding, out_channel = out_channel,
                                    in_channel=in_channel , kernel_size=kernel_size, out_feature_map_col=ofm_col,
                                    out_feature_map_row = ofm_row, in_feature_map_col=ifm_col, in_feature_map_row=ifm_row,
                                    func_mode=func_mode))

                elif layer.op.op_id in ['matmul','fc','linear']:
                    op_id = ordered_layer[name]
                    target_id = next_layer_id[name]
                    maps = layer.mapping_info.mappings
                    if len(maps) > 1:
                    index_ = (0,0,0)
                    array_row_start = maps[index_].address[0]
                    array_id = maps[index_].device.split(':')[-1]

                    array_col_start = maps[index_].address[1]
                    array_col_num = maps[index_].address[3]
                    adc_select = self.gen_adc_index(array_col_start,array_col_num)

                    func_mode = cross_index_layer[name]
                    bias_addr_offset = layer_bias_addr_offset[name]
                    relu = 0
                    if name in mvm_relu_layer:
                        relu = 1
                    bias_en = 0
                    if layer.op.bias == True:
                        bias_en = 1
                    in_channel = layer.op.in_channel
                    out_channel = layer.op.out_channel
                    module.append(FC(op_id=op_id, target_id=target_id, array_id=array_id, adc_select=adc_select,
                                    array_row_start=array_row_start, relu=relu, bias_addr_offset=bias_addr_offset,
                                    bias_en=bias_en,out_feature_map_row = out_channel, in_feature_map_row=in_channel,
                                    func_mode=func_mode))

                elif layer.op.op_id in ['maxpool2d']:
                    op_id = ordered_layer[name]
                    target_id = next_layer_id[name]
                    func_mode = cross_index_layer[name]
                    kernel_size = layer.op.kernel
                    stride = layer.op.stride
                    padding = layer.op.padding
                    ifm_row = layer.inputs[0].height
                    ifm_col = layer.inputs[0].width
                    ofm_row = layer.outputs[0].height
                    ofm_col = layer.outputs[0].width
                    in_channel = layer.inputs[0].channel
                    out_channel = layer.outputs[0].channel
                    module.append(Pool(op_id=op_id, target_id=target_id, array_id=array_id,
                                    array_row_start=array_row_start, stride=stride, padding=padding,
                                    out_channel = out_channel, in_channel=in_channel , kernel_size=kernel_size,
                                    out_feature_map_col=ofm_col, out_feature_map_row = ofm_row,
                                    in_feature_map_col=ifm_col, in_feature_map_row=ifm_row,
                                    func_mode=func_mode))

        with open(output_file,'w',encoding="utf8") as f:
            f.write("/* **************************************** */\n")
            f.write("/* This file is automatically generated !!! */\n")
            f.write("/*        Please do not modify it !!!      */\n")
            f.write("/* **************************************** */\n")
            for m in module:
                f.write(m.gen_code())
                f.write('\n')

    def gen_adc_index(self,col_start_index,col_num,adc_num=64):
        '''
        '''
        if col_num > adc_num:
        adc_index = [str(0) for i in range(adc_num)]
        for i in range(col_num):
            index = (col_start_index + i) % adc_num
            adc_index[index] = str(1)
        binary_str = ''.join(adc_index)
        number = int(binary_str, 2)
        hex_str = format(number, 'X')
        if len(hex_str) < 16:
            list_hex = list(hex_str)
            pad_zero = 16 - len(hex_str)
            for i in range(pad_zero):
                list_hex.insert(0,'0')
            hex_str = ''.join(list_hex)
        return hex_str

    def get_pre_layer(self,layers):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: pre_layer_name}
        '''
        prefix_layer = {}
        for name,layer in layers.items():
            if layer.type not in ['input']:
                prefix_layer[name] =  []
                for i in layer.inputs:
                    if ':' in i.ref:
                        temp = i.ref.split(':')[0]
                        prefix_layer[name].append(temp)
                    else:
                        prefix_layer[name].append(i.ref)
        return prefix_layer

    def get_next_layer(self,layers):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_name}
        '''
        next_layer = {}
        pre_layer = self.get_pre_layer(layers)
        for k,v in pre_layer.items():
            for name in v:
                next_layer[name] = k
        return next_layer

    def get_mvm_relu_layer(self,layers):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_name}
        '''
        mvm_relu_layer = []
        pre_layer = self.get_pre_layer(layers)
        for k,v in pre_layer.items():
            if layers[k].type == 'op' and layers[k].op.op_id in ['relu']:
                for i in v:
                    if layers[i].op.op_id in ['conv2d','matmul','linear','fc'] :
                        mvm_relu_layer.append(i)
        return mvm_relu_layer

    def get_bias_addr_offset(self,layers):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: bias_offset_addr}
        '''
        bias_addr_offset = {}
        for name,layer in layers.items():
            if layer.type == 'op' and layer.op.op_id in ['conv2d','matmul','linear','fc']:
                if list(bias_addr_offset.keys()) != []:
                    last_name = list(bias_addr_offset.keys())[-1]
                    bias_addr_offset[name] =  bias_addr_offset[last_name] + layers[last_name].op.out_channel
                else:
                    bias_addr_offset[name] = 0
        return bias_addr_offset

    def get_ordered_layer(self,layers):
        ordered_layer = {}
        cross_layer = {}
        index = 0
        cross_index = ['e','b']
        for name,layer in layers.items():
            if layer.type == 'op' and layer.op.op_id in ['conv2d','matmul','linear','fc','avgpool2d','maxpool2d']:
                ordered_layer[name] = index
                index += 1
                cross_layer[name] = cross_index[index % 2]

        return ordered_layer,cross_layer

    def get_next_layer_id(self,layers):
        '''
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_id}
        '''
        ordered_layer, _ = self.get_ordered_layer(layers)
        next_layers = self.get_next_layer(layers)
        next_layer_id = {}
        next_id = 0
        for name,layer in layers.items():
            if layer.type == 'op' and layer.op.op_id in ['conv2d','matmul','linear','fc','avgpool2d','maxpool2d']:
                next_layer_name = next_layers[name]
                next_layer = layers[next_layer_name]
                while (next_layer.type == 'op'):
                    if next_layer.op.op_id in ['conv2d','matmul','linear','fc','avgpool2d','maxpool2d']:
                        next_id = ordered_layer[next_layer_name]
                        break
                    else:
                        next_layer_name = next_layers[next_layer_name]
                        next_layer = layers[next_layer_name]
                if next_layer.type == 'output':
                    next_id = 0
                next_layer_id[name] = next_id
        return next_layer_id
