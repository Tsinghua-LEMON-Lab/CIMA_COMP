from irtool.core.ir import BaseIR, load_ir
from irtool.core.type_util import to_obj_dict
from irtool.core.datadef import DataDef

from irtool.tools import flatten_layers  # noqa
import numpy as np
import json
import math
import copy
import warnings

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

class CodeGen:

    def __init__(self, ir, in_device='HOSTI', out_device='HOSTI'):
        self.ir = ir
        self.in_device = in_device
        self.out_device = out_device

    def run(self, output_file=None, log_level = 200, trace_enable = False,
            perf_single_trace = False, log_action = 4, run_time=1260000, batch=1,
            specify_input_layer = None, specify_output_layer = None,
            dump_specify_ir_file = None, global_bitwise = 4, Dram_Read_Latency = None,
            Dram_Write_Latency = None, cfg_rram_calc_delay = None):

        self.output_file = output_file
        self.layers = None
        if isinstance(self.ir, BaseIR):
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
        elif isinstance(self.ir, str):
            self.ir = load_ir(self.ir)
            self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))
        else:
            raise ValueError(f"Message translated to English.")
        self.layers = self.ir.flatten_layers()

        first_device_key = list(self.ir.devices.keys())[0]
        MESH_HEIGHT = self.ir.devices[first_device_key].height
        MESH_WIDTH = self.ir.devices[first_device_key].width

        output_file = 'task_cfg.json'
        if self.output_file != None:
            output_file = self.output_file

        assert (self.layers != None)
        pre_layer = self.get_pre_layer(self.layers)

        next_layer = self.get_next_layer(self.layers)

        full_layers_info = copy.deepcopy(self.layers)

        if specify_input_layer != None or specify_output_layer != None:

            if specify_input_layer != None:
                assert specify_input_layer in pre_layer.keys()
                self.remove_layers(pre_layer, specify_input_layer)

            if specify_output_layer != None:
                assert specify_output_layer in next_layer.keys()
                self.remove_layers(next_layer, specify_output_layer)

            graph_inputs = []
            c = 0
            for k, v in self.layers.items():
                if 'graph_input' not in k and 'graph_output' not in k:
                    layer_info = v
                    for i in layer_info.inputs:
                        update = False
                        ref_name = i.ref
                        if ":" in ref_name:
                            ref_name = ref_name.split(':')[0]
                        if ref_name not in self.layers.keys():
                            i.ref = f'graph_input:{c}'
                            c += 1
                            graph_inputs.append(dict(channel= i.channel, width=i.width, height=i.height, channel_last= True))
                    self.layers[k] = layer_info

            graph_outputs = []
            c = 0
            for k, v in self.layers.items():
                if 'graph_input' not in k and 'graph_output' not in k:
                    # layer_info = v
                    possible_next_layers = next_layer[k]
                    for i in possible_next_layers:
                        if i not in self.layers.keys():
                            possible_next_layer_info = full_layers_info[i]
                            for in_ in possible_next_layer_info.inputs:
                                ref_name = in_.ref
                                if ':' in ref_name:
                                    ref_name = ref_name.split(':')[0]
                                if ref_name in self.layers.keys():
                                    dict_ = dict(ref = in_.ref, channel = in_.channel, width = in_.width, height= in_.height)
                                    if dict_ not in graph_outputs:
                                        graph_outputs.append(dict_)
            if graph_inputs != []:
                self.layers['graph_input'].inputs = graph_inputs

            if graph_outputs != []:
                self.layers['graph_output'].inputs = to_obj_dict(graph_outputs, DataDef)

            self.ir.layers = self.layers

            if dump_specify_ir_file != None:
                self.ir.dump_json(file=dump_specify_ir_file)

            pre_layer = self.get_pre_layer(self.layers)

            next_layer = self.get_next_layer(self.layers)

        layer_core_id, utilized_core = self.get_core_id(self.layers, MESH_WIDTH)

        ddr_core = []
        if 'DDRI' in utilized_core:
            ddr_core = utilized_core['DDRI']
            utilized_core_copy = copy.deepcopy(utilized_core)
            utilized_core_copy.pop('DDRI')
        else:
            utilized_core_copy = utilized_core
        sorted_utilized_coredict = dict(sorted(utilized_core_copy.items(), key=lambda x: (int(x[0][4:5]), int(x[0][6:7]))))

        config = {}
        # "key": "value",
        # // 500:Debug 300:Medium
        # "LOG_LEVEL":  200,
        # // DISPLAY or LOG
        # // BITMASK
        # // SC_LOG          = 0x0004, // add report to report log
        # // SC_DISPLAY      = 0x0008, // display report to screen
        # // SC_DEFAULT_INFO_ACTIONS    = SC_LOG | SC_DISPLAY,
        # "trace_enable": false,
        # "LOG_ACTION": 4,
        # "RUN_TIME": 1260000, // NS
        # config['Key'] = 'value'
        # config['Log_Level'] = log_level
        # config['Trace_Enable'] = trace_enable
        # config['Perf_Single_Trace'] = perf_single_trace
        # config['Log_Action'] = log_action
        config['Run_Time'] = run_time
        # config['Global_Batch'] = batch
        # config['Global_Bitwise'] = global_bitwise
        #     config['Dram_Read_Latency'] = Dram_Read_Latency
        #     config['Dram_Write_Latency'] = Dram_Write_Latency
        #     config['cfg_rram_calc_delay'] = cfg_rram_calc_delay
        # "vld":true,
        # "DMEM_Base":0,
        # "Conv_struct":
        # {
        #     "fin_width":1,
        #     "fin_height":1,
        #     "fout_width":32,
        #     "fout_height":32,
        #     "cin":10,
        #     "activation_type":0,
        #     "Batch":1,
        #     "cout":3,
        #     "k_size":3,
        #     "padding":1,
        #     "stride":1,
        #     "rela_pe":0  // 0:W 1:N 2:E 3:S
        # },
        # "src":
        # {
        #     "Src_0":{
        #         "core":"Core5_1",
        #         "tid":0
        #     }
        # }
        #
        next_seg_mcast_layer = self.get_mcast_next_layer_list(self.layers)

        # input layer
        input_layers = self.layers['graph_input']

        count_device = {}
        HOST_device_addr = {}
        inout_device_set = ['HOSTI', 'DDRI']
        assert self.in_device in inout_device_set
        assert self.out_device in inout_device_set

        for d in inout_device_set:
            count_device[d] = 0
            HOST_device_addr[d] = 0

        dmem_start = 0
        # input layer

        for i in input_layers.inputs:

            input_config = {}

            next_layer_name = next_layer[f'graph_input:{count_device[self.in_device]}'][0]

            out_ref_core_id = layer_core_id[next_layer_name]

            ifm_height = i.height
            ifm_width = i.width
            ifm_channel = i.channel
            #     ifm_channel += 1
            input_config['Task_Name'] = 'Input'
            input_config['Vld'] = True
            input_config['Dram_Base'] = HOST_device_addr[self.in_device]
            input_config['Dram_Size'] = math.ceil(ifm_height * ifm_width * ifm_channel / 32)
            HOST_device_addr[self.in_device] += input_config['Dram_Size']

            if count_device[self.in_device] == 0:
                input_config['Dmem_Base'] = 0
            else:
                input_config['Dmem_Base'] = dmem_start

            # data_type
            data_type = self.layers[next_layer_name].CIMA_calc_info.data_type
            next_layer_dmem_size = self.layers[next_layer_name].CIMA_mapping_info.in_line_buffer_addr[0][1]
            next_layer_width = self.layers[next_layer_name].inputs[0].width
            next_layer_channel = self.layers[next_layer_name].inputs[0].channel
            if isinstance(next_layer_dmem_size, str):
                input_config['Dmem_Size'] = math.ceil(int(next_layer_dmem_size, 16) / 32)
            elif isinstance(next_layer_dmem_size, int):
                input_config['Dmem_Size'] = next_layer_dmem_size
            input_config['Dmem_Size'] *= 2
            input_config['Dmem_Size'] = math.ceil(input_config['Dmem_Size'] / 16) * 16
            assert data_type in ['4bit', '8bit']
            # element_size = 4 if data_type == '4bit' else 8
            # single_row_flit = int(math.ceil(ifm_width * ifm_channel * element_size / 256))
            if data_type == '4bit':
                input_config['Flit_Num'] = math.ceil((ifm_height * ifm_width * ifm_channel) / 64)
            elif data_type == '8bit':
                input_config['Flit_Num'] = math.ceil((ifm_height * ifm_width * ifm_channel) / 32)

            dmem_start += input_config['Dmem_Size']

            input_config['Credit_Pix_Len'] = ifm_width

            conv_struct = {}
            conv_struct['fout_width'] = ifm_width
            conv_struct['fout_height'] = ifm_height
            conv_struct['cout'] = ifm_channel
            conv_struct['batch'] = batch
            input_config['Conv_Struct'] = conv_struct
            input_config['Op_Code'] = 'WriteIn'
            if self.in_device== 'DDRI':
                output_config['Op_Code'] = 'WriteOut'

            input_config['Data_Type'] = data_type

            # # in src
            # src_info = {}
            # src_info['src_0'] = {}
            # src_info['src_0']['core'] = self.in_device
            # src_info['src_0']['tid'] = 0
            # input_config['Src'] = src_info

            # in dst
            dst_info = {}
            dst_info['seg_0'] = {}
            dst_info['seg_0']['core'] = out_ref_core_id
            dst_info['seg_0']['tid'] = utilized_core[out_ref_core_id].index(next_layer_name)
            input_config['Dst'] = dst_info


            config[self.in_device] = {}
            config[self.in_device][f'Thread_{count_device[self.in_device]}'] = input_config

            count_device[self.in_device] += 1

        # config['HOSIT']['Thread_0'] = input_config

        # output layer
        output_layers = self.layers['graph_output']
        output_thread_base_tid = {}

        src_count = 0
        for o in output_layers.inputs:

            out_layer_name = o.ref
            out_ref_core_id = layer_core_id[out_layer_name]
            # out_layer_info = self.layers[out_layer_name]

            ofm_height = o.height
            ofm_width = o.width
            ofm_channel = o.channel
            if ofm_channel == 76:
                ofm_channel = 128
            output_config = {}
            output_config['Task_Name'] = 'Output'
            output_config['Vld'] = True
            output_config['Dram_Base'] = HOST_device_addr[self.out_device]
            output_config['Dram_Size'] = math.ceil(ofm_height * ofm_width * ofm_channel / 32)

            out_layer_dmem_size = self.layers[out_layer_name].CIMA_mapping_info.in_line_buffer_addr[0][1]
            data_type = self.layers[out_layer_name].CIMA_calc_info.data_type
            assert data_type in ['4bit', '8bit']
            if isinstance(out_layer_dmem_size, str):
                output_config['Dmem_Size'] = math.ceil(int(out_layer_dmem_size, 16) / 32)
            elif isinstance(out_layer_dmem_size, int):
                output_config['Dmem_Size'] = out_layer_dmem_size
            output_config['Dmem_Size'] *= 2
            output_config['Dmem_Size'] = math.ceil(output_config['Dmem_Size'] / 16) * 16
            HOST_device_addr[self.out_device] += output_config['Dram_Size']

            if count_device[self.out_device] == 0:
                output_config['Dmem_Base'] = 0
                dmem_start = 0
            else:
                output_config['Dmem_Base'] = dmem_start

            # dmem_start += 1024
            # output_config['Dmem_Size'] = 1024
            output_config['Credit_Pix_Len'] = ofm_width

            dmem_start += output_config['Dmem_Size']

            conv_struct = {}
            conv_struct['fin_width'] = ofm_width
            conv_struct['fin_height'] = ofm_height
            conv_struct['cin'] = ofm_channel
            conv_struct['batch'] = batch
            output_config['Conv_Struct'] = conv_struct
            output_config['Op_Code'] = 'WriteOut'
            if self.out_device == 'DDRI':
                output_config['Op_Code'] = 'WriteIn'

            # data_type
            data_type = self.layers[out_layer_name].CIMA_calc_info.data_type
            assert data_type in ['4bit', '8bit']
            output_config['Data_Type'] = data_type
            output_config['Flit_Num'] = output_config['Dmem_Size']

            # out src
            src_info = {}
            src_info['src_0'] = {}
            src_info['src_0']['core'] = out_ref_core_id
            src_info['src_0']['tid'] = utilized_core[out_ref_core_id].index(out_layer_name)

            ref_name = out_layer_name
            if ':' in ref_name:
                seg_id = int(ref_name.split(':')[-1])
            else:
                seg_id = 0
            mcast_id = 0
            if 'graph_input' not in out_layer_name:
                src_dst_layer = next_layer[out_layer_name]
                if len(self.layers[out_layer_name].outputs) == 1:
                    mcast_id = src_dst_layer.index('graph_output')
                elif len(self.layers[out_layer_name].outputs) != len(src_dst_layer):
                    mcast_list = next_seg_mcast_layer[out_layer_name][seg_id]
                    mcast_id = mcast_list.index(layer)

            src_info['src_0']['seg_id'] = seg_id
            src_info['src_0']['mcast_id'] = mcast_id

            output_thread_base_tid[out_layer_name] = count_device[self.out_device]

            output_config['Src'] = src_info

            # out dst
            # dst_info = {}
            # dst_info['seg_0'] = {}
            # dst_info['seg_0']['core'] = self.out_device
            # dst_info['seg_0']['tid'] = 0
            # output_config['Dst'] = dst_info

            # count_device[self.in_device] += 1

            if self.out_device not in config.keys():
                config[self.out_device] = {}
            config[self.out_device][f'Thread_{count_device[self.out_device]}'] = output_config

            #
            count_device[self.out_device] += 1

        # DDR thread
        config['DDRI'] = {}

        if ddr_core != []:

            task = {}
            task_id = 0
            dram_addr = 0

            for layer in ddr_core:

                # content
                task_content = {}
                # task name
                task_content['Task_Name'] = layer
                task_content['Vld'] = True
                # mapping info
                layer_info = self.layers[layer]
                mapping_info = layer_info.CIMA_mapping_info
                index_ = (0,0,0)
                addr_info = mapping_info.mappings[index_]
                xb_addr = addr_info.device.split('.')

                ifm_row = layer_info.inputs[0].height
                ifm_col = layer_info.inputs[0].width
                in_channel = layer_info.inputs[0].channel
                out_channel = layer_info.outputs[0].channel

                # data type
                data_type = layer_info.CIMA_calc_info.data_type
                assert data_type in ['8bit', '4bit']

                element_size = 4 if data_type == '4bit' else 8
                task_content['Dram_Base'] = dram_addr
                if data_type == '8bit':
                    dram_size = math.ceil((layer_info.outputs[0].height * layer_info.outputs[0].width * out_channel) / 32)
                elif  data_type == '4bit':
                    dram_size = math.ceil((layer_info.outputs[0].height * layer_info.outputs[0].width * out_channel) / 64)
                task_content['Dram_Size'] = dram_size

                dram_addr += dram_size
                if dram_addr > (0x80000000 // 32):
                    warnings.warn(f'Message translated to English.')

                if isinstance(mapping_info.in_line_buffer_addr[0][0], str):
                    task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32
                elif isinstance(mapping_info.in_line_buffer_addr[0][0], int):
                    task_content['Dmem_Base'] = mapping_info.in_line_buffer_addr[0][0] // 32
                # task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32

                # task_content['Dmem_Size'] = int(mapping_info.in_line_buffer_addr[0][1], 16) // 2 // 32
                if isinstance(mapping_info.in_line_buffer_addr[0][1], str):
                    task_content['Dmem_Size'] = math.ceil(int(mapping_info.in_line_buffer_addr[0][1], 16) / 64)
                elif isinstance(mapping_info.in_line_buffer_addr[0][1], int):
                    task_content['Dmem_Size'] = math.ceil(mapping_info.in_line_buffer_addr[0][1] // 32 // 2)

                # task_content['Dmem_Size'] = int(mapping_info.in_line_buffer_addr[0][1], 16) // 32
                task_content['Credit_Pix_Len'] = mapping_info.credit_len[0]

                kernel_size = 1
                stride = 1
                padding = 0
                activate = 0

                conv_struct = {}
                conv_struct['fin_width'] = ifm_row
                conv_struct['fin_height'] = ifm_col
                conv_struct['cin'] = in_channel
                conv_struct['batch'] = batch

                # single row flit
                element_size = 4 if data_type == '4bit' else 8

                if layer_info.op.op_id in ['resize']:
                    conv_struct['scale_factor'] = scale_factor

                task_content['Conv_Struct'] = conv_struct
                #
                task_content['Data_Type'] = data_type

                # "op_code":"PEConv",
                task_content['Op_Code'] = "WriteIn"
                # "src":
                # {
                #     "Src_0":{
                #         "core":"HOSTI",
                #         "tid":0
                #     }
                # },
                src = {}
                src_layer = pre_layer[layer]
                src_count = 0
                for l in src_layer:

                    ref_name = self.layers[layer].inputs[src_count].ref
                    if ':' in ref_name:
                        seg_id = int(ref_name.split(':')[-1])
                    else:
                        seg_id = 0
                    mcast_id = 0
                    if 'graph_input' not in l:
                        if len(self.layers[l].outputs) == 1:
                            src_dst_layer = next_layer[l]
                            mcast_id = src_dst_layer.index(layer)

                    src_info = {}
                    if 'graph_input' in l:
                        src_info['core'] = 'HOSTI'
                        src_info['tid'] = 0
                    else:

                        src_info['core'] = layer_core_id[l]
                        src_info['tid'] = int(utilized_core[layer_core_id[l]].index(l))
                        if layer_info.op.op_id in ['add', 'fused_add']:
                            current_layer_inputs = []
                            for n in layer_info.inputs:
                                current_layer_inputs.append(n.ref)
                            src_info['rdc_pe_id'] = int(current_layer_inputs.index(l))
                        elif layer_info.op.op_id in ['concat', 'fused_concat']:
                            current_layer_inputs = []
                            for n in layer_info.inputs:
                                current_layer_inputs.append(n.ref)
                            src_info['da_pe_id'] = int(current_layer_inputs.index(l))

                    src_info['seg_id'] = seg_id
                    src_info['mcast_id'] = mcast_id

                    src[f'src_{src_count}'] = {}
                    src[f'src_{src_count}'].update(src_info)
                    src_count += 1
                task_content['Src'] = src

                task[f'Thread_{task_id}'] = task_content.copy()
                task_id += 1

                task_content.pop('Src')
                task_content['Op_Code'] = "WriteOut"

                # task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32 + int(mapping_info.in_line_buffer_addr[0][1], 16) // 2 // 32
                # task_content['Dmem_Size'] = int(mapping_info.in_line_buffer_addr[0][1], 16) // 32
                if isinstance(mapping_info.in_line_buffer_addr[0][0], str):
                    task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32 + int(mapping_info.in_line_buffer_addr[0][1], 16) // 2 // 32
                elif isinstance(mapping_info.in_line_buffer_addr[0][0], int):
                    task_content['Dmem_Base'] = mapping_info.in_line_buffer_addr[0][0] + mapping_info.in_line_buffer_addr[0][1] // 2
                # DMEM size
                if isinstance(mapping_info.in_line_buffer_addr[0][1], str):
                    task_content['Dmem_Size'] = math.ceil(int(mapping_info.in_line_buffer_addr[0][1], 16) / 64)
                elif isinstance(mapping_info.in_line_buffer_addr[0][1], int):
                    task_content['Dmem_Size'] = math.ceil(mapping_info.in_line_buffer_addr[0][1] / 64)

                # "dst":
                # {
                #     "Seg_0":{
                #         "core":"Core0_1",
                #         "tid":0
                #     }
                # }

                dst = {}
                dst_layer = next_layer[layer]
                dst_count = 0

                if layer_info.op.op_id in ['split'] or (layer_info.op.op_id in ['fused_add', 'fused_concat'] and layer_info.op.split != None):
                    mcast_count = {}
                    for l in dst_layer:
                        seg_count = self.layers[l].inputs[0].ref.split(':')[-1]
                        if seg_count not in mcast_count.keys():
                            mcast_count[seg_count] = 0
                        else:
                            mcast_count[seg_count] += 1

                        dst_info = {}
                        if 'graph_output' in l:
                            # dst_info['core'] = 'HOSTI'
                            # dst_info['tid'] = 0
                            dst_info['core'] = self.out_device
                            dst_info['tid'] = output_thread_base_tid[layer]

                        else:
                            dst_info['core'] = layer_core_id[l]
                            dst_info['tid'] = int(utilized_core[layer_core_id[l]].index(l))
                            if layer_info.op.op_id in ['fused_add', 'fused_concat']:

                                next_layer_info = self.layers[l]
                                if next_layer_info.op.op_id in ['add', 'fused_add']:
                                    next_layer_inputs = []
                                    for n in next_layer_info.inputs:
                                        next_layer_inputs.append(n.ref)
                                    dst_info['rdc_pe_id'] = int(next_layer_inputs.index(layer))
                                elif next_layer_info.op.op_id in ['concat', 'fused_concat']:
                                    next_layer_inputs = []
                                    for n in next_layer_info.inputs:
                                        next_layer_inputs.append(n.ref)
                                    dst_info['da_pe_id'] = int(next_layer_inputs.index(layer))
                        if f'seg_{seg_count}' not in dst.keys():
                            dst[f'seg_{seg_count}'] = {}
                        dst[f'seg_{seg_count}'].update({f'mcast_{mcast_count[seg_count]}': dst_info})
                        # dst_count += 1
                else:
                    dst_info = {}
                    dst['seg_0'] = {}
                    for l in dst_layer:
                        if 'graph_output' in l:
                            # dst_info['core'] = 'HOSTI'
                            # dst_info['tid'] = 0
                            dst_info['core'] = self.out_device
                            dst_info['tid'] = output_thread_base_tid[layer]
                        else:
                            dst_info['core'] = layer_core_id[l]
                            dst_info['tid'] = int(utilized_core[layer_core_id[l]].index(l))
                            next_layer_info = self.layers[l]
                            if next_layer_info.op.op_id in ['add', 'fused_add']:
                                next_layer_inputs = []
                                for n in next_layer_info.inputs:
                                    next_layer_inputs.append(n.ref)
                                dst_info['rdc_pe_id'] = int(next_layer_inputs.index(layer))
                            elif next_layer_info.op.op_id in ['concat', 'fused_concat']:
                                next_layer_inputs = []
                                for n in next_layer_info.inputs:
                                    next_layer_inputs.append(n.ref)
                                dst_info['da_pe_id'] = int(next_layer_inputs.index(layer))

                        if len(dst_layer) == 1:
                            dst['seg_0'].update(dst_info)
                        else:
                            dst['seg_0'][f'mcast_{dst_count}'] = {}
                            dst['seg_0'][f'mcast_{dst_count}'].update(dst_info)
                        dst_count += 1

                task_content['Dst'] = dst
                # task
                task[f'Thread_{task_id}'] = task_content
                task_id += 1

            config['DDRI'] = task

        if config["DDRI"] == {}:
            config.pop("DDRI")

        # op layer

        for core, layers in sorted_utilized_coredict.items():
            task = {}
            task_id = 0
            PE_tid = [-1, -1, -1, -1]
            for layer in layers:
                # content
                task_content = {}
                # task name
                task_content['Task_Name'] = layer
                task_content['Vld'] = True
                # mapping info
                layer_info = self.layers[layer]
                mapping_info = layer_info.CIMA_mapping_info
                index_ = (0,0,0)
                addr_info = mapping_info.mappings[index_]
                xb_addr = addr_info.device.split('.')

                data_type = layer_info.CIMA_calc_info.data_type
                assert data_type in ['8bit', '4bit']
                element_size = 4 if data_type == '4bit' else 8
                # "DMEM_Base":0
                # task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32
                # task_content['Dmem_Size'] = int(mapping_info.in_line_buffer_addr[0][1], 16) // 32
                if isinstance(mapping_info.in_line_buffer_addr[0][0], str):
                    task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32
                elif isinstance(mapping_info.in_line_buffer_addr[0][0], int):
                    task_content['Dmem_Base'] = mapping_info.in_line_buffer_addr[0][0]
                # task_content['Dmem_Base'] = int(mapping_info.in_line_buffer_addr[0][0], 16) // 32

                # task_content['Dmem_Size'] = int(mapping_info.in_line_buffer_addr[0][1], 16)   // 32
                if isinstance(mapping_info.in_line_buffer_addr[0][1], str):
                    task_content['Dmem_Size'] = math.ceil(int(mapping_info.in_line_buffer_addr[0][1], 16) / 32)
                elif isinstance(mapping_info.in_line_buffer_addr[0][1], int):
                    task_content['Dmem_Size'] = mapping_info.in_line_buffer_addr[0][1]

                task_content['Credit_Pix_Len'] = mapping_info.credit_len[0]

                direction = 0
                lpe = [0]

                # data type
                data_type = layer_info.CIMA_calc_info.data_type
                assert data_type in ['8bit', '4bit']

                dmac_sn_1 = layer_info.CIMA_calc_info.accumulate_shift_num
                dmac_sn_0 = layer_info.CIMA_calc_info.scale_shift_num
                pe_sn = layer_info.CIMA_calc_info.accumulate_shift_num
                ADC_range = 0

                if layer_info.op.op_id in ['conv2d','matmul','fc','linear','fused_conv2d','fused_fc']:

                    if layer_info.op.op_id in ['conv2d', 'fused_conv2d']:
                        kernel_size = layer_info.op.kernel
                        stride = layer_info.op.stride
                        padding = layer_info.op.padding
                        ifm_row = layer_info.inputs[0].height
                        ifm_col = layer_info.inputs[0].width
                        ADC_range = layer_info.CIMA_calc_info.ADC_quant_level
                    else:
                        kernel_size = 1
                        stride = 1
                        padding = 0
                        ifm_row = 1
                        ifm_col = 1

                        pl = pre_layer[layer][0]
                        if ':' in pl:
                            pl = pl.split(':')[0]
                        if 'graph_input' in pl:
                            pl_ifm_row = self.layers[pl].inputs[0].height
                            pl_ifm_col = self.layers[pl].inputs[0].width
                            pl_channel = self.layers[pl].inputs[0].channel
                        else:
                            pl_ifm_row = self.layers[pl].outputs[0].height
                            pl_ifm_col = self.layers[pl].outputs[0].width
                            pl_channel = self.layers[pl].outputs[0].channel
                        #     ifm_row = self.layers[pl].outputs[0].height
                        #     ifm_col = self.layers[pl].outputs[0].width
                        #     kernel_size = ifm_row
                        #     current_in_channel = self.layers[pl].outputs[0].channel
                        # else:
                        #     ifm_row = self.layers[pl].inputs[0].height
                        #     ifm_col = self.layers[pl].inputs[0].width
                        #     kernel_size = ifm_row
                        #     current_in_channel = self.layers[pl].inputs[0].channel

                        #
                        current_in_channel = pl_ifm_row * pl_ifm_col * pl_channel



                    if layer_info.op.op_id in ['conv2d', 'fused_conv2d']:
                        # current_in_channel = int(addr_info.address[2] / kernel_size **(2))
                        current_in_channel = layer_info.inputs[0].channel

                    # current_out_channel = addr_info.address[3]
                    current_out_channel = layer_info.outputs[0].channel

                    activate = 0
                    if layer_info.op.op_id in ['fused_conv2d', 'fused_fc']:
                        if layer_info.op.relu != None:
                            activate = 1
                        elif layer_info.op.silu != None:
                            activate = 2

                    in_channel = current_in_channel
                    out_channel = current_out_channel


                    op_code = 'PEConv'


                    if 'dmac' in addr_info.device:
                        # Digital MAC
                        op_code = 'DMACConv'

                    else:
                        # RRAM xb
                        direction = int(xb_addr[2].split(':')[1]) # N:0,  E:1, S:2, W:3 (IR defined)
                        direction = (direction + 1) % 4 # W:0, N:1, E:2, S:3 (SystemC defined)
                        PE_tid[direction] += 1
                        assert PE_tid[direction] < 2, f'Message translated to English.'
                        # logic pe
                        lpe = []
                        # pe_num = math.ceil(current_in_channel * kernel_size **(2)  / 576) * math.ceil(current_out_channel / 128)
                        #     lpe.append(l)
                        pe_num = xb_addr[-1].split(':')[-1]
                        if '-' in pe_num:
                            num_list = pe_num.split('-')
                            assert int(num_list[1]) + 1 <= 16
                            assert int(num_list[0]) >= 0
                            for pn in range(int(num_list[0]), int(num_list[1])+1):
                                lpe.append(pn)
                        else:
                            pe_num = int(pe_num)
                            lpe.append(pe_num)

                elif layer_info.op.op_id in ['add', 'fused_add']:

                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    if layer_info.op.op_id in ['add']:
                        assert (in_channel == out_channel)

                    activate = 0
                    if layer_info.op.op_id in ['fused_add']:
                        if layer_info.op.relu != None:
                            activate = 1
                        elif layer_info.op.silu != None:
                            activate = 2

                    kernel_size = 1
                    stride = 1
                    padding = 0

                    op_code = 'Transfer'

                elif layer_info.op.op_id == 'avg':
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel
                    assert (in_channel == out_channel)

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 0

                    op_code = 'Avg'

                elif layer_info.op.op_id in ['avgpool2d','avg_pool2d','max_pool2d','maxpool2d','global_avg_pool2d']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel
                    if 'global' in layer_info.op.op_id:
                        kernel_size = ifm_row
                        stride = 1
                        padding = 0
                    else:
                        kernel_size = layer_info.op.kernel
                        stride = layer_info.op.stride
                        padding = layer_info.op.padding

                    assert (in_channel == out_channel)

                    # kernel_size = 1
                    # stride = 1
                    # padding = 0
                    activate = 0
                    if layer_info.op.op_id in ['max_pool2d','maxpool2d']:
                        op_code = 'MaxPooling'
                    else:
                        op_code = 'AvgPooling'

                elif layer_info.op.op_id in ['concat', 'fused_concat']:

                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel
                    total_in_channel = 0
                    for in_c in layer_info.inputs:
                        total_in_channel += in_c.channel
                    total_out_channel = 0
                    for out_c in layer_info.outputs:
                        total_out_channel += out_c.channel

                    activate = 0
                    if layer_info.op.op_id in ['fused_concat']:
                        if layer_info.op.relu != None:
                            activate = 1
                        elif layer_info.op.silu != None:
                            activate = 2

                    kernel_size = 1
                    stride = 1
                    padding = 0

                    op_code = 'Transfer'

                elif layer_info.op.op_id in ['flatten','reshape']:
                    # ifm_row = layer_info.inputs[0].height
                    # ifm_col = layer_info.inputs[0].width
                    # in_channel = layer_info.inputs[0].channel
                    # out_channel = layer_info.outputs[0].channel

                    # kernel_size = 1
                    # stride = 1
                    # padding = 0
                    # activate = 0

                    # op_code = 'Transfer'
                    continue

                elif layer_info.op.op_id in ['identity', 'type_conversion']:

                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 0

                    op_code = 'Transfer'
                    if layer_info.op.op_id in ['type_conversion']:
                        in_dtype = layer_info.op.in_dtype
                        out_dtype = layer_info.op.out_dtype

                elif layer_info.op.op_id in ['split']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 0

                    op_code = 'Transfer'

                elif layer_info.op.op_id in ['silu']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 2

                    op_code = 'Transfer'

                elif layer_info.op.op_id in ['relu']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 1

                    op_code = 'Transfer'

                elif layer_info.op.op_id in ['mul_add']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 0

                    mul_add_scale = layer_info.CIMA_calc_info.scale
                    mul_add_shift = layer_info.CIMA_calc_info.scale_shift_num
                    mul_add_offset = layer_info.CIMA_calc_info.offset
                    op_code = 'Transfer'

                elif layer_info.op.op_id in ['resize']:
                    ifm_row = layer_info.inputs[0].height
                    ifm_col = layer_info.inputs[0].width
                    in_channel = layer_info.inputs[0].channel
                    out_channel = layer_info.outputs[0].channel

                    kernel_size = 1
                    stride = 1
                    padding = 0
                    activate = 0

                    scale_factor = layer_info.op.scale[-1]

                    op_code = 'Upsample'

                else:
                    raise ValueError(f'don\'t support {layer_info.op.op_id}!')
                # "Conv_struct":
                # {
                #     "fin_width":32,
                #     "fin_height":32,
                #     "cin":3,
                #     "activation_type":0,
                #     "Batch":1,
                #     "cout":32,
                #     "k_size":3,
                #     "padding":1,
                #     "stride":1,
                #     "rela_pe":2  // 0:W 1:N 2:E 3:S
                # },

                if in_channel == 38:
                    in_channel = 64
                elif in_channel == 76:
                    in_channel = 128

                if out_channel == 38:
                    out_channel = 64
                elif out_channel == 76:
                    out_channel = 128

                conv_struct = {}
                conv_struct['fin_width'] = ifm_col
                conv_struct['fin_height'] = ifm_row
                conv_struct['cin'] = in_channel
                conv_struct['activation_type'] = activate
                conv_struct['batch'] = batch
                conv_struct['cout'] = out_channel
                conv_struct['k_size'] = kernel_size
                conv_struct['padding'] = padding
                conv_struct['stride'] = stride
                conv_struct['rela_pe'] = direction
                conv_struct['lpe'] = lpe
                conv_struct['pe_tid'] = PE_tid[direction]

                if layer_info.op.op_id in ['resize']:
                    conv_struct['scale_factor'] = scale_factor

                if layer_info.op.op_id in ['type_conversion']:
                    conv_struct['in_dtype'] = in_dtype
                    conv_struct['out_dtype'] = out_dtype
                    data_type = in_dtype

                if layer_info.op.op_id in ['mul_add']:
                    conv_struct['qt_mtply'] = mul_add_scale
                    conv_struct['qt_shift'] = mul_add_shift
                    conv_struct['qt_offset'] = mul_add_offset

                task_content['Conv_Struct'] = conv_struct
                #
                task_content['Data_Type'] = data_type
                # "op_code":"PEConv",
                task_content['Op_Code'] = op_code
                task_content['DMAC_sn_1'] = dmac_sn_1
                if isinstance(dmac_sn_0, list):
                    task_content['DMAC_sn_0'] = dmac_sn_0[0]
                elif isinstance(dmac_sn_0, int):
                    task_content['DMAC_sn_0'] = dmac_sn_0
                else:
                    task_content['DMAC_sn_0'] = 0

                task_content['PE_sn'] = pe_sn
                task_content['ADC_range'] = ADC_range
                # "src":
                # {
                #     "Src_0":{
                #         "core":"HOSTI",
                #         "tid":0
                #     }
                # },
                src = {}
                src_layer = pre_layer[layer]
                src_count = 0
                for l in src_layer:

                    ref_name = self.layers[layer].inputs[src_count].ref
                    if ':' in ref_name:
                        seg_id = int(ref_name.split(':')[-1])
                    else:
                        seg_id = 0
                    mcast_id = 0
                    if 'graph_input' not in l:
                        src_dst_layer = next_layer[l]
                        if len(self.layers[l].outputs) == 1:
                            mcast_id = src_dst_layer.index(layer)
                        elif len(self.layers[l].outputs) != len(src_dst_layer):
                            mcast_list = next_seg_mcast_layer[l][seg_id]
                            mcast_id = mcast_list.index(layer)

                    src_info = {}
                    if 'graph_input' in l:
                        src_info['core'] = 'HOSTI'
                        src_info['tid'] = 0
                    else:
                        src_info['core'] = layer_core_id[l]
                        tid = int(utilized_core[layer_core_id[l]].index(l))
                        if layer_core_id[l] == 'DDRI':
                            src_info['tid'] = tid * 2 + 1
                        else:
                            src_info['tid'] = tid
                        if layer_info.op.op_id in ['add', 'fused_add']:
                            current_layer_inputs = []
                            for n in layer_info.inputs:
                                current_layer_inputs.append(n.ref)
                            src_info['rdc_pe_id'] = int(current_layer_inputs.index(l))
                        elif layer_info.op.op_id in ['concat', 'fused_concat']:
                            current_layer_inputs = []
                            for n in layer_info.inputs:
                                current_layer_inputs.append(n.ref)
                            src_info['da_pe_id'] = int(current_layer_inputs.index(l))

                    src_info['seg_id'] = seg_id
                    src_info['mcast_id'] = mcast_id

                    src[f'src_{src_count}'] = {}
                    src[f'src_{src_count}'].update(src_info)
                    src_count += 1
                task_content['Src'] = src

                # "dst":
                # {
                #     "Seg_0":{
                #         "core":"Core0_1",
                #         "tid":0
                #     }
                # }

                dst = {}
                dst_layer = next_layer[layer]
                dst_count = 0

                if layer_info.op.op_id in ['split'] or (layer_info.op.op_id in ['fused_add', 'fused_concat'] and layer_info.op.split != None):
                    mcast_count = {}
                    for l in dst_layer:
                        seg_count = self.layers[l].inputs[0].ref.split(':')[-1]
                        if seg_count not in mcast_count.keys():
                            mcast_count[seg_count] = 0
                        else:
                            mcast_count[seg_count] += 1

                        dst_info = {}
                        if 'graph_output' in l:
                            # dst_info['core'] = 'HOSTI'
                            # dst_info['tid'] = 0
                            dst_info['core'] = self.out_device
                            dst_info['tid'] = output_thread_base_tid[layer]

                        else:
                            dst_info['core'] = layer_core_id[l]
                            dst_info['tid'] = int(utilized_core[layer_core_id[l]].index(l))
                            if layer_info.op.op_id in ['fused_add', 'fused_concat']:

                                next_layer_info = self.layers[l]
                                if next_layer_info.op.op_id in ['add', 'fused_add']:
                                    next_layer_inputs = []
                                    for n in next_layer_info.inputs:
                                        next_layer_inputs.append(n.ref)
                                    dst_info['rdc_pe_id'] = int(next_layer_inputs.index(layer))
                                elif next_layer_info.op.op_id in ['concat', 'fused_concat']:
                                    next_layer_inputs = []
                                    for n in next_layer_info.inputs:
                                        next_layer_inputs.append(n.ref)
                                    dst_info['da_pe_id'] = int(next_layer_inputs.index(layer))
                        if f'seg_{seg_count}' not in dst.keys():
                            dst[f'seg_{seg_count}'] = {}
                        dst[f'seg_{seg_count}'].update({f'mcast_{mcast_count[seg_count]}': dst_info})
                        # dst_count += 1
                else:
                    dst_info = {}
                    dst['seg_0'] = {}
                    for l in dst_layer:
                        if 'graph_output' in l:
                            # dst_info['core'] = 'HOSTI'
                            # dst_info['tid'] = 0
                            dst_info['core'] = self.out_device
                            dst_info['tid'] = output_thread_base_tid[layer]
                        else:
                            dst_info['core'] = layer_core_id[l]
                            tid = int(utilized_core[layer_core_id[l]].index(l))
                            if layer_core_id[l] == 'DDRI':
                                dst_info['tid'] = tid * 2
                            else:
                                dst_info['tid'] = tid
                            # dst_info['tid'] = int(utilized_core[layer_core_id[l]].index(l))
                            next_layer_info = self.layers[l]
                            if next_layer_info.op.op_id in ['add', 'fused_add']:
                                next_layer_inputs = []
                                for n in next_layer_info.inputs:
                                    next_layer_inputs.append(n.ref)
                                dst_info['rdc_pe_id'] = int(next_layer_inputs.index(layer))
                            elif next_layer_info.op.op_id in ['concat', 'fused_concat']:
                                next_layer_inputs = []
                                for n in next_layer_info.inputs:
                                    next_layer_inputs.append(n.ref)
                                dst_info['da_pe_id'] = int(next_layer_inputs.index(layer))

                        if len(dst_layer) == 1:
                            dst['seg_0'].update(dst_info)
                        else:
                            dst['seg_0'][f'mcast_{dst_count}'] = {}
                            dst['seg_0'][f'mcast_{dst_count}'].update(dst_info)
                        dst_count += 1

                task_content['Dst'] = dst
                # task
                task[f'Thread_{task_id}'] = task_content
                task_id += 1

            config[core] = task

        config = self.apply_systemc_patches(config)
        self.make_json(config, output_file)

    def make_json(self, program_dict, output_file):
        json_str = json.dumps(program_dict, cls = MyEncoder, indent=4)
        with open(output_file, 'w') as json_file:
            json_file.write(json_str)

    def apply_systemc_patches(self, config):
        """Apply post-generation patches formerly maintained in test scripts."""
        patched = copy.deepcopy(config)
        self._apply_bn_link_patch(patched)
        self._apply_dmem_patch(patched)
        return patched

    def _apply_bn_link_patch(self, config):
        pe_dir_map = {0: "W", 1: "N", 2: "E", 3: "S"}
        skip_cores = {"Run_Time", "HOSTI", "DDRI"}
        for core, core_cfg in config.items():
            if core in skip_cores or not isinstance(core_cfg, dict):
                continue

            bn_index = {"E": 0, "S": 0, "W": 0, "N": 0}
            for _, t_cfg in core_cfg.items():
                if not isinstance(t_cfg, dict) or t_cfg.get("Op_Code") != "PEConv":
                    continue

                conv_struct = t_cfg.get("Conv_Struct", {})
                pe_dir = pe_dir_map.get(conv_struct.get("rela_pe"))
                lpe = conv_struct.get("lpe", [])
                cout = conv_struct.get("cout")
                if pe_dir is None or not isinstance(lpe, list):
                    continue

                bn_buf = []
                if cout in [64, 128]:
                    bn_buf = [bn_index[pe_dir]]
                    bn_index[pe_dir] += 1
                elif cout == 256:
                    bn_buf = [bn_index[pe_dir], bn_index[pe_dir] + 1]
                    bn_index[pe_dir] += 2
                elif cout == 512:
                    bn_buf = [bn_index[pe_dir], bn_index[pe_dir] + 1, bn_index[pe_dir], bn_index[pe_dir] + 1]
                    bn_index[pe_dir] += 2
                else:
                    continue

                if len(bn_buf) != len(lpe):
                    warnings.warn(
                        f"Skip BN patch for {t_cfg.get('Task_Name', 'unknown')}: "
                        f"bn count {len(bn_buf)} != lpe count {len(lpe)}."
                    )
                    continue

                conv_struct["bn"] = bn_buf

    def _apply_dmem_patch(self, config):
        hosti_cfg = config.get("HOSTI")
        if not isinstance(hosti_cfg, dict):
            return

        dmem_start = 640
        for _, t_info in hosti_cfg.items():
            if not isinstance(t_info, dict):
                continue

            dmem_size = self._to_int(t_info.get("Dmem_Size"), default=0)
            t_info["Dmem_Base"] = dmem_start
            dmem_start += dmem_size

            task_name = t_info.get("Task_Name")
            conv_struct = t_info.get("Conv_Struct", {})
            if not isinstance(conv_struct, dict):
                continue

            if task_name == "Output":
                fin_w = self._to_int(conv_struct.get("fin_width"))
                fin_h = self._to_int(conv_struct.get("fin_height"))
                fin_c = self._to_int(conv_struct.get("cin"))
                if None not in [fin_w, fin_h, fin_c]:
                    t_info["Flit_Num"] = fin_w * fin_h * fin_c // 64
            elif task_name in ["input", "Input"]:
                fout_w = self._to_int(conv_struct.get("fout_width"))
                fout_h = self._to_int(conv_struct.get("fout_height"))
                fout_c = self._to_int(conv_struct.get("cout"))
                if None not in [fout_w, fout_h, fout_c]:
                    t_info["Flit_Num"] = fout_w * fout_h * fout_c // 64

        core_cfg = config.get("Core0_4")
        if not isinstance(core_cfg, dict):
            return

        for _, t_info in core_cfg.items():
            if not isinstance(t_info, dict):
                continue
            dmem_size = self._to_int(t_info.get("Dmem_Size"), default=0)
            t_info["Dmem_Base"] = dmem_start
            dmem_start += dmem_size

    @staticmethod
    def _to_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def remove_layers(self, relative_layers, specify_layer):

        specify_relative_layers = relative_layers[specify_layer]
        for i in specify_relative_layers:
            if 'graph_input' not in i and 'graph_output' not in i:
                if i in self.layers.keys():
                    self.layers.pop(i)
                self.remove_layers(relative_layers, i)

    def get_core_id(self, layers, MESH_WIDTH):
        '''
        input: {layer_name:layer_info}
        output: (layer_core: {layer_name: core_id}, utilized_core: {core_id: [layer_name, ...]})
        '''
        layer_core = {}
        utilized_core = {}

        for name, layer in layers.items():
            if layer.type == 'op':
                if layer.CIMA_mapping_info == None:
                    continue
                maps = layer.CIMA_mapping_info.mappings
                index_ = (0,0,0)
                maps_info = maps[index_]

                device = maps_info.device.split('.')
                if ":" in device[1]:
                    index_ = int(device[1].split(':')[1])
                    current_mesh_height = index_ // MESH_WIDTH
                    current_mesh_width = index_ % MESH_WIDTH

                    core_name = f'Core{current_mesh_height}_{current_mesh_width}'
                    layer_core[name] = core_name
                else:
                    core_name = 'DDRI'
                    layer_core[name] = core_name

                if core_name not in utilized_core:
                    utilized_core[core_name] = []

                utilized_core[core_name].append(name)

        return layer_core, utilized_core

    def get_pre_layer(self,layers):
        ''''
        input: {layer_name:layer_object}
        return: {current_layer_name: pre_layer_name}
        '''
        prefix_layer = {}
        for name,layer in layers.items():
            if layer.type not in ['input']:
                if layer.type == 'op' and layer.op.op_id in ['constant']:
                    continue
                prefix_layer[name] =  []
                for i in layer.inputs:

                    if 'graph_input' not in i.ref:
                        # pre_layer = layers[i.ref]
                        ref = i.ref
                        if ':' in ref:
                            ref = ref.split(':')[0]
                        pre_layer = layers[ref]
                        if pre_layer.type == 'op' and pre_layer.op.op_id in ['flatten', 'reshape']:
                            for j in pre_layer.inputs:
                                prefix_layer[name].append(j.ref)
                        else:
                            prefix_layer[name].append(ref)
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
            if  self.layers[k].type == 'op' and self.layers[k].op.op_id in ['flatten', 'reshape']:
                continue
            for name in v:
                if name not in next_layer.keys():
                    next_layer[name] = []
                next_layer[name].append(k)
        return next_layer


    def get_mcast_next_layer_list(self, layers):
        mcast_layer = {}
        for ln, layer_info in layers.items():
            if layer_info.type == 'op':
                if layer_info.op.op_id in ['constant']:
                    continue
                for in_ in layer_info.inputs:
                    if ':' in in_.ref and 'graph_input' not in in_.ref:
                        ref_layer_name = in_.ref
                        if ref_layer_name not in mcast_layer.keys():
                            mcast_layer[ref_layer_name] = []
                        mcast_layer[ref_layer_name].append(ln)
        reduced_mcast_layer = {}
        for k,v in mcast_layer.items():
            ref_name = k.split(':')[0]
            if ref_name not in reduced_mcast_layer.keys():
                reduced_mcast_layer[ref_name] = []
            reduced_mcast_layer[ref_name].append(v)
        return reduced_mcast_layer


