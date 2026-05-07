
import json
import math
import copy
import numpy as np
import os
import time
import logging

#
from mapper.device.CIMA import * #noqa
from mapper.self_defined_op.cima_op import * #noqa

from mapper.placement.CIMA import get_pre_layer, get_minimal_circle_subgraph_end_layer_v2, \
                                            get_layer_depth_index_forward,  get_next_layer, get_max_length_direct_graph, get_min_length_direct_graph

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

def make_json(program_dict, output_file):
    json_str = json.dumps(program_dict, cls = MyEncoder, indent=4)
    with open(output_file, 'w') as json_file:
        json_file.write(json_str)


class CIMA_DMEM_Optimizer:

    def __init__(self, ir, sc_config, optim_path = None, early_stop_threshold = 5, inner_loop_threshold = 5):
        self.ir = ir
        self.sc_config = sc_config
        self.optim_path = optim_path
        self.early_stop_threshold = early_stop_threshold
        self.inner_loop_threshold = inner_loop_threshold
        self.init_logger()


    def get_optimization_layer(self, layers):
        optimization_layer = []
        single_row_value_dict = {}
        calculate_number = {}
        # current_row_value_dict = {}
        for k,v in layers.items():
            if v.type == 'op' and v.op.op_id in ['add', 'fused_add', 'concat', 'fused_concat']:
                optimization_layer.append(k)
                # single_row
                in_channel_num = v.inputs[0].channel
                fin_width = v.inputs[0].width
                if v.op.op_id in ['concat', 'fused_concat']:
                    in_channel_num = in_channel_num * len(v.inputs)
                single_row_value_dict[k] = int(math.ceil(in_channel_num * fin_width  / 64))
                #
                fin_height = v.inputs[0].height
                calculate_number[k] = fin_width * fin_height
                # current_row_value_dict[k] = v.CIMA_mapping_info.in_line_buffer_addr[0][1] // single_row_value_dict[k]
        return optimization_layer, single_row_value_dict, calculate_number

    def get_layer_core_loc(self, optimization_layer):
        layer_core_loc = {}
        for k,v in self.sc_config.items():
            if isinstance(v, dict):
                for k_v,v_v in v.items():
                    if v_v['Task_Name'] in optimization_layer:
                        layer_core_loc[v_v['Task_Name']] = [k, k_v]
        return layer_core_loc

    def sort_node(self, optimization_layer, calculate_number, pre_layers_dict, layer_depth_index_dict):
        #
        eval_value = {}
        for ol in optimization_layer:
            end_layer_name = get_minimal_circle_subgraph_end_layer_v2(pre_layers_dict, ol)
            max_length_direct_graph = get_max_length_direct_graph(layer_depth_index_dict, ol, end_layer_name)
            min_length_direct_graph = get_min_length_direct_graph(layer_depth_index_dict, ol, end_layer_name)
            #
            distance = len(max_length_direct_graph.keys()) - len(min_length_direct_graph.keys())

            eval_value[ol] = distance * calculate_number[ol]
            # eval_value[ol] = distance
            # eval_value[ol] = calculate_number[ol]

        sorted_key = sorted(eval_value.keys(), key=lambda x: eval_value[x], reverse=True)

        return sorted_key

    def init_logger(self):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        if self.optim_path == None:
            log_dir = os.getcwd()
            if not os.path.exists(log_dir + "/" + f'{time_str}'):
                os.mkdir(log_dir + "/" + f'{time_str}')
        else:
            log_dir = self.optim_path
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        log_file = log_dir + "/" + f'dmem_optimize_{time_str}.log'
        logging.basicConfig(level=logging.DEBUG,
                        filename=log_file,
                        filemode='a',
                        format= '%(asctime)s - %(levelname)s: %(message)s'
                        )
        # cmd output
        console_handler = logging.StreamHandler()
        self.logger = logging.getLogger()
        self.logger.addHandler(console_handler)
        self.logger.info('Log file: ' + str(log_file))

    def run(self, module_name='None'):

        layer_info = self.ir.layers
        layer_depth_index_dict = {}
        # pre layer
        pre_layers_dict  = get_pre_layer(layer_info)
        # next layer
        next_layers_dict = get_next_layer(layer_info)

        pre_layer_count = {}
        for k,v in pre_layers_dict.items():
            sum_ = 0
            for i in v:
                if 'Constant' not in i:
                    sum_ += 1
            pre_layer_count[k] = sum_
            # pre_layer_count[k] = len(v)
        #
        pre_layer_count['graph_input:0'] = 1

        #
        layer_count = {}
        get_layer_depth_index_forward(next_layers_dict, pre_layer_count, #
                                    'graph_input:0', None, 0, layer_count, layer_depth_index_dict)

        # optimization layer
        optimization_layer, single_row_value_dict, calculate_number = self.get_optimization_layer(layer_info)
        # sort optimization layer
        sorted_ol = self.sort_node(optimization_layer, calculate_number, pre_layers_dict, layer_depth_index_dict)
        #
        sc_config = copy.deepcopy(self.sc_config)
        core_loc = self.get_layer_core_loc(sorted_ol)
        #
        latency_history = []
        iterative = 0
        #
        early_stop_count = 0
        while True:
            c = 0
            traversal_index = 0
            for ol in sorted_ol:
                traversal_index += 1
                core_id = core_loc[ol][0]
                thread_id = core_loc[ol][1]
                #
                self.logger.info(f'Target Layer: {ol}, Core ID: {core_id}, Thread ID: {thread_id}')

                dmem_start = 640
                #
                original_dmem = sc_config[core_id][thread_id]['Dmem_Size']
                for k,v in sc_config[core_id].items():
                    if v['Task_Name'] == ol:
                        v['Dmem_Base'] = dmem_start
                        v['Dmem_Size'] += single_row_value_dict[ol]
                        dmem_start += v['Dmem_Size']
                    else:
                        v['Dmem_Base'] = dmem_start
                        dmem_start += v['Dmem_Size']
                #
                if dmem_start > 16384:
                    sc_config[core_id][thread_id]['Dmem_Size'] = original_dmem
                    continue
                #
                make_json(sc_config, self.optim_path + f'{module_name}_optim.json')
                # change cfg
                with open(f'{self.optim_path}\\cfg.json', 'r') as f:
                    cfg = json.load(f)
                cfg['cfg_path'] = self.optim_path + f'{module_name}_optim.json'
                make_json(cfg, self.optim_path + f'cfg.json')
                #
                os.system(f"simu -i {self.optim_path}\\cfg.json -o {self.optim_path}\\systemc_optim_{module_name}")
                IsSimuSuccess = False
                with open(f"{self.optim_path}\\systemc_optim_{module_name}\\" + f'interface.txt', 'r') as f1:
                    rec = f1.readline()
                    while True:
                        if f'latency' in rec:
                            rec = rec.split(':')
                            rec = rec[-1].split(' ')[0]
                            if rec != '0':
                                IsSimuSuccess = True
                            break
                        rec = f1.readline()
                        if not rec:
                            break
                iterative += 1
                if IsSimuSuccess:
                    latency = float(rec)
                    if latency_history == []:
                        latency_history.append(latency)
                        self.logger.info(f"Iteration: #{iterative}, Dmem Size: {sc_config[core_id][thread_id]['Dmem_Size']} flits, Latency: {latency}, Accept !!!")
                    else:
                        if latency < latency_history[-1]:
                            latency_history.append(latency)
                            #
                            c = 0
                            early_stop_count = 0
                            self.logger.info(f"Iteration: #{iterative}, Dmem Size: {sc_config[core_id][thread_id]['Dmem_Size']} flits, Latency: {latency}, Accept !!!")
                        else:
                            self.logger.info(f"Iteration: #{iterative}, Dmem Size: {sc_config[core_id][thread_id]['Dmem_Size']} flits, Latency: {latency}, Reject !!!")
                            sc_config[core_id][thread_id]['Dmem_Size'] = original_dmem
                            # break
                            c += 1
                            # continue
                else:
                    continue
                # c += 1
                if c == self.inner_loop_threshold:
                    early_stop_count += 1
                    self.logger.info(f"Inner Loop Threshold: {self.inner_loop_threshold} reached, break!")
                    break
            #
            if  early_stop_count == self.early_stop_threshold or traversal_index == self.inner_loop_threshold:
                self.logger.info(f"Early Stop Threshold: {self.inner_loop_threshold} reached, break!")
                break
