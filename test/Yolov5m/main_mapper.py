import torch
import torch.nn.functional as F
import numpy as np
import json
import pickle
import argparse
import sys
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CIM toolchain
from mapper.parser import *
from mapper.placement import *
from mapper.search import *
from mapper.helper import *
from mapper.mapper import *
from cimruntime.CIMA.simulation.cima_rt import CIMANumpyRT as CNRT
from mapper.self_defined_op.cima_op import *
from backend.CIMA.pytorch.gen_code import TorchRTCodeGen
from optimizer.CIMA.sim_in_loop_dmem_optim import CIMA_DMEM_Optimizer
from mapper.placement.CIMA import CIMA_DMEM_allocation

dmac_layer = ['backbone_Conv_0']


def dump_ir_yaml(ir_obj, file_path):
    """Dump IR to a real YAML file instead of JSON-with-.yaml-extension."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to export YAML IR files. "
            "Please install it with: pip install pyyaml"
        )
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ir_obj.to_json_obj(), f, sort_keys=False, allow_unicode=False)

def make_silu_lut(in_scale, out_scale, in_dtype, out_dtype):
    outputs = []
    if  in_dtype == '4bit':
        in_thd = 8
    elif in_dtype == '8bit':
        in_thd = 128
    else:
        raise ValueError(f'Unsupported in_dtype: {in_dtype}')
    
    if out_dtype == '4bit':
        out_thd = 8
    elif out_dtype == '8bit':
        out_thd = 128
    else:
        raise ValueError(f'Unsupported out_dtype: {out_dtype}')
    
    for input in range(-in_thd, in_thd):
        output = F.silu(input * in_scale)
        output_quant = torch.clamp(output / out_scale, min= -out_thd, max= out_thd -1).round().int().item()
        outputs.append(output_quant)

    outputs = torch.tensor(outputs).to(torch.int32)
    return outputs

def get_lut_params(hard_params):

    activation_lut = {}
    for k, v in hard_params.items():
        if 'Silu' in k:
            in_scale = v['in_scale']
            out_scale = v['out_scale']
            if v['out_thd_pos'] == 7:
                out_dtype = '4bit'
            elif v['out_thd_pos'] == 127:
                out_dtype = '8bit'
            else:
                raise ValueError(f'Unsupported out_thd_pos: {v["out_thd_pos"]} in layer {k}')
            in_dtype = '4bit'
            lut = make_silu_lut(in_scale, out_scale, in_dtype, out_dtype)
            activation_lut[k] = lut

    torch.save(activation_lut, 'algo\\activation_lut.pth')

    return activation_lut

def get_layer_dtype(hard_params):

    layer_dtype = {}
    for k, v in hard_params.items():
        if 'bn' not in k:
            if 'Conv' in k:
                out_thd_pos = v['out_thd_pos']
                assert out_thd_pos in [7, 127], f'Unsupported out_thd_pos: {out_thd_pos} in layer {k}'
                layer_dtype[k] = '4bit' if out_thd_pos == 7 else '8bit'
            elif 'Add' in k:
                bit = v['bit']
                assert bit in [4, 8], f'Unsupported bit: {bit} in layer {k}'
                layer_dtype[k] = '4bit' if bit == 4 else '8bit'

    return layer_dtype

def get_soft_scale(hard_params):

    soft_scale = {}
    for k, v in hard_params.items():
        if 'Conv' in k and 'Silu' not in k and 'bn' not in k:
            assert 'soft_scale' in v, f"soft_scale not in hard_params for layer {k}"
            soft_scale[k] = v['soft_scale']

    return soft_scale

def get_chip_weight_adc_range(w_int, soft_scale, w_code_upper_limit=22, w_code_lower_limit=20):

    # hardware scale
    hard_in_scale = 0.0957 / 7  # input voltage level
    hard_w_scale = 204.48 / 127  # RRAM conductance level
    current_level = [32, 40, 64, 80, 120, 160, 200]
    adc_range = {}
    w_int_ = {}
    for k, v in w_int.items():
        v_ = 0
        if k not in dmac_layer:
            soft_scale_ = float(soft_scale[k])
            w_code = []
            diff = []
            for i in current_level:
                hard_out_scale = 127 / i
                w_code_ = round((soft_scale_ / (hard_in_scale * hard_w_scale * hard_out_scale)))
                if w_code_ > w_code_upper_limit:
                    w_code_ = w_code_upper_limit
                if w_code_ < w_code_lower_limit:
                    w_code_ = w_code_lower_limit
                hard_scale = hard_in_scale * hard_w_scale * hard_out_scale * w_code_
                diff.append(abs(soft_scale_ - hard_scale))
                w_code.append(w_code_)
            
            min_diff_index = torch.argmin(torch.tensor(diff)).item()
            adc_range[k] = int(min_diff_index)
            v_ = v * w_code[min_diff_index]
        else:
            adc_range[k] = 0
            v_ = v
        w_int_[k + '.weight'] = v_

    torch.save(w_int_, 'algo\\weight_int_chip.pth')

    return w_int_, adc_range

def get_insert_op(hard_params):

    insert_op_list = []

    for k, v in hard_params.items():
        if 'Add' in k and 'Silu' not in k:
            assert 'scale_num_shift_list' in v, f"scale_num_shift_list not in hard_params for layer {k}"
            if v['scale_num_shift_list'][0] != 1 or v['scale_num_shift_list'][1] != 0 and \
            (v['scale_num_shift_list'][0] != 1 << v['scale_num_shift_list'][1]):
                insert_op_list.append((k, [v['scale_ind']]))
        
        if 'Concat' in k and 'Silu' not in k:
            c = 0
            t = []
            IsInsert = False
            for scale in v['scale_num_shift_list']:
                if scale[0] != 1 or scale[1] != 0:
                    IsInsert = True
                    t.append(c)
                c += 1
            if IsInsert:
                insert_op_list.append((k, t))

    return insert_op_list   

def get_insert_op_params(hard_params, insert_op_name_dict):
    insert_op_params = {}
    for k, v in insert_op_name_dict.items():
        if 'Concat' in k:
            params = hard_params[k]
            for (ln, index) in v:
                insert_op_params[ln] = {}
                insert_op_params[ln]['scale'] = int(params['scale_num_shift_list'][index][0])
                insert_op_params[ln]['scale_shift_num'] = int(params['scale_num_shift_list'][index][1])
                insert_op_params[ln]['offset'] = 0
        elif 'Add' in k:
            params = hard_params[k]
            for (ln, index) in v:
                insert_op_params[ln] = {}
                insert_op_params[ln]['scale'] = int(params['scale_num_shift_list'][0])
                insert_op_params[ln]['scale_shift_num'] = int(params['scale_num_shift_list'][1])
                insert_op_params[ln]['offset'] = 0

    return insert_op_params

def compile(model_name, insert_op_list=None, specify_output_layer=None):

    devices = {
        'name':'cima-0',
        'kind':'cima-node',
        'num': 36,
        'height':4,
        'width':9,
        'task_num': 128,
    }  

    onnx_path = f'model\\{model_name}.onnx'
    onnx_obj = ConvertONNX(onnx_path, specify_output_layer=specify_output_layer)
    onnx_ir = onnx_obj.ir
    dump_ir_yaml(onnx_ir, f'ir\\{model_name}_onnx_ir.yaml')

    masked_id_list = [(0,3), (0,4), (0,5), (3,4), (3,5)]
    masked_pe = [
        'cima-0.cima-node:2.cima-pe-cluster:0',
        'cima-0.cima-node:7.cima-pe-cluster:2',
        'cima-0.cima-node:7.cima-pe-cluster:3',
        'cima-0.cima-node:7.cima-pe-cluster:0',
        'cima-0.cima-node:8.cima-pe-cluster:3',
        'cima-0.cima-node:12.cima-pe-cluster:2',
        'cima-0.cima-node:12.cima-pe-cluster:1',
        'cima-0.cima-node:13.cima-pe-cluster:1',
        'cima-0.cima-node:14.cima-pe-cluster:2',
        'cima-0.cima-node:15.cima-pe-cluster:1',
        'cima-0.cima-node:16.cima-pe-cluster:3',
        'cima-0.cima-node:17.cima-pe-cluster:1',
        'cima-0.cima-node:17.cima-pe-cluster:3',
        'cima-0.cima-node:18.cima-pe-cluster:1',
        'cima-0.cima-node:21.cima-pe-cluster:3',
        'cima-0.cima-node:25.cima-pe-cluster:0',
        'cima-0.cima-node:25.cima-pe-cluster:3',
        'cima-0.cima-node:26.cima-pe-cluster:3',
        'cima-0.cima-node:28.cima-pe-cluster:2',
        'cima-0.cima-node:33.cima-pe-cluster:0',
        'cima-0.cima-node:33.cima-pe-cluster:2',
        'cima-0.cima-node:35.cima-pe-cluster:2',
    ]
    masked_xb = []
    for pe in masked_pe:
        for i in range(16):
            masked_xb.append(pe + f'.cima-xb:{i}')
    # masked_xb = []
    # print('Masked devices:')
    # for device in masked_device:
    #     print(device)

    # mapping
    map = mapper(ir=onnx_ir, device = devices,
                weight_format = 'HWC',
                place_strategy = CIMAPlacement,
                relu_fuse= True,
                pool_fuse= True,
                split_fuse = False,
                silu_fuse = False,
                masked_device_id_list = masked_id_list,
                type_conversion_list= None,
                adaptive_split_ir = True,
                operator_replace = True,
                target_device = 'cima',
                # layer_data_type_dict = layer_data_type_dict
                )
    method = 'workload_balance'
    datawidth = 4
    map.run(CIMA_method=method, CIMA_datawidth=datawidth, CIMA_dmac_layer=dmac_layer,
            CIMA_insert_mul_add_op=insert_op_list, masked_xb=masked_xb)
    mapped_ir = map.ir
    dump_ir_yaml(mapped_ir, f'ir\\{model_name}_mapped_ir.yaml')

    insert_op_name_dict = map.place.insert_op_name_dict

    path = f'ir\\{model_name}_mapped_ir.yaml'

    ir = load_ir(file = path)

    ir_dmem_opt = CIMA_DMEM_allocation(ir)
    dump_ir_yaml(ir_dmem_opt, f'ir\\{model_name}_dmem_opt_mapped_ir.yaml')
    return insert_op_name_dict

def get_dmac_params(hard_params):
    dmac_params = {}
    for l in dmac_layer:
        assert l in hard_params.keys(), f"{l} not in hard_params"
        dmac_params[l] = {}
        dmac_params[l]['accumulate_shift_num'] = int(hard_params[l]['DMAC_sn'])

    return dmac_params

def get_bn_params(hard_params):
    bn_params = {}
    for k, v in hard_params.items():
        if 'bn' in k:
            k_ = k.replace('_bn', '')
            assert k_ in hard_params.keys(), f"{k_} not in hard_params for bn layer {k}"
            bn_params[k_] = {}
            bn_params[k_]['scale'] = []
            bn_params[k_]['offset'] = []
            bn_params[k_]['scale_shift_num'] = []
            if v['sharedBN']:
                bn_params[k_]['scale'].append([int(i) for i in v['scale_mul']]*2)
                bn_params[k_]['offset'].append([int(i) for i in v['offset']]*2)
                bn_params[k_]['scale_shift_num'].append(v['scale_shift'])
            else:
                bn_params[k_]['scale'].append([int(i) for i in v['scale_mul']])
                bn_params[k_]['offset'].append([int(i) for i in v['offset']])
                bn_params[k_]['scale_shift_num'].append(v['scale_shift'])
            bn_params[k_]['sharedBN'] = v['sharedBN']

    return bn_params

def ir_update(model_name, insert_op_params, adc_range, bn_params, dmac_params):

    ir = load_ir(file = f'ir\\{model_name}_dmem_opt_mapped_ir.yaml')
    for k, v in ir.layers.items():
        if adc_range!= None and k in adc_range.keys():
            v.CIMA_calc_info.ADC_quant_level = adc_range[k]
        if k in bn_params.keys():
            v.CIMA_calc_info.scale_shift_num = bn_params[k]['scale_shift_num']
            v.CIMA_calc_info.scale = bn_params[k]['scale']
            v.CIMA_calc_info.offset = bn_params[k]['offset']
            v.CIMA_calc_info.sharedBN = bn_params[k]['sharedBN']
        elif k in dmac_params.keys():
            v.CIMA_calc_info.accumulate_shift_num = dmac_params[k]['accumulate_shift_num']
        elif k in insert_op_params.keys():
            v.CIMA_calc_info.scale_shift_num = int(insert_op_params[k]['scale_shift_num'])
            v.CIMA_calc_info.scale = int(insert_op_params[k]['scale'])
            v.CIMA_calc_info.offset = 0
        else:
            pass

    dump_ir_yaml(ir, f'ir\\{model_name}_dmem_opt_mapped_ir_w_params.yaml')

def run_mapper(model_name):

    path = f'algo\\'
    hard_params = torch.load(path + 'hard_params_dict_cpu.pth')
    activation_lut = get_lut_params(hard_params)
    print('Activation LUT generated.')
    # print("activation_lut keys:", activation_lut.keys())

    soft_scale = get_soft_scale(hard_params)
    w_int = torch.load(path + 'weight_int_dict_cpu.pth')
    weights, adc_range = get_chip_weight_adc_range(w_int, soft_scale)
    print('Weights for chip generated.')

    current_level = [32, 40, 64, 80, 120, 160, 200]
    # for k, v in adc_range.items():
    #     print(f'Layer: {k},  Current level: {current_level[v]}')
    insert_op_list = get_insert_op(hard_params)
    # print('Insert op list:')
    # print(insert_op_list)
    insert_op_name_dict = compile(model_name,insert_op_list=insert_op_list)
    print('IR mapping completed.')
    insert_params = get_insert_op_params(hard_params, insert_op_name_dict)
    print('insert op params extracted.')
    # print('Insert op list:', insert_op_list)
    # print('Insert op name dict:', insert_op_name_dict)
    # print('Insert op params:', insert_params)

    bn_params = get_bn_params(hard_params)
    print('BN params extracted.')
    # print("bn_params keys:", bn_params.keys())

    dmac_params = get_dmac_params(hard_params)
    print('DMAC params extracted.')
    # print("dmac_params keys:", dmac_params.keys())

    ir_update(model_name, insert_params, adc_range, bn_params, dmac_params)
    print('IR updated with hardware parameters.')


def params_init(model_name):
    """Backward-compatible wrapper."""
    run_mapper(model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run mapping flow and export mapped IR files.")
    parser.add_argument("--model-name", default="yolov5m_wo_head")
    args = parser.parse_args()
    run_mapper(args.model_name)