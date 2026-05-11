import json
import torch
import numpy
import math
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irtool.core import load_ir
from mapper.device.CIMA import *  # noqa
from mapper.self_defined_op import *  # noqa


def gen_weight(
    weight_chip,
    sys_cfg,
    ir,
    output_file="chip\\yolov5m_wo_head_weight.json",
    detail_log=print,
    progress_callback=None,
):
    # remove 'Run_Time', 'HOSTI', 'DDRI' in sys_cfg
    sys_cfg.pop('Run_Time', None)
    sys_cfg.pop('HOSTI', None)
    sys_cfg.pop('DDRI', None)

    def split_weight(tensor: torch.Tensor, row_split_num: int, col_split_num: int) -> list[list[int]]:
        """
        convert 4D tensor [cout, cin, kh, kw] to 2D matrix [cin * kh * kw, cout](row: cin -> kw -> kh)
        """ 
        cout, cin, kh, kw = tensor.shape
        # 
        tensor = tensor.permute(0, 3, 2, 1)
        col_matrix = tensor.reshape(cout, cin * kh * kw).t()
        # print(col_matrix)
        if col_matrix.shape[0] % row_split_num != 0 or col_matrix.shape[1] % col_split_num != 0:
            raise ValueError(f"matrix size({col_matrix.shape[0]},{col_matrix.shape[1]})can not be divided by({row_split_num},{col_split_num})")
        h_block = col_matrix.shape[0] // row_split_num
        w_block = col_matrix.shape[1] // col_split_num
        # print(f"Block size: {h_block}x{w_block}")
        blocks = []
        for col_idx in range(col_split_num):
            for row_idx in range(row_split_num):
                # calculate the start and end positions of the current block
                row_start = row_idx * h_block
                row_end = (row_idx + 1) * h_block
                col_start = col_idx * w_block
                col_end = (col_idx + 1) * w_block

                # extract sub-matrix
                block = col_matrix[row_start:row_end, col_start:col_end]
                block = block.T.flatten().tolist()
                block = [int(round(w)) for w in block]
                blocks.append(block)

        return blocks
    def split_list(lst: list, chunk_size: int) -> list[list]:
        """
        Split a list into chunks of a specified size.
        """
        assert chunk_size > 0, "Chunk size must be positive"
        assert len(lst) % chunk_size == 0, f"List length {len(lst)} must be divisible by chunk size {chunk_size}"
        chunk_size = len(lst) // chunk_size
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    pdk_config = {}
    pe_h = 576
    pe_w = 128

    core_items = list(sys_cfg.items())
    total_cores = len(core_items)
    if progress_callback is not None:
        progress_callback(0, total_cores, None)

    for core_idx, (core, core_cfg) in enumerate(core_items, start=1):
        pdk_config[core] = {}
        E_PE = {}
        S_PE = {}
        W_PE = {}
        N_PE = {}
        PE_dir_dict = {0: 'W', 1: 'N', 2: 'E', 3: 'S'}
        is_used_pe = False
        buffer_index_E = 0
        buffer_index_S = 0
        buffer_index_W = 0
        buffer_index_N = 0

        detail_log(f"-------------------{core}------------------------")
        for t, t_cfg in core_cfg.items():
            # if t_cfg['Task_Name'] not in ['Conv_26', 'Conv_29']:
            #     continue
            if t_cfg['Op_Code'] == 'PEConv':
                is_used_pe = True
                layer_name = t_cfg['Task_Name']
                pe_dir = PE_dir_dict[t_cfg['Conv_Struct']['rela_pe']]
                detail_log(f"Processing layer: {layer_name}, PE direction: {pe_dir}")
                lpe = t_cfg['Conv_Struct']['lpe']
                bn = t_cfg['Conv_Struct']['bn']
                assert len(lpe) == len(bn), f"lpe {len(lpe)} != bn {len(bn)}"   
                # print(f"Processing layer: {layer_name}, PE direction: {pe_dir}")
                assert layer_name + '.weight' in weight_chip.keys(), f"{layer_name} not in weight_chip"

                # get_weight_blocks
                weight = weight_chip[layer_name + '.weight']
                # [cout, cin, h, w] -> [cout, h, w, cin] -> flatten
                shape = weight.shape
                cout, cin, kh, kw = shape
                image2col_shape = (cin * kh * kw, cout)
                pe_row_split_num = math.ceil(image2col_shape[0] / pe_h)
                pe_col_split_num = math.ceil(image2col_shape[1] / pe_w)
                pe_split_num = (pe_row_split_num, pe_col_split_num)
                assert len(lpe) == pe_row_split_num * pe_col_split_num
                weight_blocks = split_weight(weight, pe_row_split_num, pe_col_split_num)
                assert len(weight_blocks) == len(lpe), f"weight blocks {len(weight_blocks)} != lpe {len(lpe)}"
                block_shape = (image2col_shape[0] // pe_row_split_num, image2col_shape[1] // pe_col_split_num)
                # get scale
                assert layer_name in ir.layers.keys(), f"{layer_name} not in ir layers"
                scale = ir.layers[layer_name].CIMA_calc_info.scale[0]
                offset = ir.layers[layer_name].CIMA_calc_info.offset[0]
                scale_shift_num = ir.layers[layer_name].CIMA_calc_info.scale_shift_num[0]
                accumulate_shift_num = ir.layers[layer_name].CIMA_calc_info.accumulate_shift_num
                ADC_range = ir.layers[layer_name].CIMA_calc_info.ADC_quant_level
                assert isinstance(scale, list), f"scale {scale} must be a list"
                assert isinstance(offset, list), f"offset {offset} must be a list"
                assert isinstance(scale_shift_num, int), f"scale_shift_num {scale_shift_num} must be a int"
                # if isinstance(scale, int):
                #     scale = [scale] * cout
                # if isinstance(offset, int):
                #     offset = [offset] * cout
                # if isinstance(scale_shift_num, int):
                #     scale_shift_num = [scale_shift_num] * cout
                split_scale = split_list(scale, pe_col_split_num)
                split_offset = split_list(offset, pe_col_split_num)
                # split_scale_shift_num = split_list(scale_shift_num, pe_col_split_num)

                for block_idx, pe_idx in enumerate(lpe):
                    if pe_dir == 'E':
                        E_PE.setdefault(pe_idx, {})
                        E_PE[pe_idx]['task'] = layer_name + f':{block_idx}'
                        E_PE[pe_idx]['shape'] = block_shape
                        E_PE[pe_idx]['weight'] = weight_blocks[block_idx]
                        E_PE[pe_idx]['scale'] = split_scale[block_idx]
                        E_PE[pe_idx]['offset'] = split_offset[block_idx]
                        # E_PE[pe_idx]['scale_shift_num'] = split_scale_shift_num[idx]
                        E_PE[pe_idx]['scale_shift_num'] = scale_shift_num
                        E_PE[pe_idx]['accumulate_shift_num'] = accumulate_shift_num
                        E_PE[pe_idx]['ADC_range'] = ADC_range
                        E_PE[pe_idx]['buffer_index'] = bn[block_idx]

                    elif pe_dir == 'S':
                        S_PE.setdefault(pe_idx, {})
                        S_PE[pe_idx]['task'] = layer_name + f':{block_idx}'
                        S_PE[pe_idx]['shape'] = block_shape
                        S_PE[pe_idx]['weight'] = weight_blocks[block_idx]
                        S_PE[pe_idx]['scale'] = split_scale[block_idx]
                        S_PE[pe_idx]['offset'] = split_offset[block_idx]
                        # S_PE[pe_idx]['scale_shift_num'] = split_scale_shift_num[idx]
                        S_PE[pe_idx]['scale_shift_num'] = scale_shift_num
                        S_PE[pe_idx]['accumulate_shift_num'] = accumulate_shift_num
                        S_PE[pe_idx]['ADC_range'] = ADC_range
                        S_PE[pe_idx]['buffer_index'] = bn[block_idx]

                    elif pe_dir == 'W':
                        W_PE.setdefault(pe_idx, {})
                        W_PE[pe_idx]['task'] = layer_name + f':{block_idx}'
                        W_PE[pe_idx]['shape'] = block_shape
                        W_PE[pe_idx]['weight'] = weight_blocks[block_idx]
                        W_PE[pe_idx]['scale'] = split_scale[block_idx]
                        W_PE[pe_idx]['offset'] = split_offset[block_idx]
                        # W_PE[pe_idx]['scale_shift_num'] = split_scale_shift_num[idx]
                        W_PE[pe_idx]['scale_shift_num'] = scale_shift_num
                        W_PE[pe_idx]['accumulate_shift_num'] = accumulate_shift_num
                        W_PE[pe_idx]['ADC_range'] = ADC_range
                        W_PE[pe_idx]['buffer_index'] = bn[block_idx]

                    elif pe_dir == 'N':
                        N_PE.setdefault(pe_idx, {})
                        N_PE[pe_idx]['task'] = layer_name + f':{block_idx}'
                        N_PE[pe_idx]['shape'] = block_shape
                        N_PE[pe_idx]['weight'] = weight_blocks[block_idx]
                        N_PE[pe_idx]['scale'] = split_scale[block_idx]
                        N_PE[pe_idx]['offset'] = split_offset[block_idx]
                        # N_PE[pe_idx]['scale_shift_num'] = split_scale_shift_num[idx]
                        N_PE[pe_idx]['scale_shift_num'] = scale_shift_num
                        N_PE[pe_idx]['accumulate_shift_num'] = accumulate_shift_num
                        N_PE[pe_idx]['ADC_range'] = ADC_range
                        N_PE[pe_idx]['buffer_index'] = bn[block_idx]

                    else:
                        raise ValueError(f"Unknown PE direction: {pe_dir}")
                    
        if len(E_PE) > 0:
            pdk_config[core]['E'] = E_PE
            for k, v in E_PE.items():
                detail_log(f"    PE: E, lpe: {k}, layer: {v['task']}")
        if len(S_PE) > 0:
            pdk_config[core]['S'] = S_PE
            for k, v in S_PE.items():
                detail_log(f"    PE: S, lpe: {k}, layer: {v['task']}")
        if len(W_PE) > 0:
            pdk_config[core]['W'] = W_PE
            for k, v in W_PE.items():
                detail_log(f"    PE: W, lpe: {k}, layer: {v['task']}")
        if len(N_PE) > 0:
            pdk_config[core]['N'] = N_PE
            for k, v in N_PE.items():
                detail_log(f"    PE: N, lpe: {k}, layer: {v['task']}")
        if not is_used_pe:
            pdk_config.pop(core)
        if progress_callback is not None:
            progress_callback(core_idx, total_cores, core)

    with open(output_file, 'w') as f:
        json.dump(pdk_config, f, indent=4)

def gen_feas(inout, sys_cfg):
    print(inout.keys())
    print(inout['Conv_0']['input_int'][0].shape)

    with open('sdk\\layer_feas_batch_images100.txt', 'w') as f:
        for k, v in inout.items():
            f.write(f'Layer: {k}\n')
            f.write(f"input_int(shape: {v['input_int'][0].shape})\n")
            for i in range(len(v['input_int'])):
                input_feat = v['input_int'][i]
                # flatten
                input_feat_ = input_feat.permute(0, 2, 3, 1)  # NCHW to NHWC
                input_feat_ = input_feat_.flatten()
                # print all values
                for val in input_feat_:
                    f.write(f"{int(val.item())}, ")
                f.write("\n")
            
            f.write(f"output_int(shape: {v['output_int'][0].shape})\n")
            for i in range(len(v['output_int'])):
                output_feat = v['output_int'][i]
                # flatten
                output_feat = output_feat.flatten()
                # print all values
                for val in output_feat:
                    f.write(f"{int(val.item())}, ")
                f.write("\n")
            f.write("\n")

def gen_targets(targets):
    with open("sdk\\batch_targets100.txt", "w") as f:
        for t in targets:
            f.write(f"{t}\n")


def _infer_model_name_from_output_file(output_file):
    stem = Path(output_file).stem
    return stem[:-7] if stem.endswith("_weight") else stem


def _render_progress_bar(current, total, width=30):
    if total <= 0:
        return "[------------------------------] 100.0% (0/0)"
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


def run_weight_extract(
    weight_chip_file="algo\\weight_int_chip.pth",
    systemc_json_file="uvm\\yolov5m_wo_head_systemc.json",
    ir_file="ir\\yolov5m_wo_head_dmem_opt_mapped_ir_w_params.yaml",
    output_file="chip\\yolov5m_wo_head_weight.json",
    log_file=None,
):
    print("[STEP] Weight extract started.")
    model_name = _infer_model_name_from_output_file(output_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else output_path.parent / f"{model_name}_weight_extract.log"

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write(f"[INFO] Weight extract started at {datetime.now().isoformat(timespec='seconds')}\n")
        log_fp.write(f"[INFO] Model name: {model_name}\n")
        log_fp.write(f"[INFO] weight_chip_file: {weight_chip_file}\n")
        log_fp.write(f"[INFO] systemc_json_file: {systemc_json_file}\n")
        log_fp.write(f"[INFO] ir_file: {ir_file}\n")
        log_fp.write(f"[INFO] output_file: {output_file}\n")
        log_fp.write(f"[INFO] log_file: {log_path.as_posix()}\n")
        try:
            weight_chip = torch.load(weight_chip_file)
            with open(systemc_json_file) as f:
                sys_cfg = json.load(f)
            ir = load_ir(file=ir_file)
            log_fp.write(f"[DETAIL] weight tensor count: {len(weight_chip)}\n")
            log_fp.write(f"[DETAIL] system cfg core count: {len(sys_cfg)}\n")
            log_fp.write(f"[DETAIL] ir layer count: {len(ir.layers)}\n")

            def progress_callback(current, total, core_name):
                progress_text = _render_progress_bar(current, total)
                if core_name:
                    sys.stdout.write(f"\r[STEP] Weight extract progress {progress_text} core={core_name}")
                else:
                    sys.stdout.write(f"\r[STEP] Weight extract progress {progress_text}")
                sys.stdout.flush()

            gen_weight(
                weight_chip,
                sys_cfg,
                ir,
                output_file=output_file,
                detail_log=lambda msg: log_fp.write(f"{msg}\n"),
                progress_callback=progress_callback,
            )
            sys.stdout.write("\n")
            sys.stdout.flush()
            log_fp.write(f"[INFO] Weight extract finished at {datetime.now().isoformat(timespec='seconds')}\n")
            print("[STEP] Weight extract finished.")
        except Exception:
            sys.stdout.write("\n")
            sys.stdout.flush()
            log_fp.write(f"[ERROR] Weight extract failed at {datetime.now().isoformat(timespec='seconds')}\n")
            log_fp.write(traceback.format_exc())
            print("[STEP] Weight extract failed. Check log for details.")
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chip weight config from mapper outputs.")
    parser.add_argument("--model-name", default="yolov5m_wo_head")
    parser.add_argument("--weight-chip-file", default="algo\\weight_int_chip.pth")
    parser.add_argument("--systemc-json-file", default="uvm\\yolov5m_wo_head_systemc.json")
    parser.add_argument("--ir-file", default="ir\\yolov5m_wo_head_dmem_opt_mapped_ir_w_params.yaml")
    parser.add_argument("--output-file", default="chip\\yolov5m_wo_head_weight.json")
    parser.add_argument("--log-file", default=None, help="Weight extract log file path. Default: chip/{model_name}_weight_extract.log")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    systemc_json_file = args.systemc_json_file
    ir_file = args.ir_file
    output_file = args.output_file
    if "--systemc-json-file" not in sys.argv:
        systemc_json_file = f"uvm\\{args.model_name}_systemc.json"
    if "--ir-file" not in sys.argv:
        ir_file = f"ir\\{args.model_name}_dmem_opt_mapped_ir_w_params.yaml"
    if "--output-file" not in sys.argv:
        output_file = f"chip\\{args.model_name}_weight.json"
    run_weight_extract(
        weight_chip_file=args.weight_chip_file,
        systemc_json_file=systemc_json_file,
        ir_file=ir_file,
        output_file=output_file,
        log_file=args.log_file,
    )