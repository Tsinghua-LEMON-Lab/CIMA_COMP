from .regs import *
import json
import copy

REG_BASE_ADDR_DICT = {
    'rec_cfg_merge': reg_rec_addr,
    'thread_pe_cfg0_id': reg_pe_cfg0_addr,
    'thread_pe_cfg1_id': reg_pe_cfg1_addr,
    'thread_pe_cfg2_id': reg_pe_cfg2_addr,
    'thread_pe_cfg3_id': reg_pe_cfg3_addr,
    'thread_mfop_cfg0_id': reg_mfop_cfg0_addr,
    'thread_mfop_cfg1_id': reg_mfop_cfg1_addr,
    'thread_mfop_cfg2_id': reg_mfop_cfg2_addr,
    'thread_mfop_cfg3_id': reg_mfop_cfg3_addr,
    'thread_dmac_cfg0': reg_dmac_cfg0_addr,
    'thread_dmac_cfg1': reg_dmac_cfg1_addr,
    'thread_dmac_cfg2': reg_dmac_cfg2_addr,
    'thread_dmac_cfg3': reg_dmac_cfg3_addr,
    'pull_cfg_merge': reg_pull_cfg_merge_addr,
    'pull_info_0': reg_pull_info_0_addr,
    'pull_info_1': reg_pull_info_1_addr,
    'pull_info_2': reg_pull_info_2_addr,
    'cfg_mfop_info': reg_cfg_mfop_info_addr,
    'cfg_done_out_merge': reg_cfg_done_out_merge_addr,
    'cfg_done_cal_thread_merge': reg_cfg_done_cal_thread_merge_addr,
    'thread_vld_mask': reg_thread_vld_mask_addr,
    'common_cfg_calcfgdata_rsp_info': reg_common_cfg_calc_data_rsp_info_addr,
    'common_cfg_mcsend_update_rsp_info': reg_common_cfg_mcsend_update_rsp_info_addr,
    'sub_array_info': reg_sub_array_info_addr,
    'sub_array_idx_seg0123_t': reg_sub_array_idx_seg0123_addr,
    'sub_array_idx_seg4567_t': reg_sub_array_idx_seg4567_addr,
}

def register_json2sv(json_file, sv_file = f'sv.txt'):

    with open(json_file, 'r') as f:
        json_config = json.load(f)

    register_sv_str = []
    for k, v in json_config.items():
        register_sv_str.append(f'// {k}')
        register_sv_str.append('\n')
        register_sv_str.append('\n')
        for v_k, v_v in v.items():
            if 'Thread' in v_k:
                register_sv_str.append(f"// {v_k}, Task Name: {v_v['task_name']}, OP_Code: {v_v['thread_type']}")
                register_sv_str.append(f'\n')
                for thread_k, thread_v in v_v.items():
                    if thread_k not in ['task_name', 'thread_type']:
                        if 'dmac' in thread_k:
                            reg_name = thread_k
                            reg_offset = 0
                        else:
                            reg_name_list = thread_k.split('_')
                            reg_offset = int(reg_name_list[-1])
                            reg_name = '_'.join(reg_name_list[0:-1])
                        if reg_name == 'pull_info':
                            reg_name = reg_name + f'_{reg_offset // 1024}'
                            reg_offset = reg_offset - (reg_offset // 1024) * 1024
                        assert reg_name in REG_BASE_ADDR_DICT.keys(), f'{reg_name}'
                        reg_base_obj = copy.deepcopy(REG_BASE_ADDR_DICT[reg_name])
                        if reg_name in ['sub_array_info', 'pull_info_0', 'pull_info_1', 'pull_info_2']:
                            bits_offset_0 = reg_offset % 256
                            bits_offset_1 = reg_offset // 256
                            reg_base_obj = reg_base_obj(bits_offset_0=bits_offset_0, bits_offset_1=bits_offset_1)
                        else:
                            reg_base_obj = reg_base_obj()
                            original_base_addr = reg_base_obj.bits_offset_0
                            if 'pe' in reg_name or 'mfop' in reg_name:
                                reg_base_obj.bits_offset_0 = original_base_addr + 4 * reg_offset
                            elif 'sub_array_idx' in reg_name:
                                reg_base_obj.bits_offset_0 = original_base_addr + 2 * reg_offset
                            else:
                                reg_base_obj.bits_offset_0 = original_base_addr + reg_offset

                        reg_addr = reg_base_obj.to_verilog_hex()
                        reg_ea = reg_base_obj.ea
                        reg_value = get_value_from_bit_name(thread_v)
                        register_sv_str.append(f"dc_cfg_op(.flag(1), .id(`LOCAL_CORE_ID), .op(0), .sn(30), .a({reg_addr}), .ea({reg_ea}), .d({reg_value}), .rc(8));")
                        register_sv_str.append(f'\n')
            elif v_k == 'cfg_mfop_info':
                reg_name = v_k
                assert reg_name in REG_BASE_ADDR_DICT.keys()
                reg_base_obj = copy.deepcopy(REG_BASE_ADDR_DICT[reg_name])
                reg_base_obj = reg_base_obj()
                reg_addr = reg_base_obj.to_verilog_hex()
                reg_ea = reg_base_obj.ea
                reg_value = get_value_from_bit_name(v_v)
                register_sv_str.append(f'// MFOP Extra Info configuration\n')
                register_sv_str.append(f"dc_cfg_op(.flag(1), .id(`LOCAL_CORE_ID), .op(0), .sn(30), .a({reg_addr}), .ea({reg_ea}), .d({reg_value}), .rc(8));")
                register_sv_str.append(f'\n')
            elif v_k == 'valid_config':
                register_sv_str.append(f'// valid configuration\n')
                for thread_k, thread_v in v_v.items():
                    reg_name = thread_k
                    assert reg_name in REG_BASE_ADDR_DICT.keys()
                    reg_base_obj = copy.deepcopy(REG_BASE_ADDR_DICT[reg_name])
                    reg_base_obj = reg_base_obj()
                    reg_addr = reg_base_obj.to_verilog_hex()
                    reg_ea = reg_base_obj.ea
                    reg_value = get_value_from_bit_name(thread_v)
                    register_sv_str.append(f"dc_cfg_op(.flag(1), .id(`LOCAL_CORE_ID), .op(0), .sn(30), .a({reg_addr}), .ea({reg_ea}), .d({reg_value}), .rc(8));")
                    register_sv_str.append(f'\n')
            register_sv_str.append(f'\n')
    with open(sv_file, 'w') as f:
        f.write('''
// **************************************** #
// This file is automatically generated !!! #
//        Please do not modify it !!!       #
// **************************************** #
''')
        f.write('\n')
        for str_ in register_sv_str:
            f.write(str_)

def get_value_from_bit_name(bit_name):
    value = 0
    for bn, bv in bit_name.items():
        bit_pos = bn.split('[')[1]
        bit_pos = int(bit_pos.split(':')[0])
        #
        value += bv * 2**(bit_pos)
    hex_ = hex(value)
    str_list = list(hex_)
    while len(str_list) < 18:
        str_list.insert(2, '0')
    hex_ = ''.join(str_list)
    return hex_.replace("0x", "64'h")

def convert_hex_to_verilog_hex_64bit(value):
    if isinstance(value, str):
        value = int(value, 16)
    hex_ = hex(value)
    head_str = "64'h"
    len_ = 18
    str_list = list(hex_)
    while len(str_list) < len_:
        str_list.insert(2, '0')
    hex_ = ''.join(str_list)
    return hex_.replace("0x", head_str)

def convert_hex_to_verilog_hex_32bit(value):
    if isinstance(value, str):
        value = int(value, 16)
    hex_ = hex(value)
    head_str = "32'h"
    len_ = 10
    str_list = list(hex_)
    while len(str_list) < len_:
        str_list.insert(2, '0')
    hex_ = ''.join(str_list)
    return hex_.replace("0x", head_str)
