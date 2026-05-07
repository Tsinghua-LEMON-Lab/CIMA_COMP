from ..uvm.gen_code import UVMCodeGen
from ..uvm.regs import *
from ..uvm.helper import convert_hex_to_verilog_hex_32bit, convert_hex_to_verilog_hex_64bit
from collections import OrderedDict
#
import json
import math

class ChipCodeGen(UVMCodeGen):

    def __init__(self, systemc_config_path, actf_file=None, cutlength: dict = None, module_name='Inference',valid_core=None,
                dump_register_field_value=False):
        super().__init__(systemc_config_path, actf_file, cutlength, module_name, valid_core)
        self.dump_register_field_value = dump_register_field_value

    def gen_layers(self, chip_json_file=None, actf_file=None, use_dc=0, simu=0):
        self.chip_config = {}

        with open(self.systemc_config_path, 'r') as f:
            sc_config = json.load(f)

        self.all_thread_hardware_index = {}
        self.all_layer_hardware_index = {}
        PE_hardware_index = {'E':0, 'S':2, 'W':4, 'N':6}
        for core_name, threads in sc_config.items():
            if 'Core' in core_name or 'HOSTI' in core_name or 'DDRI' in core_name:
                #     core_name = 'Core0_4'
                #     core_name = 'Core3_4'
                PEConv_num = {'E':0, 'S':0, 'W':0, 'N':0}
                DMAC_num = 0
                MFOP_num = 0
                transfer_num = 0
                writein_num = 0
                writeout_num = 0
                for thread_id, thread_info in threads.items():
                    if (thread_info['Op_Code'] == 'WriteIn' and 'HOSTI' in core_name) or (thread_info['Op_Code'] == 'DDRReadIn' and 'DDRI' in core_name):
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = 0 + writein_num
                        if thread_info['Task_Name'] not in self.all_layer_hardware_index:
                            self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] = []
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"].append(writein_num)

                        writein_num += 1
                        if writein_num == 11:
                            writein_num = 43

                    if (thread_info['Op_Code'] == 'DDRWriteOut' and 'DDRI' in core_name) or (thread_info['Op_Code'] == 'WriteOut' and 'HOSTI' in core_name):
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = 48 + writeout_num
                        if thread_info['Task_Name'] not in self.all_layer_hardware_index:
                            self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] = []
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"].append(48 + writeout_num)

                        writeout_num += 1
                    if thread_info['Op_Code'] in ['Transfer']:
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = 11 + transfer_num
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] = 11 + transfer_num
                        transfer_num += 1
                    if thread_info['Op_Code'] == 'PEConv':
                        pe_direction = {0:'W', 1:'N', 2:'E', 3:'S'}
                        pe_dir = pe_direction[thread_info['Conv_Struct']['rela_pe']]
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = PE_hardware_index[pe_dir] + PEConv_num[pe_dir]
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] = PE_hardware_index[pe_dir] + PEConv_num[pe_dir]
                        PEConv_num[pe_dir] += 1
                    if thread_info['Op_Code'] == 'DMACConv':
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = 8 + DMAC_num
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] = 8 + DMAC_num
                    if thread_info['Op_Code'] in ['MaxPooling', 'AvgPooling', 'Upsample']:
                        self.all_thread_hardware_index[f'{core_name}-{thread_id}'] = 9 + MFOP_num
                        self.all_layer_hardware_index[f"{thread_info['Task_Name']}"] =  9 + MFOP_num
                        MFOP_num += 1

        # valid core index
        core_height = 4
        core_width = 9
        if self.valid_core == None:
            self.valid_core = []
            for i in range(core_height):
                for j in range(core_width):
                    self.valid_core.append(f'Core{i}_{j}')
        #

        self.PE_thread_all_core = {}

        self.PE_addr_offset = self.get_PE_addr_offset()

        self.mfop_thread_all_core_count = {}

        self.transfer_thread_all_core_count = {}

        self.transfer_seg_mcast_all_core_count = {}

        self.record_all_core_all_register_value = {}

        # yield f'// {self.module_name} chip codes (CIMA)'
        # yield

        # thread_core
        thread_core = []

        # bypass_core
        bypass_core = []

        # valid_core
        valid_core = []
        for core_name in sc_config.keys():
            if (core_name not in ['Run_Time']):
                if (len(sc_config[core_name].keys()) > 0):
                    valid_core.append(core_name)

        src_info = []
        dcid_list = []
        scid_list = []
        for core_name, threads in sc_config.items():
            if 'Core' in core_name or 'HOSTI' in core_name or 'DDRI' in core_name:
                for threads_id, threads_info in threads.items():
                    if 'Src' in threads_info.keys():
                        for src_id, src_info in threads_info['Src'].items():
                            # src_core[core_name] = src_info['core']
                            #     dst_src_link[core_name] = {}
                            # dst_src_link[core_name].setdefault(threads_id, {}).update({src_id: src_info['core']})

                            #     dst_src_link[core_name] = src_info['core']
                            # else:
                            #     dst_src_link[core_name].append(src_info['core'])

                            dcid_list.append(core_name)
                            scid_list.append(src_info['core'])

        delta_x = []
        delta_y = []
        bypass_list = []
        bypass_dict = {}

        index = 0
        while(index < len(dcid_list)):
            dcid = dcid_list[index]
            scid = scid_list[index]
            if dcid == 'HOSTI':
                dcid = 'Core0_4'
            if dcid == 'DDRI':
                dcid = 'Core3_4'
            if scid == 'HOSTI':
                scid = 'Core0_4'
            if scid == 'DDRI':
                scid = 'Core3_4'
            # delta_x.append(int(dcid[6]) - int(scid[6]))
            # delta_y.append(int(dcid[4]) - int(scid[4]))
            delta_x = int(dcid[6]) - int(scid[6])
            delta_y = int(dcid[4]) - int(scid[4])

            bypass_dict[index] = {}
            if delta_y == 0:
                bypass_dict[index][scid] = 'w' if delta_x < 0 else 'e'
            else:
                bypass_dict[index][scid] = 'n' if delta_y < 0 else 's'

            if abs(delta_x) + abs(delta_y) == 1:
                pass
            elif abs(delta_x) == 0:
                for i in range(max(int(scid[4]), int(dcid[4])) - 1, min(int(scid[4]), int(dcid[4])), -1):

                    if f'Core{i}_{scid[6]}'!= dcid:
                        bypass_dict[index][f'Core{i}_{scid[6]}'] = 'n' if (int(dcid[4]) - i) < 0 else 's'

                    if(f'Core{i}_{scid[6]}') not in bypass_list:
                        bypass_list.append(f'Core{i}_{scid[6]}')
            elif abs(delta_y) == 0:
                for i in range(max(int(scid[6]), int(dcid[6])) - 1, min(int(scid[6]), int(dcid[6])), -1):

                    if f'Core{scid[4]}_{i}'!= dcid:
                        bypass_dict[index][f'Core{scid[4]}_{i}'] = 'w' if (int(dcid[6]) - i) < 0 else 'e'

                    if(f'Core{scid[4]}_{i}') not in bypass_list:
                        bypass_list.append(f'Core{scid[4]}_{i}')
            else:
                for i in range(max(int(scid[4]), int(dcid[4])) - 1, min(int(scid[4]), int(dcid[4])), -1):

                    if f'Core{i}_{scid[6]}'!= dcid:
                        bypass_dict[index][f'Core{i}_{scid[6]}'] = 'n' if (int(dcid[4]) - i) < 0 else 's'

                    if(f'Core{i}_{scid[6]}') not in bypass_list:
                        bypass_list.append(f'Core{i}_{scid[6]}')

                for i in range(max(int(scid[6]), int(dcid[6])), min(int(scid[6]), int(dcid[6])) - 1, -1):

                    if f'Core{dcid[4]}_{i}'!= dcid:
                        bypass_dict[index][f'Core{dcid[4]}_{i}'] = 'w' if (int(dcid[6]) - i) < 0 else 'e'

                    if(f'Core{dcid[4]}_{i}') not in bypass_list:
                        bypass_list.append(f'Core{dcid[4]}_{i}')
            index+=1

        core_pull_cnt = {}
        for seq in bypass_dict.keys():
            for pull_cid, pull_rx in bypass_dict[seq].items():
                if pull_cid not in core_pull_cnt:
                    core_pull_cnt[pull_cid] = {}
                if pull_rx in core_pull_cnt[pull_cid].keys():
                    core_pull_cnt[pull_cid][pull_rx] += 1
                else:
                    core_pull_cnt[pull_cid][pull_rx] = 1

        # sorted_core_pull_cnt = sorted(core_pull_cnt.items(), key=lambda x: x[1], reverse=True)

        if 'Core0_4' in bypass_list:
            bypass_list.remove('Core0_4')
        if 'Core3_4' in bypass_list and 'DDRI' in sc_config.keys():
            bypass_list.remove('Core3_4')

        bypass_core = [item for item in bypass_list if item not in valid_core]
        #

        # gen uvm function
        all_config = sc_config
        for core_name in bypass_core:
            if core_name not in sc_config.keys():
                all_config[core_name] = {}

        pe_addr_value = {}
        pe_record_register_value = {}

        for core_name, threads in all_config.items():

            if core_name == 'HOSTI':
                self.record_all_core_all_register_value['HOSTI'] = {}
                # yield f'==================================================== HOSTI ===================================================='
                # yield
                self.chip_config['HOSTI'] = {}
                # yield f'// core schedule and TX configuration'
                self.chip_config['HOSTI']['Core_Sch_TX'] = {}
                register_addr_value_list = self.gen_tx_coresch_register_io_core()
                for (addr, ea, value) in register_addr_value_list:
                    addr = int(addr, 16)
                    addr = convert_hex_to_verilog_hex_32bit(addr)
                    value = convert_hex_to_verilog_hex_64bit(value)
                    self.chip_config['HOSTI']['Core_Sch_TX'][f"[ {addr} , {ea} ]"] = value
                    # yield f"dc_cfg_op(.flag(1), .id('HOST_CORE_ID), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                # yield

                # yield f'// IO configuration'
                register_addr_value_list = self.gen_hosti_register(sc_config, threads)
                self.chip_config['HOSTI']['IO'] = {}
                for (addr, ea, value) in register_addr_value_list:
                    # yield f"dc_cfg_op(.flag(1), .id('HOST_CORE_ID), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                    self.chip_config['HOSTI']['IO'][f"[ {addr} , {ea} ]"] = value
                # yield

            elif core_name == 'DDRI':
                self.record_all_core_all_register_value['DDRI'] = {}
                # yield f'==================================================== DDRI ===================================================='
                # yield
                self.chip_config['DDRI'] = {}
                # yield f'// core DDRI and TX configuration'
                self.chip_config['DDRI']['Core_Sch_TX'] = {}
                register_addr_value_list = self.gen_tx_coresch_register_io_core()
                for (addr, ea, value) in register_addr_value_list:
                    addr = int(addr, 16)
                    addr = convert_hex_to_verilog_hex_32bit(addr)
                    value = convert_hex_to_verilog_hex_64bit(value)
                    self.chip_config['DDRI']['Core_Sch_TX'][f"[ {addr} , {ea} ]"] = value
                    # yield f"dc_cfg_op(.flag(1), .id('HOST_CORE_ID), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                # yield

                # yield f'// IO configuration'
                register_addr_value_list = self.gen_ddri_register(sc_config, threads)
                self.chip_config['DDRI']['IO'] = {}
                for (addr, ea, value) in register_addr_value_list:
                    # yield f"dc_cfg_op(.flag(1), .id('HOST_CORE_ID), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                    self.chip_config['DDRI']['IO'][f"[ {addr} , {ea} ]"] = value
                # yield

            elif core_name in self.valid_core:
                thread_core.append(core_name)

                #     bypass_core.append(core_name)

                #
                if core_name not in self.record_all_core_all_register_value.keys():
                    self.record_all_core_all_register_value[core_name] = {}

                # yield f'// {core_name}'
                # yield
                self.chip_config[f'{core_name}'] = {}

                if core_name not in bypass_core:
                    # yield f'// core schedule and TX configuration'
                    self.chip_config[f'{core_name}']['Core_Sch_TX'] = {}
                    register_addr_value_list = self.gen_tx_coresch_register_cal_core()
                    for (addr, ea, value) in register_addr_value_list:
                        addr = convert_hex_to_verilog_hex_32bit(addr)
                        value = convert_hex_to_verilog_hex_64bit(value)
                        self.chip_config[f'{core_name}']['Core_Sch_TX'][f"[ {addr} , {ea} ]"] = value
                        # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                    # yield

                    # yield f'// virtual channel configuration'
                    self.chip_config[f'{core_name}']['VC'] = {}
                    vc_register = self.gen_vc_register()
                    for (addr, ea, value) in vc_register:
                        # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                        self.chip_config[f'{core_name}']['VC'][f"[ {addr} , {ea} ]"] = value
                    # yield

                mfop_extra_info_addr_obj = reg_cfg_mfop_info_addr()
                mfop_extra_info_addr = mfop_extra_info_addr_obj.to_verilog_hex()
                mfop_extra_info_ea = mfop_extra_info_addr_obj.ea
                mfop_extra_info_value_obj = reg_cfg_mfop_info()
                mfop_sram_base = 0

                cutlength_addr_obj = {'E': reg_tx_cutlen_e_addr(), 'S': reg_tx_cutlen_s_addr(), 'W': reg_tx_cutlen_w_addr(), 'N': reg_tx_cutlen_n_addr()}
                cutlength_value_obj = {'E': reg_tx_cutlen(), 'S': reg_tx_cutlen(), 'W': reg_tx_cutlen(), 'N': reg_tx_cutlen()}
                for k_c,v_c in cutlength_value_obj.items():
                    v_c.pe_cut_length = 3
                    v_c.dmac_cut_length = 3
                    v_c.mfop_cut_length = 3

                cfg_thread = {}
                self.chip_config[f'{core_name}']['OP'] = {}

                for t_id, t_info in threads.items():
                    # yield f"// {t_id}, Task Name: {t_info['Task_Name']}, OP_Code: {t_info['Op_Code']}"
                    op_id = t_info['Op_Code']
                    # register {addr: value}
                    if op_id in ['MaxPooling', 'AvgPooling', 'Upsample']:
                        op_id = 'MFOP'
                    op_register, op_register_value_dict  = self.gen_layer(op_id, core_name, t_info)

                    if op_id == 'PEConv':
                        pe_direction = {0:'W', 1:'N', 2:'E', 3:'S'}
                        pe_dir = pe_direction[t_info['Conv_Struct']['rela_pe']]
                        chip_PE_dir = (t_info['Conv_Struct']['rela_pe'] + 2) % 4 *2
                        pe_tid = str(t_info['Conv_Struct']['pe_tid'])
                        # pe_offset = self.PE_addr_offset[f'{pe_dir}_{pe_tid}']
                        if core_name+'_pe_'+pe_dir not in pe_addr_value.keys():
                            pe_addr_value[core_name+'_pe_'+pe_dir] = {}
                        pe_addr_value[core_name+'_pe_'+pe_dir][pe_tid] = op_register
                        cur_ctl2_value = op_register[6][2]
                        cur_ctl5_value = op_register[7][2]
                        cur_ctl6_value = op_register[8][2]
                        cur_ctl9_value = op_register[9][2]
                        cur_ctl15_value = op_register[11][2]
                        cur_ctl4_value = op_register[14][2]
                        cur_ctl0_value = op_register[17][2]
                        if len(pe_addr_value[core_name+'_pe_'+pe_dir])>1:
                            pre_ctl2_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][6][2]
                            pre_ctl5_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][7][2]
                            pre_ctl6_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][8][2]
                            pre_ctl9_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][9][2]
                            pre_ctl15_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][11][2]
                            pre_ctl4_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][14][2]
                            pre_ctl0_value = pe_addr_value[core_name+'_pe_'+pe_dir][f'{1-int(pe_tid)}'][17][2]
                            new_cur_ctl2_value=format((int(('0x'+cur_ctl2_value[4:20]), 16) | int(('0x'+pre_ctl2_value[4:20]), 16)), '016x')
                            new_cur_ctl5_value=format((int(('0x'+cur_ctl5_value[4:20]), 16) | int(('0x'+pre_ctl5_value[4:20]), 16)), '016x')
                            new_cur_ctl6_value=format((int(('0x'+cur_ctl6_value[4:20]), 16) | int(('0x'+pre_ctl6_value[4:20]), 16)), '016x')
                            new_cur_ctl9_value=format((int(('0x'+cur_ctl9_value[4:20]), 16) | int(('0x'+pre_ctl9_value[4:20]), 16)), '016x')
                            new_cur_ctl15_value=format((int(('0x'+cur_ctl15_value[4:20]), 16) | int(('0x'+pre_ctl15_value[4:20]), 16)), '016x')
                            new_cur_ctl4_value=format((int(('0x'+cur_ctl4_value[4:20]), 16) | int(('0x'+pre_ctl4_value[4:20]), 16)), '016x')
                            new_cur_ctl0_value=format((int(('0x'+cur_ctl0_value[4:20]), 16) | int(('0x'+pre_ctl0_value[4:20]), 16)), '016x')

                        if core_name+'_pe_'+pe_dir not in pe_record_register_value.keys():
                            pe_record_register_value[core_name+'_pe_'+pe_dir] = {}
                        pe_record_register_value[core_name+'_pe_'+pe_dir][pe_tid] = op_register_value_dict
                        #                 op_register_value_dict[pe_regs][signals]=pe_record_register_value[core_name+'_pe_'+pe_dir]['0'][f'{pe_regs[0:19]}{pe_offset-1}'][signals] | pe_record_register_value[core_name+'_pe_'+pe_dir]['1'][pe_regs][signals]
                        #                 op_register_value_dict[pe_regs][signals]=pe_record_register_value[core_name+'_pe_'+pe_dir]['0'][f'{pe_regs[0:18]}{pe_offset-1}'][signals] | pe_record_register_value[core_name+'_pe_'+pe_dir]['1'][pe_regs][signals]
                        #                 op_register_value_dict[pe_regs][signals]=pe_record_register_value[core_name+'_pe_'+pe_dir]['0'][f'{pe_regs[0:18]}{pe_offset-1}'][signals] | pe_record_register_value[core_name+'_pe_'+pe_dir]['1'][pe_regs][signals]

                    for (addr, ea, value) in op_register:
                        # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"

                        if op_id == 'PEConv' and (addr, ea) in [(f'32\'h0000{chip_PE_dir}010', 0), (f'32\'h0000{chip_PE_dir}028', 0), (f'32\'h0000{chip_PE_dir}030', 0), (f'32\'h0000{chip_PE_dir}048', 0), (f'32\'h0000{chip_PE_dir}078', 0), (f'32\'h0000{chip_PE_dir}020', 0), (f'32\'h0000{chip_PE_dir}000', 0)] and len(pe_addr_value[core_name+'_pe_'+pe_dir])>1:
                            pe_ctl_mod={f'32\'h0000{chip_PE_dir}010': f'64\'h{new_cur_ctl2_value}', f'32\'h0000{chip_PE_dir}028': f'64\'h{new_cur_ctl5_value}', f'32\'h0000{chip_PE_dir}030': f'64\'h{new_cur_ctl6_value}', f'32\'h0000{chip_PE_dir}048': f'64\'h{new_cur_ctl9_value}', f'32\'h0000{chip_PE_dir}078': f'64\'h{new_cur_ctl15_value}', f'32\'h0000{chip_PE_dir}020': f'64\'h{new_cur_ctl4_value}', f'32\'h0000{chip_PE_dir}000': f'64\'h{new_cur_ctl0_value}',}
                            self.chip_config[f'{core_name}']['OP'][f"[ {addr} , {ea} ]"] = pe_ctl_mod[addr]
                        else:
                            self.chip_config[f'{core_name}']['OP'][f"[ {addr} , {ea} ]"] = value

                        if op_id == 'PEConv' and (addr, ea) in [(f'32\'h0000{chip_PE_dir}008', 0)] and simu == 1:
                            self.chip_config[f'{core_name}']['OP'][f"[ 32\'h0000{chip_PE_dir}008 , 0 ]"] = "64'h01cf0bb800030080"
                    # yield

                    if op_id == 'PEConv':
                        # pe_direction = {0:'N', 1:'E', 2:'S', 3:'W'}
                        pe_direction = {0:'W', 1:'N', 2:'E', 3:'S'}
                        pe_dir = pe_direction[t_info['Conv_Struct']['rela_pe']]
                        if pe_dir not in cfg_thread.keys():
                            cfg_thread[pe_dir] = []
                        cfg_thread[pe_dir].append(t_info['Task_Name'])
                        if self.cutlength != None and t_info['Task_Name'] in self.cutlength.keys():
                            cutlength_value_obj[pe_dir].pe_cut_length = min(self.cutlength[t_info['Task_Name']], cutlength_value_obj[pe_dir].pe_cut_length)

                    elif op_id == 'DMACConv':
                        if 'DMAC' not in cfg_thread.keys():
                            cfg_thread['DMAC'] = []
                        cfg_thread['DMAC'].append(t_info['Task_Name'])
                    elif op_id == 'MFOP':
                        if 'MFOP' not in cfg_thread.keys():
                            cfg_thread['MFOP'] = []
                        cfg_thread['MFOP'].append(t_info['Task_Name'])
                        mfop_kernel_size = t_info['Conv_Struct']['k_size']
                        mfop_in_channel = t_info['Conv_Struct']['cin']
                        if t_info['Op_Code'] == 'Upsample':
                            mfop_kernel_size = int(t_info['Conv_Struct']['scale_factor'])
                        d_type = '4bit'
                        if d_type == '4bit':
                            # sram_size = min(int(math.ceil(mfop_kernel_size ** (2) * mfop_in_channel * 4 / 256)), 31)
                            sram_size = int(math.ceil(mfop_kernel_size * mfop_in_channel * 4 / 256))
                            channel_group = int(math.ceil(mfop_in_channel * 4 / 256))
                        else:
                            # sram_size = min(int(math.ceil(mfop_kernel_size ** (2) * mfop_in_channel * 8 / 256)), 31)
                            sram_size = int(math.ceil(mfop_kernel_size * mfop_in_channel * 8 / 256))
                            channel_group = int(math.ceil(mfop_in_channel * 8 / 256))

                        if self.mfop_thread_all_core_count[core_name] == 1:
                            mfop_extra_info_value_obj.sram_base_0 = mfop_sram_base
                            mfop_extra_info_value_obj.sram_end_0 = mfop_sram_base + sram_size - 1
                            mfop_sram_base = mfop_sram_base + sram_size
                            mfop_extra_info_value_obj.channel_group_0 = channel_group
                        elif self.mfop_thread_all_core_count[core_name] == 2:
                            mfop_extra_info_value_obj.sram_base_1 = mfop_sram_base
                            mfop_extra_info_value_obj.sram_end_1 = mfop_sram_base + sram_size - 1
                            mfop_sram_base = mfop_sram_base + sram_size
                            mfop_extra_info_value_obj.channel_group_1 = channel_group

                        if mfop_sram_base > 63:
                            raise Warning('MFOP SRAM overflow')

                    elif op_id == 'Transfer':
                        if 'Transfer' not in cfg_thread.keys():
                            cfg_thread['Transfer'] = []
                        cfg_thread['Transfer'].append(t_info['Task_Name'])

                    self.record_all_core_all_register_value[core_name][t_id] = dict(thread_type=t_info['Op_Code'], task_name=t_info['Task_Name'])
                    self.record_all_core_all_register_value[core_name][t_id].update(op_register_value_dict)

                if core_name not in bypass_core:
                    self.chip_config[f'{core_name}']['MFOP_Extra_Info'] = {}
                    self.chip_config[f'{core_name}']['MFOP_Extra_Info'][f"[ {mfop_extra_info_addr} , {mfop_extra_info_ea} ]"] = mfop_extra_info_value_obj.to_verilog_hex()
                    # yield f'// MFOP Extra Info configuration'
                    # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({mfop_extra_info_addr}), .ea({mfop_extra_info_ea}), .d({mfop_extra_info_value_obj.to_verilog_hex()}), .rc(8));"
                    # yield
                    self.record_all_core_all_register_value[core_name]['cfg_mfop_info'] = mfop_extra_info_value_obj.get_field_dict()

                # yield f'// cutlength configuration'
                # self.chip_config[f'{core_name}']['Cutlength'] = {}
                #     cut_addr_obj = cutlength_addr_obj[k_c]
                #     cut_addr = cut_addr_obj.to_verilog_hex()
                #     cut_ea = cut_addr_obj.ea
                #     cut_value = v_c.to_verilog_hex()
                #     # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({cut_addr}), .ea({cut_ea}), .d({cut_value}), .rc(8));"
                #     self.chip_config[f'{core_name}']['Cutlength'][f"[ {cut_addr} , {cut_ea} ]"] = cut_value
                #     self.record_all_core_all_register_value[core_name][f'cfg_cutlen_{k_c}'] = v_c.get_field_dict()
                # yield
                # yield f'// valid configuration'
                self.chip_config[f'{core_name}']['Valid'] = {}
                valid_register, valid_register_value_dict = self.gen_valid_register(cfg_thread, core_name, bypass_core, actf_file)
                for (addr, ea, value) in valid_register:
                    self.chip_config[f'{core_name}']['Valid'][f"[ {addr} , {ea} ]"] = value
                    # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
                # yield
                self.record_all_core_all_register_value[core_name]['valid_config'] = valid_register_value_dict




            #     # yield f'// {core_name}'
            #     # yield
            #     # yield f'// virtual channel configuration'
            #     self.chip_config[f'{core_name}'] = {}
            #     self.chip_config[f'{core_name}']['VC'] = {}
            #     vc_register = self.gen_vc_register()
            #         self.chip_config[f'{core_name}']['VC'][f"[ {addr} , {ea} ]"] = value
            #         # yield f"dc_cfg_op(.flag(1), .id(6'h{int(core_name[4])}{int(core_name[6])}), .op(0), .sn(30), .a({addr}), .ea({ea}), .d({value}), .rc(8));"
            #     # yield

        self.chip_config_sort = {}
        all_cfg_done = {}

        # self.chip_config_sort['Reset']={}
        # self.chip_config_sort['Reset']['Core_cfg_done']={}
        #     self.chip_config_sort['Reset']['Core_cfg_done'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8'] = convert_hex_to_verilog_hex_64bit(int(15))

        # self.chip_config_sort['Reset']['Thread_vld']={}
        #     self.chip_config_sort['Reset']['Thread_vld'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091e0'] = convert_hex_to_verilog_hex_64bit(int(0))
        #     self.chip_config_sort['Reset']['Thread_vld'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}09a00'] = convert_hex_to_verilog_hex_64bit(int(0))

        # self.chip_config_sort['Reset']['DDRI']={}
        # self.chip_config_sort['Reset']['DDRI'][f'64\'h00000008034001e8'] = convert_hex_to_verilog_hex_64bit(int(0))
        # self.chip_config_sort['Reset']['DDRI'][f'64\'h00000008034000b8'] = convert_hex_to_verilog_hex_64bit(int(0))

        # self.chip_config_sort['Reset']['HOSTI']={}
        # self.chip_config_sort['Reset']['HOSTI'][f'64\'h0000000804000628'] = convert_hex_to_verilog_hex_64bit(int(0))
        # self.chip_config_sort['Reset']['HOSTI'][f'64\'h00000008040004f8'] = convert_hex_to_verilog_hex_64bit(int(0))

        for core_name, core_config in self.chip_config.items():
            self.chip_config_sort[core_name] = {}
            for reg_name, reg_config in core_config.items():
                for reg_addr, reg_value in reg_config.items():
                    addr_value = reg_addr.split(' ')
                    # parse addr
                    addr = addr_value[1]
                    # input()
                    chip_addr = self.convert_chip_addr(addr, core_name)
                    ea = addr_value[3]
                    if ea not in self.chip_config_sort[core_name].keys():
                        # self.
                        self.chip_config_sort[core_name][ea] = {}
                        if core_name in ['HOSTI']:
                            self.chip_config_sort[core_name][ea][f'64\'h000000080400e020'] = convert_hex_to_verilog_hex_64bit(int(ea))
                        elif core_name == 'DDRI':
                            self.chip_config_sort[core_name][ea][f'64\'h000000080340e020'] = convert_hex_to_verilog_hex_64bit(int(ea))
                        else:
                            if ea != 'vc_dir' and ea not in ['actf-0', 'actf-1', 'actf-2', 'actf-3', 'DMAC-init']:
                                self.chip_config_sort[core_name][ea][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}0e020'] = convert_hex_to_verilog_hex_64bit(int(ea))
                    self.chip_config_sort[core_name][ea][f'{chip_addr}'] = reg_value
            self.chip_config_sort[core_name] = dict(sorted(self.chip_config_sort[core_name].items(), key=lambda x:x[0], reverse=True))
            # self.chip_config_sort[core_name] = OrderedDict(sorted(self.chip_config_sort[core_name].items(), key=lambda x:x[0], reverse=True))
        # self.chip_config_sort.move_to_end('Reset', last=False)
        self.chip_config_sort['Cfg_done_wo_HOSTI'] = {}
        self.chip_config_sort['Cfg_done_wo_HOSTI']['0'] = {}

        # use_dc = 1
        if use_dc == 0:
            for core_name, threads in all_config.items():
                mfopt_thread_vld = 0
                if core_name not in ['HOSTI', 'Core0_4', 'Run_Time']:
                    if core_name == 'DDRI':
                        self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'64\'h00000008034091d8'] = convert_hex_to_verilog_hex_64bit(int(47))
                        # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h00000008034091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})
                    elif core_name not in bypass_core:
                        for t_id, t_info in threads.items():
                            if 'Op_Code' in t_info.keys() and t_info['Op_Code'] in ['MaxPooling', 'AvgPooling', 'Upsample']:
                                mfopt_thread_vld = 1
                        if mfopt_thread_vld == 1:
                            self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8'] = convert_hex_to_verilog_hex_64bit(int(63))
                            # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(63))}}})
                        else:
                            self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8'] = convert_hex_to_verilog_hex_64bit(int(47))
                            # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})
                    else:
                        self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8'] = convert_hex_to_verilog_hex_64bit(int(47))
                        # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})

            self.chip_config_sort.update({'All_cfg_done': {'0': {f'64\'h00000008004091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})
        else:
            for core_name, threads in all_config.items():
                mfopt_thread_vld = 0
                if core_name not in ['HOSTI', 'Core0_4', 'Run_Time']:
                    if core_name == 'DDRI':
                        ddri_cfg_done_addr = self.convert_dc_addr(f'32\'h000091d8', 0, 'Core3_4')
                        self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'{ddri_cfg_done_addr}'] = convert_hex_to_verilog_hex_64bit(int(47))
                        # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h00000008034091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})
                    elif core_name not in bypass_core:
                        for t_id, t_info in threads.items():
                            if 'Op_Code' in t_info.keys() and t_info['Op_Code'] in ['MaxPooling', 'AvgPooling', 'Upsample']:
                                mfopt_thread_vld = 1
                        if mfopt_thread_vld == 1:
                            cimcore_cfg_done_addr = self.convert_dc_addr(f'32\'h000091d8', 0, core_name)
                            self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'{cimcore_cfg_done_addr}'] = convert_hex_to_verilog_hex_64bit(int(63))
                            # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(63))}}})
                        else:
                            cimcore_cfg_done_addr = self.convert_dc_addr(f'32\'h000091d8', 0, core_name)
                            self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'{cimcore_cfg_done_addr}'] = convert_hex_to_verilog_hex_64bit(int(47))
                            # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})
                    else:
                        cimcore_cfg_done_addr = self.convert_dc_addr(f'32\'h000091d8', 0, core_name)
                        self.chip_config_sort['Cfg_done_wo_HOSTI']['0'][f'{cimcore_cfg_done_addr}'] = convert_hex_to_verilog_hex_64bit(int(47))
                        # self.chip_config_sort.update({'Cfg_done_wo_HOSTI': {'0': {f'64\'h000000080{int(core_name[4])}{int(core_name[6])}091d8': convert_hex_to_verilog_hex_64bit(int(47))}}})

            hosti_cfg_done_addr = self.convert_dc_addr(f'32\'h000091d8', 0, 'Core0_4')
            self.chip_config_sort.update({'All_cfg_done': {'0': {f'{hosti_cfg_done_addr}': convert_hex_to_verilog_hex_64bit(int(47))}}})

        if chip_json_file != None:
            self.make_json(self.chip_config_sort, chip_json_file)
        else:
            chip_json_file = f'{self.module_name}_chip_config.json'
            self.make_json(self.chip_config_sort, chip_json_file)


        if self.dump_register_field_value:
            register_value_json_file = chip_json_file.split('.')[0] + '_register_value.json'
            self.make_json(self.record_all_core_all_register_value, register_value_json_file)

    def convert_dc_addr(self, addr, ea, core_name): #addr=32'h0000xxxx
        dc_addr = 0xffff_8000_0000 + (int(core_name[4]) << 27) + (int(core_name[6]) << 23) + (ea << 16) + int(addr[4:], 16)
        dc_addr = convert_hex_to_verilog_hex_64bit(dc_addr)
        return dc_addr

    def convert_chip_addr(self, addr, core_name):
        if core_name in ['HOSTI']:
            addr = int(addr[4:], 16) + 0x8_0400_0000
            addr = convert_hex_to_verilog_hex_64bit(addr)
        else:
            if core_name == 'DDRI':
                core_name = 'Core3_4'
            addr = int(addr[4:], 16) + 0x8_0000_0000 + (int(core_name[4]) << 24) + (int(core_name[6]) << 20)
            addr = convert_hex_to_verilog_hex_64bit(addr)
        return addr

    def gen_tx_coresch_register_io_core(self):
        register_addr_value_list = [('0x91d8', 0, '0x0'), ('0x8008', 0, '0x10002000100030'), ('0x8010', 0, '0x10004000100050'), ('0x8020', 0, '0x10008000100090'),
                 ('0x8028', 0, '0x1000a0001000b0'), ('0x8038', 0, '0x1000e0001000f0'), ('0x8040', 0, '0x10010000100110'), ('0x8050', 0, '0x10014000100150'),
                 ('0x8058', 0, '0x10016000100170'), ('0x8068', 0, '0x1001a0001001b0'), ('0x8070', 0, '0x1001c0001001d0'), ('0x8080', 0, '0x10020000100210'),
                 ('0x8088', 0, '0x10022000100230'),
                #  ('0x80a8', 0, '0xc040400400404020')
                 ('0x9c08', 0, '0x2850a143060c18'), ('0x91d8', 0, '0xf'),]
        return register_addr_value_list

    def gen_valid_register(self, cfg_thread, core_name, bypass_core, actf_file):
        register_addr_value = []
        thread_record_register_value = {}

        reg_cfg_done_out_valid_addr_obj =  reg_cfg_done_out_merge_addr()
        reg_cfg_done_out_valid_addr = reg_cfg_done_out_valid_addr_obj.to_verilog_hex()
        reg_cfg_done_out_valid_ea = reg_cfg_done_out_valid_addr_obj.ea

        reg_cfg_done_calc_valid_addr_obj =  reg_cfg_done_cal_thread_merge_addr()
        reg_cfg_done_calc_valid_addr = reg_cfg_done_calc_valid_addr_obj.to_verilog_hex()
        reg_cfg_done_calc_valid_ea = reg_cfg_done_calc_valid_addr_obj.ea

        reg_cfg_done_out_valid_value_obj = reg_cfg_done_out_merge()
        reg_cfg_calc_out_valid_value_obj = reg_cfg_done_cal_thread_merge()

        reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_E = 1
        reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_N = 1
        reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_S = 1
        reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_W = 1
        # reg_cfg_done_out_valid_value_obj.cfg_done_mfop = 1
        reg_cfg_done_out_valid_value_obj.cfg_done_dmac = 1

        if 'E' in cfg_thread.keys():
            # reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_E = 1
            reg_cfg_calc_out_valid_value_obj.pet_thread_vld_e0 = 1
            if len(cfg_thread['E']) == 2:
                reg_cfg_calc_out_valid_value_obj.pet_thread_vld_e1 = 1
        if 'N' in cfg_thread.keys():
            # reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_N = 1
            reg_cfg_calc_out_valid_value_obj.pet_thread_vld_n0 = 1
            if len(cfg_thread['N']) == 2:
                reg_cfg_calc_out_valid_value_obj.pet_thread_vld_n1 = 1
        if 'S' in cfg_thread.keys():
            # reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_S = 1
            reg_cfg_calc_out_valid_value_obj.pet_thread_vld_s0 = 1
            if len(cfg_thread['S']) == 2:
                reg_cfg_calc_out_valid_value_obj.pet_thread_vld_s1 = 1
        if 'W' in cfg_thread.keys():
            # reg_cfg_done_out_valid_value_obj.sch2tx_reg_setting_vld_W = 1
            reg_cfg_calc_out_valid_value_obj.pet_thread_vld_w0 = 1
            if len(cfg_thread['W']) == 2:
                reg_cfg_calc_out_valid_value_obj.pet_thread_vld_w1 = 1
        if 'MFOP' in cfg_thread.keys():
            reg_cfg_done_out_valid_value_obj.cfg_done_mfop = 1
            reg_cfg_calc_out_valid_value_obj.mfopt_thread_vld0 = 1
            if len(cfg_thread['MFOP']) == 2:
                reg_cfg_calc_out_valid_value_obj.mfopt_thread_vld1 = 1
        if 'DMAC' in cfg_thread.keys():
            # reg_cfg_done_out_valid_value_obj.cfg_done_dmac = 1
            reg_cfg_calc_out_valid_value_obj.dmact_thread_vld = 1

        reg_transfer_thread_valid_addr_obj =  reg_thread_vld_mask_addr()
        reg_transfer_thread_valid_addr = reg_transfer_thread_valid_addr_obj.to_verilog_hex()
        reg_transfer_thread_valid_ea = reg_transfer_thread_valid_addr_obj.ea
        reg_transfer_thread_valid_value_obj = reg_transfer_thread_valid_mask()
        transfer_length = 0
        if 'Transfer' in cfg_thread.keys():
            transfer_length = len(cfg_thread['Transfer'])
        value_sum = 0
        for i in range(transfer_length):
            value_sum += 2**(i)
        reg_transfer_thread_valid_value_obj.thread_vld_mask = value_sum

        reg_vc_dir_addr_obj = reg_cfg_vc_dir_addr()
        reg_vc_dir_addr = reg_vc_dir_addr_obj.to_verilog_hex()
        reg_vc_dir_ea = reg_vc_dir_addr_obj.ea
        reg_vc_dir_value_obj = reg_cfg_vc_dir()
        if int(core_name[6]) > 4:
            vc_dir = 2
        elif int(core_name[6]) < 4:
            vc_dir = 0
        else:
            vc_dir = 3
        reg_vc_dir_value_obj.vc_dir = vc_dir


        register_addr_value.append((reg_vc_dir_addr, reg_vc_dir_ea, reg_vc_dir_value_obj.to_verilog_hex()))

        if 'Transfer' in cfg_thread.keys() and core_name not in ['Core0_3', 'Core0_5', 'Core3_5', 'Core0_4', 'Core3_4']:
            if actf_file != None:
                with open(actf_file, 'r') as actf:
                    actf_cfg_file = json.load(actf)

            if actf_file == None:
                register_addr_value.append((f'32\'h0000e028', f'actf-0', f'64\'h0000000000000000'))
                register_addr_value.append((f'32\'h0000a008', f'actf-0', f'64\'hfedcba9876543210'))
                register_addr_value.append((f'32\'h0000e028', f'actf-1', f'64\'h0000000000000008'))
                register_addr_value.append((f'32\'h0000a008', f'actf-1', f'64\'hfedcba9876543210'))
                register_addr_value.append((f'32\'h0000e028', f'actf-2', f'64\'h0000000000000010'))
                register_addr_value.append((f'32\'h0000a008', f'actf-2', f'64\'hfedcba9876543210'))
                register_addr_value.append((f'32\'h0000e028', f'actf-3', f'64\'h0000000000000018'))
                register_addr_value.append((f'32\'h0000a008', f'actf-3', f'64\'hfedcba9876543210'))
            elif core_name in actf_cfg_file.keys():
                sign_hex_4bit=['8', '9', 'a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7']
                actf_value_hex = {}
                actf_cfg = {}

                for dir in ['E', 'S', 'W', 'N']:
                    if dir in actf_cfg_file[core_name].keys():
                        actf_value_hex[dir] = [0]*16
                        for i in range (0,16):
                            if i < 8:
                                actf_value_hex[dir][i] = sign_hex_4bit[actf_cfg_file[core_name][dir]['LUT'][i+8]+8]
                            if i >= 8:
                                actf_value_hex[dir][i] = sign_hex_4bit[actf_cfg_file[core_name][dir]['LUT'][i-8]+8]

                dir_ea={
                    'E': 0,
                    'S': 1,
                    'W': 2,
                    'N': 3
                }
                dir_code={
                    'E': '00',
                    'S': '08',
                    'W': '10',
                    'N': '18'
                }
                for dir in ['E', 'S', 'W', 'N']:
                    if dir in actf_value_hex.keys():
                        for i in range (0,16):
                            if dir not in actf_cfg.keys():
                                actf_cfg[dir] = f'64\'h'+str(actf_value_hex[dir][15-i])
                            else:
                                actf_cfg[dir] = str(actf_cfg[dir]) + str(actf_value_hex[dir][15-i])

                        register_addr_value.append((f'32\'h0000e028', f'actf-{dir_ea[dir]}', f'64\'h00000000000000{dir_code[dir]}'))
                        register_addr_value.append((f'32\'h0000a008', f'actf-{dir_ea[dir]}', actf_cfg[dir]))

        if 'DMAC' in cfg_thread.keys() and core_name not in ['Core0_3', 'Core0_5', 'Core3_5', 'Core0_4', 'Core3_4']:
            register_addr_value.append((f'32\'h0000e030', f'DMAC-init', f'64\'h0000000000000000'))

        if 'E' in cfg_thread.keys() or 'S' in cfg_thread.keys() or 'W' in cfg_thread.keys() or 'N' in cfg_thread.keys():
            register_addr_value.append((f'32\'h00000000', reg_vc_dir_ea, f'64\'h0000000000000000'))
            register_addr_value.append((f'32\'h00002000', reg_vc_dir_ea, f'64\'h0000000000000000'))
            register_addr_value.append((f'32\'h00004000', reg_vc_dir_ea, f'64\'h0000000000000000'))
            register_addr_value.append((f'32\'h00006000', reg_vc_dir_ea, f'64\'h0000000000000000'))

        # register_addr_value.append((reg_cfg_done_out_valid_addr, reg_cfg_done_out_valid_ea, reg_cfg_done_out_valid_value_obj.to_verilog_hex()))
        if core_name not in bypass_core:
            register_addr_value.append((reg_cfg_done_calc_valid_addr, reg_cfg_done_calc_valid_ea, reg_cfg_calc_out_valid_value_obj.to_verilog_hex()))
            register_addr_value.append((reg_transfer_thread_valid_addr, reg_transfer_thread_valid_ea, reg_transfer_thread_valid_value_obj.to_verilog_hex()))
        # register_addr_value.append((reg_cfg_done_out_valid_addr, reg_cfg_done_out_valid_ea, reg_cfg_done_out_valid_value_obj.to_verilog_hex()))

        thread_record_register_value['vc_dir'] = reg_vc_dir_value_obj.get_field_dict()
        # thread_record_register_value['cfg_done_out_merge'] = reg_cfg_done_out_valid_value_obj.get_field_dict()
        if core_name not in bypass_core:
            thread_record_register_value['cfg_done_cal_thread_merge'] = reg_cfg_calc_out_valid_value_obj.get_field_dict()
            thread_record_register_value['thread_vld_mask'] = reg_transfer_thread_valid_value_obj.get_field_dict()
        # thread_record_register_value['cfg_done_out_merge'] = reg_cfg_done_out_valid_value_obj.get_field_dict()

        return register_addr_value, thread_record_register_value
