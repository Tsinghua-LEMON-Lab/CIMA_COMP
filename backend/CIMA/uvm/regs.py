from .base_reg import Reg16, Reg32, Reg64

# =============================================================
#                     Register Address
# =============================================================

class reg_slave_addr(Reg32):
    FIELDS = (
        ('default', 3),
        ('bits_offset_0', 8),
        ('bits_offset_1', 2),
        ('bits_offset_2', 3),
    )
    ea = 0

# RatioNumCH10: 0xC060
class reg_dmac_cfg_4_addr(Reg32):
    FIELDS = (
        ('bits_offset_0', 13),
        ('bits_offset_1', 3),
    )

class reg_vc_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 0):
        default_value = dict(bits_offset_1 = 0, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_rec_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 0):
        default_value = dict(bits_offset_1 = 1, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_pe_cfg0_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 0):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_pe_cfg1_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 1):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_pe_cfg2_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 2):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_pe_cfg3_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 3):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_dmac_cfg0_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 32):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_dmac_cfg1_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 33):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_dmac_cfg2_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 34):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_dmac_cfg3_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 35):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_dmac_cfg4_addr(reg_dmac_cfg_4_addr):

    def __init__(self, bits_offset_0 = 0x60):
        default_value = dict(bits_offset_1 = 0x6)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_mfop_cfg0_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 36):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_mfop_cfg1_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 37):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_mfop_cfg2_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 38):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_mfop_cfg3_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 39):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_tx_credit_for_cal_merge_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 44):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_sch_arb_prior_cfg_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 45):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_cfg_mfop_info_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 46):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

# cutlen reg address
class reg_tx_cutlen_e_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 48):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_tx_cutlen_s_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 51):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_tx_cutlen_w_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 54):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_tx_cutlen_n_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 57):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_cfg_done_out_merge_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 59):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_cfg_done_cal_thread_merge_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 60):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

#add vc_dir config
class reg_cfg_vc_dir_addr(reg_slave_addr):
    ea = 'vc_dir'
    def __init__(self, bits_offset_0 = 61):
        default_value = dict(bits_offset_1 = 2, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_common_cfg_calc_data_rsp_info_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 0):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_common_cfg_mcsend_update_rsp_info_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 32):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_thread_vld_mask_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 64):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_sub_array_idx_seg0123_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 65):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_sub_array_idx_seg4567_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 66):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_tx_credit_merge_addr(reg_slave_addr):

    def __init__(self, bits_offset_0 = 129):
        default_value = dict(bits_offset_1 = 3, bits_offset_2 = 4)
        super().__init__(**default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_sub_array_info_addr(reg_slave_addr):
    ea = 3
    def __init__(self, bits_offset_0 = 0, bits_offset_1 = 0):
        default_value = dict( bits_offset_2 = 4)
        super().__init__(ea=self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)
        setattr(self, 'bits_offset_1', bits_offset_1)

class reg_pull_cfg_merge_addr(reg_slave_addr):
    ea = 2
    def __init__(self, bits_offset_0 = 0):
        default_value = dict(bits_offset_1 = 0, bits_offset_2 = 4)
        super().__init__(ea = self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)

class reg_pull_info_0_addr(reg_slave_addr):
    ea = 4
    def __init__(self, bits_offset_0 = 0, bits_offset_1 = 0):
        default_value = dict(bits_offset_2 = 4)
        super().__init__(ea = self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)
        setattr(self, 'bits_offset_1', bits_offset_1)

class reg_pull_info_1_addr(reg_slave_addr):
    ea = 5
    def __init__(self, bits_offset_0 = 0, bits_offset_1 = 0):
        default_value = dict(bits_offset_2 = 4)
        super().__init__(ea = self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)
        setattr(self, 'bits_offset_1', bits_offset_1)

class reg_pull_info_2_addr(reg_slave_addr):
    ea = 6
    def __init__(self, bits_offset_0 = 0, bits_offset_1 = 0):
        default_value = dict(bits_offset_2 = 4)
        super().__init__(ea = self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)
        setattr(self, 'bits_offset_1', bits_offset_1)

class reg_pull_info_3_addr(reg_slave_addr):
    ea = 7
    def __init__(self, bits_offset_0 = 0, bits_offset_1 = 0):
        default_value = dict(bits_offset_2 = 4)
        super().__init__(ea = self.ea, **default_value)
        setattr(self, 'bits_offset_0', bits_offset_0)
        setattr(self, 'bits_offset_1', bits_offset_1)

class reg_IO_offset_addr(Reg32):
    FIELDS = (
        ('offset', 32),
    )
    ea = 0

class reg_hosti_write_and_read_vld_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x4f8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# read dmem base
class reg_hosti_read_dmem_base_3_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x500):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_base_7_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x508):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_base_11_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x510):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_base_15_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x518):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# read dmem size
class reg_hosti_read_dmem_size_3_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x520):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_size_7_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x528):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_size_11_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x530):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_dmem_size_15_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x538):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# hosti read thread flit transmit
class reg_hosti_read_thread_flit_num_1_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset =  0x540):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_3_to_2_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x548):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_5_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x550):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_7_to_6_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x558):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_9_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x560):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_11_to_10_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x568):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_13_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x570):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_read_thread_flit_num_15_to_14_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x578):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# write dmem base
class reg_hosti_write_dmem_base_3_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x580):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_base_7_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x588):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_base_11_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x590):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_base_15_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x598):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# write dmem size
class reg_hosti_write_dmem_size_3_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5a0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_size_7_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset =  0x5a8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_size_11_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5b0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_dmem_size_15_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5b8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# hosti write thread flit number
class reg_hosti_write_thread_flit_num_1_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5c0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_3_to_2_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5c8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_5_to_4_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5d0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_7_to_6_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5d8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_9_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5e0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_11_to_10_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5e8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_13_to_12_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5f0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_write_thread_flit_num_15_to_14_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x5f8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# hosti send data info addr

class reg_hosti_send_dir_and_transid_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x600):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_send_dst_thread_id_7_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x608):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_send_dst_thread_id_15_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x610):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_send_dst_core_id_7_to_0_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x618):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_hosti_send_dst_core_id_15_to_8_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x620):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# # hoist write thread receiev module
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

# # hosti write thread pull cfg merge
#     ea = 2
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

# # hosti write thread pull info appendix
#     ea = 7
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

#     ea = 4
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

#     ea = 5
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

#     ea = 6
#         default_value = dict(offset = offset  )
#         super().__init__(**default_value)

# DRAM memory base and size
class reg_dram_IO_offset_addr(Reg32):
    FIELDS = (
        ('offset', 32),
    )
    ea = 0

class reg_dram_base_id_1_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x038):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_3_to_2_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x040):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_5_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x048):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_7_to_6_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x050):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_9_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x058):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_11_to_10_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x060):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_13_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x068):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_base_id_15_to_14_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x070):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_1_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x078):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_3_to_2_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x080):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_5_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x088):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_7_to_6_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x090):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_9_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x098):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_11_to_10_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0a0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_13_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0a8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_dram_size_id_15_to_14_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0b0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# DDRI valid reg
class reg_ddri_read_and_write_vld_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0b8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# DDRI dmem read base and size
class reg_ddri_read_dmem_base_id_3_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0c0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_base_id_7_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0c8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_base_id_11_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0d0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_base_id_15_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0d8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_size_id_3_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0e0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_size_id_7_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0e8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_size_id_11_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0f0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_dmem_size_id_15_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x0f8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# dmem read thread flit number

class reg_ddri_read_thread_flit_num_1_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x100):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_3_to_2_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x108):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_5_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x110):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_7_to_6_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x118):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_9_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x120):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_11_to_10_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x128):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_13_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x130):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_flit_num_15_to_14_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x138):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)


# DDRI dmem write base and size
class reg_ddri_write_dmem_base_id_3_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x140):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_base_id_7_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x148):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_base_id_11_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x150):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_base_id_15_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x158):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_size_id_3_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x160):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_size_id_7_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x168):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_size_id_11_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x170):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_dmem_size_id_15_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x178):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# ddri write thread flit number

class reg_ddri_write_thread_flit_num_1_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x180):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_3_to_2_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x188):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_5_to_4_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x190):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_7_to_6_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x198):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_9_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1a0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_11_to_10_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1a8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_13_to_12_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1b0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_write_thread_flit_num_15_to_14_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1b8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# ddri read thread send info (adjacent core and direction)
class reg_ddri_read_thread_send_info_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1c0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# ddri read thread dst thread id
class reg_ddri_read_thread_dst_thread_id_7_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1c8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_dst_thread_id_15_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1d0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

# ddri read thread dst core id
class reg_ddri_read_thread_dst_core_id_7_to_0_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1d8):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)

class reg_ddri_read_thread_dst_core_id_15_to_8_addr(reg_dram_IO_offset_addr):
    def __init__(self, offset = 0x1e0):
        default_value = dict(offset = offset  )
        super().__init__(**default_value)


class reg_pe_ctl_addr(Reg32):
    FIELDS = (
        ('offset_0', 13),
        ('offset_1', 3)
    )
    ea = 0

# PE_ctl 0x0008
class reg_pe_ctl_1_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x8, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0010
class reg_pe_ctl_2_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x10, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0018
class reg_pe_ctl_3_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x18, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0028
class reg_pe_ctl_5_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x28, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0030
class reg_pe_ctl_6_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x30, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0048
class reg_pe_ctl_9_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x48, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0078
class reg_pe_ctl_15_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x78, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0088
class reg_pe_ctl_17_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x88, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# Bias_in_ctrl 0x0568
class reg_Bias_in_ctrl_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x568, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_misc 0x0570
class reg_pe_ctl_misc_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x570, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0000
class reg_pe_ctl_0_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x0, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)

# PE_ctl 0x0020
class reg_pe_ctl_4_addr(reg_pe_ctl_addr):
    def __init__(self, offset_0 = 0x20, offset_1 = 0):
        default_value = dict(offset_0 = offset_0, offset_1 = offset_1)
        super().__init__(**default_value)



# =============================================================
#                     Register Value
# =============================================================

# e2s, e2n, e2w, w2s, w2n, w2e, n2s, n2e, n2w, s2e, s2n, s2w
class reg_rqst_vc_dmem(Reg32):
    FIELDS = (
        ('rqst_dmem_base', 16),
        ('rqst_dmem_size', 16)
    )

class reg_resp_vc_dmem(Reg32):
    FIELDS = (
        ('resp_dmem_base', 16),
        ('resp_dmem_size', 16)
    )

class reg_data_vc_dmem(Reg32):
    FIELDS = (
        ('data_dmem_base', 16),
        ('data_dmem_size', 16)
    )

class reg_e2self_rqst_w2self_rqst(Reg64):
    FIELDS = (
        ('w2self_rqst_dmem_base', 16),
        ('w2self_rqst_dmem_size', 16),
        ('e2self_rqst_dmem_base', 16),
        ('e2self_rqst_dmem_size', 16)
    )

class reg_n2self_rqst_s2self_rqst(Reg64):
    FIELDS = (
        ('s2self_rqst_dmem_base', 16),
        ('s2self_rqst_dmem_size', 16),
        ('n2self_rqst_dmem_base', 16),
        ('n2self_rqst_dmem_size', 16)
    )

class reg_credit_init_resp_rqst(Reg64):
    FIELDS = (
        ('crdt_rsp_init_e', 7),
        ('crdt_rsp_init_w', 7),
        ('crdt_rsp_init_s', 7),
        ('crdt_rsp_init_n', 7),
        ('crdt_req_init_e', 7),
        ('crdt_req_init_w', 7),
        ('crdt_req_init_s', 7),
        ('crdt_req_init_n', 7),
        ('default', 8)
    )

class reg_credit_init_data_mid_final(Reg64):
    FIELDS = (
        ('crdt_data_mid_init_e', 7),
        ('crdt_data_mid_init_w', 7),
        ('crdt_data_mid_init_s', 7),
        ('crdt_data_mid_init_n', 7),
        ('crdt_data_final_init_e', 7),
        ('crdt_data_final_init_w', 7),
        ('crdt_data_final_init_s', 7),
        ('crdt_data_final_init_n', 7),
        ('default', 8)
    )

class reg_rec_cfg_merge(Reg64):
    FIELDS = (
        ('rdc_data_type', 1),
        ('flit_num', 4),
        ('rdc_num', 4),
        ('da_num', 4),
        ('default', 51)
    )

class reg_pe_cfg_0(Reg64):
    FIELDS = (
        ('pet_dtid', 7),
        ('pet_dcid', 6),
        ('pet_pe_credit', 6),
        ('pet_cutloc', 3),
        ('pet_sign', 1),
        ('pet_dst_dtype', 2),
        ('pet_src_dtype', 2),
        ('pet_bfr_size', 14),
        ('pet_bfr_base', 14),
        ('default', 9),
    )

class reg_pe_cfg_1(Reg64):
    FIELDS = (
        ('pet_cout', 12),
        ('pet_cin', 12),
        ('pet_knl_size', 4),
        ('pet_idle_th', 12),
        ('pet_row_th', 12),
        ('default', 12),
    )

class reg_pe_cfg_2(Reg64):
    FIELDS = (
        ('pet_wout', 12),
        ('pet_win', 12),
        ('pet_hout', 12),
        ('pet_hin', 12),
        ('pet_stride', 3),
        ('pet_rdcid', 3),
        ('pet_daid', 3),
        ('pet_lut_type', 2),
        ('pet_vcid', 2),
        ('default', 3)
    )

class reg_pe_cfg_3(Reg64):
    FIELDS = (
        ('pet_qt_offset', 8),
        ('pet_qt_shift', 4),
        ('pet_qt_mtply', 4),
        ('pet_frm_num', 32),
        ('pet_pad_rt', 3),
        ('pet_pad_lt', 3),
        ('pet_pad_lo', 3),
        ('pet_pad_up', 3),
        ('default', 4)
    )

class reg_dmac_cfg_0(Reg64):
    FIELDS = (
        ('dmact_dtid', 7),
        ('dmact_dcid', 6),
        ('dmact_cutloc', 3),
        ('dmact_sign', 1),
        ('dmact_dst_dtype', 2),
        ('dmact_src_dtype', 2),
        ('dmact_bfr_size', 14),
        ('dmact_bfr_base', 14),
        ('default', 15),
    )

class reg_dmac_cfg_1(Reg64):
    FIELDS = (
        ('dmact_dir', 2),
        ('dmact_cout', 12),
        ('dmact_cin', 12),
        ('dmact_knl_size', 4),
        ('default', 34),
    )

class reg_dmac_cfg_2(Reg64):
    FIELDS = (
        ('dmact_wout', 12),
        ('dmact_win', 12),
        ('dmact_hout', 12),
        ('dmact_hin', 12),
        ('dmact_stride', 3),
        ('dmact_rdcid', 3),
        ('dmact_daid', 3),
        ('dmact_lut_type', 2),
        ('dmact_vcid', 2),
        ('default', 3)
    )

class reg_dmac_cfg_3(Reg64):
    FIELDS = (
        ('dmact_qt_offset', 8),
        ('dmact_qt_shift', 4),
        ('dmact_qt_mtply', 4),
        ('dmact_frm_num', 32),
        ('dmact_pad_rt', 3),
        ('dmact_pad_lt', 3),
        ('dmact_pad_lo', 3),
        ('dmact_pad_up', 3),
        ('default', 4)
    )

class reg_dmac_cfg_4(Reg64):
    FIELDS = (
        ('data_type_i', 2),
        ('kernel_size_i', 4),
        ('channel_num_i', 10),
        ('stride_i', 2),
        ('mode_entry', 3),
        ('correction_mode', 1),
        ('shift_num_0', 4),
        ('shift_num_1', 5),
        ('tx_dir', 2),
        ('default', 31)
    )

class reg_mfop_cfg_0(Reg64):
    FIELDS = (
        ('mfopt_dtid', 7),
        ('mfopt_dcid', 6),
        ('mfopt_credit', 7),
        ('mfopt_cutloc', 3),
        ('mfopt_sign', 1),
        ('mfopt_dst_dtype', 2),
        ('mfopt_src_dtype', 2),
        ('mfopt_bfr_size', 14),
        ('mfopt_bfr_base', 14),
        ('default', 8),
    )

class reg_mfop_cfg_1(Reg64):
    FIELDS = (
        ('mfopt_dir', 2),
        ('mfopt_cout', 12),
        ('mfopt_cin', 12),
        ('mfopt_knl_size', 4),
        ('mfopt_function', 2),
        ('default', 32),
    )

class reg_mfop_cfg_2(Reg64):
    FIELDS = (
        ('mfopt_wout', 12),
        ('mfopt_win', 12),
        ('mfopt_hout', 12),
        ('mfopt_hin', 12),
        ('mfopt_stride', 3),
        ('mfopt_rdcid', 3),
        ('mfopt_daid', 3),
        ('mfopt_lut_type', 2),
        ('mfopt_vcid', 2),
        ('default', 3)
    )

class reg_mfop_cfg_3(Reg64):
    FIELDS = (
        ('mfopt_qt_offset', 8),
        ('mfopt_qt_shift', 4),
        ('mfopt_qt_mtply', 4),
        ('mfopt_frm_num', 32),
        ('mfopt_pad_rt', 3),
        ('mfopt_pad_lt', 3),
        ('mfopt_pad_lo', 3),
        ('mfopt_pad_up', 3),
        ('default', 4)
    )

class reg_tx_credit_for_cal_merge(Reg64):
    FIELDS = (
        ('mfopt_tx_credit1', 7),
        ('mfopt_tx_credit0', 7),
        ('mfopt_tx_credit', 7),
        ('pet_tx_credit_s', 7),
        ('pet_tx_credit_n', 7),
        ('pet_tx_credit_w', 7),
        ('pet_tx_credit_e', 7),
        ('default', 15)
    )

class reg_sch_arb_prior_cfg(Reg64):
    FIELDS = (
        ('arb_prior_s', 15),
        ('arb_prior_n', 15),
        ('arb_prior_w', 15),
        ('arb_prior_e', 15),
        ('default', 4)
    )

class reg_cfg_mfop_info(Reg64):
    FIELDS = (
        ('sram_base_0', 6),
        ('sram_end_0', 6),
        ('sram_base_1', 6),
        ('sram_end_1', 6),
        ('channel_group_0', 5),
        ('channel_group_1', 5),
        ('default', 30)
    )

class reg_tx_cutlen(Reg64):
    FIELDS = (
        ('pe_cut_length', 3),
        ('dmac_cut_length', 3),
        ('mfop_cut_length', 3),
        ('default', 55)
    )

class reg_common_cfg_calc_data_rsp_info(Reg64):
    FIELDS = (
        ('dmem_base', 14),
        ('cutloc', 3),
        ('data_sign', 1),
        ('src_dtype', 2),
        ('dst_dtype', 2),
        ('act_type', 2),
        ('qt_mtply', 4),
        ('qt_offset', 8),
        ('qt_shift', 4),
        ('default', 24)
    )

class reg_common_mcsend_update_rsp_info(Reg64):
    FIELDS = (
        ('seg_granu', 3),
        ('seg_num', 2),
        ('trans_crdit', 4),
        ('dmem_size', 14),
        ('mc_num', 2),
        ('default', 39)
    )

class reg_cfg_done_out_merge(Reg64):
    FIELDS = (
        ('sch2tx_reg_setting_vld_S', 1),
        ('sch2tx_reg_setting_vld_N', 1),
        ('sch2tx_reg_setting_vld_W', 1),
        ('sch2tx_reg_setting_vld_E', 1),
        ('cfg_done_mfop', 1),
        ('cfg_done_dmac', 1),
        ('default', 58)
    )

class reg_cfg_done_cal_thread_merge(Reg64):
    FIELDS = (
        ('mfopt_thread_vld1', 1),
        ('mfopt_thread_vld0', 1),
        ('dmact_thread_vld', 1),
        ('pet_thread_vld_n1', 1),
        ('pet_thread_vld_n0', 1),
        ('pet_thread_vld_w1', 1),
        ('pet_thread_vld_w0', 1),
        ('pet_thread_vld_s1', 1),
        ('pet_thread_vld_s0', 1),
        ('pet_thread_vld_e1', 1),
        ('pet_thread_vld_e0', 1),
        ('default', 53)
    )

# vc_dir
class reg_cfg_vc_dir(Reg64):
    FIELDS = (
        ('vc_dir', 2),
        ('default', 62)
    )

class reg_transfer_thread_valid_mask(Reg64):
    FIELDS = (
        ('thread_vld_mask', 32),
        ('default', 32)
    )

class reg_sub_array_idx_thread_seg0123(Reg64):
    FIELDS = (
        ('seg_3', 9),
        ('seg_2', 9),
        ('seg_1', 9),
        ('seg_0', 9),
        ('default', 28)
    )

class reg_sub_array_idx_thread_seg4567(Reg64):
    FIELDS = (
        ('seg_7', 9),
        ('seg_6', 9),
        ('seg_5', 9),
        ('seg_4', 9),
        ('default', 28)
    )

class reg_tx_credit_merge(Reg64):
    FIELDS = (
        ('crdt_final_tx_e', 7),
        ('crdt_final_tx_w', 7),
        ('crdt_final_tx_s', 7),
        ('crdt_final_tx_n', 7),
        ('crdt_mid_tx_e', 7),
        ('crdt_mid_tx_w', 7),
        ('crdt_mid_tx_s', 7),
        ('crdt_mid_tx_n', 7),
        ('default', 8)
    )

class reg_sub_array_info(Reg64):
    FIELDS = (
        ('next_dir', 2),
        ('is_last_dst', 1),
        ('rdcid', 3),
        ('daid', 3),
        ('dst_tid', 6),
        ('dst_coreid', 6),
        ('default', 43)
    )

# pull [0-42]
class reg_pull_cfg_merge(Reg64):
    FIELDS = (
        ('dmem_size', 14),
        ('rdc_num', 4),
        ('flit_num', 10),
        ('da_num', 4),
        ('default', 32)
    )

class reg_pull_info(Reg64):
    FIELDS = (
        ('mcid', 3),
        ('segid', 3),
        ('stid', 6),
        ('scid', 6),
        ('default', 46)
    )


# HOSTI core

class reg_hosti_write_and_read_vld(Reg64):
    FIELDS = (
        ('read_id_15_to_0', 16),
        ('write_id_15_to_0', 16)
    )

# read dmem base

class reg_hosti_read_dmem_base_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_base_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_base_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_base_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# read dmem size

class reg_hosti_read_dmem_size_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_size_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_size_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_dmem_size_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# read thread flit num
class reg_hosti_read_thread_flit_num_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_read_thread_flit_num_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

# write dmem base

class reg_hosti_write_dmem_base_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_base_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_base_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_base_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# write dmem size

class reg_hosti_write_dmem_size_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_size_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_size_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_dmem_size_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# write thread flit num
class reg_hosti_write_thread_flit_num_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_write_thread_flit_num_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

# hosti read thread send data info
# send_dir: E:2'b00, S:2'b01, W:2'b10, N:2'b11
# trans_id: if adjacent core is next to hosti core, yes is 1, no is 0
class reg_hosti_read_thread_send_info(Reg64):
    FIELDS = (
        ('send_dir', 32),
        ('trans_id', 16),
        ('default', 16),
    )

class reg_hosti_send_dst_thread_id_7_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_send_dst_thread_id_15_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_send_dst_core_id_7_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_hosti_send_dst_core_id_15_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

# dram memory base and size

class reg_dram_mem_base_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_base_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_dram_mem_size_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri thread valid
class reg_ddri_thread_valid_info(Reg64):
    FIELDS = (
        ('read_thread_vld', 16),
        ('write_thread_vld', 16),
        ('read_only_thread_vld',16),
        ('write_only_thread_vld',16),
    )

# ddri read dmem base
class reg_ddri_read_thread_dmem_base_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_base_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_base_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_base_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri read dmem size
class reg_ddri_read_thread_dmem_size_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_size_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_size_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dmem_size_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri read thread flit number
class reg_ddri_read_thread_flit_num_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_flit_num_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri write thread base and size
class reg_ddri_write_thread_dmem_base_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_base_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_base_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_base_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_size_3_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_size_7_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_size_11_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_dmem_size_15_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri write thread flit number
class reg_ddri_write_thread_flit_num_1_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_3_to_2(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_5_to_4(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_7_to_6(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_9_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_11_to_10(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_13_to_12(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_write_thread_flit_num_15_to_14(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri read thread send info
# send_dir: E:2'b00, S:2'b01, W:2'b10, N:2'b11
# trans_id: if adjacent core is next to hosti core, yes is 1, no is 0
class reg_ddri_read_thread_send_info(Reg64):
    FIELDS = (
        ('send_dir', 32),
        ('trans_id', 16),
        ('default', 16),
    )

# ddri read thread dst thread id
class reg_ddri_read_thread_dst_thread_id_7_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dst_thread_id_15_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

# ddri read thread dst core id
class reg_ddri_read_thread_dst_core_id_7_to_0(Reg64):
    FIELDS = (
        ('value', 64),
    )

class reg_ddri_read_thread_dst_core_id_15_to_8(Reg64):
    FIELDS = (
        ('value', 64),
    )

# PE_ctl 0x0008
class reg_pe_ctl_1(Reg64):
    FIELDS = (
        ('read_times', 3),
        ('read_col_num', 3),
        ('cal_wait_cycle', 8),
        ('cal_cycle', 10),
        ('cal_link_1st_wait_cycle', 8),
        ('az_wait_cycle', 14),
        ('az_pulse_cycle', 8),
        ('verify_adc_range', 3),
        ('default', 7)
    )

# PE_ctl 0x0010
class reg_pe_ctl_2(Reg64):
    FIELDS = (
        ('bias0_en', 2),
        ('bias0_line_num_h', 2),
        ('bias0_line_num_l', 2),
        ('bias0_h0', 4),
        ('bias0_h1', 4),
        ('bias0_l0', 4),
        ('bias0_l1', 4),
        ('bias1_en', 2),
        ('bias1_line_num_h', 2),
        ('bias1_line_num_l', 2),
        ('bias1_h0', 4),
        ('bias1_h1', 4),
        ('bias1_l0', 4),
        ('bias1_l1', 4),
        ('cal_1st_cycle', 10),
        ('default', 10)
    )

# PE_ctl 0x0018
class reg_pe_ctl_3(Reg64):
    FIELDS = (
        ('default_0', 30),
        ('sel_144_mode_en0', 1),
        ('sel144_acc_en0', 1),
        ('sel_144_cfg0', 4),
        ('sel_144_mode_en1', 1),
        ('sel144_acc_en1', 1),
        ('sel_144_cfg1', 4),
        ('az_wait0_cycle', 8),
        ('az_wait1_cycle', 8),
        ('wl_en_l_cycle', 2),
        ('default_1', 4)
    )

# PE_ctl 0x0028
class reg_pe_ctl_5(Reg64):
    FIELDS = (
        ('buf0_thread_coef_num', 6),
        ('buf1_thread_coef_num', 6),
        ('buf2_thread_coef_num', 6),
        ('buf3_thread_coef_num', 6),
        ('thread_coef0_sel_link', 6),
        ('thread_coef1_sel_link', 6),
        ('thread_coef2_sel_link', 6),
        ('thread_coef3_sel_link', 6),
        ('thread_coef4_sel_link', 6),
        ('thread_coef5_sel_link', 6),
        ('default', 4)
    )

# PE_ctl 0x0030
class reg_pe_ctl_6(Reg64):
    FIELDS = (
        ('thread_coef6_sel_link', 6),
        ('thread_coef7_sel_link', 6),
        ('thread_coef8_sel_link', 6),
        ('thread_coef9_sel_link', 6),
        ('thread_coef10_sel_link', 6),
        ('thread_coef11_sel_link', 6),
        ('thread_coef12_sel_link', 6),
        ('thread_coef13_sel_link', 6),
        ('thread_coef14_sel_link', 6),
        ('thread_coef15_sel_link', 6),
        ('default', 4)
    )

# PE_ctl 0x0048
class reg_pe_ctl_9(Reg64):
    FIELDS = (
        ('bl_init_rst', 8),
        ('bl_step_rst', 8),
        ('bl_limit_rst', 8),
        ('wl_init_rst', 8),
        ('wl_step_rst', 8),
        ('wl_limit_rst', 8),
        ('adc_range0', 3),
        ('adc_range1', 3),
        ('default', 10)
    )

# pe_ctl_15 0x0078
class reg_pe_ctl_15(Reg64):
    FIELDS = (
        ('kernel_size_0', 2),
        ('channel_i_0', 3),
        ('channel_o_0', 3),
        ('stride_0', 1),
        ('data_type0', 1),
        ('kernel_valid0', 1),
        ('kernel_size_1', 2),
        ('channel_i_1', 3),
        ('channel_o_1', 3),
        ('stride_1', 1),
        ('data_type1', 1),
        ('kernel_valid1', 1),
        ('ch64_o_adc_sel_0', 1),
        ('ch64_o_adc_sel_1', 1),
        ('default', 36)
    )

# pe_ctl_17 0x0088
class reg_pe_ctl_17(Reg64):
    FIELDS = (
        ('d2a_config_0', 32),
        ('d2a_config_1', 32)
    )

# Bias_in_ctrl 0x0568
class reg_Bias_in_ctrl(Reg64):
    FIELDS = (
        ('bias0_h2', 4),
        ('bias0_h3', 4),
        ('bias0_l2', 4),
        ('bias0_l3', 4),
        ('bias1_h2', 4),
        ('bias1_h3', 4),
        ('bias1_l2', 4),
        ('bias1_l3', 4),
        ('default', 32)
    )

# pe_misc 0x0570
class reg_pe_ctl_misc(Reg64):
    FIELDS = (
        ('wl_close', 1),
        ('coef_ADC_mode', 1),
        ('blgrp_sel_mode', 1),
        ('blgrp_sel', 6),
        ('default_0', 3),
        ('pd_mode', 1),
        ('ckaz_high_en', 1),
        ('prog_mode', 4),
        ('default_1', 46)
    )

class reg_pe_ctl_0(Reg64):
    FIELDS = (
        ('form_en', 1),
        ('prog_en', 1),
        ('read_en', 1),
        ('cal_en', 1),
        ('az_en', 1),
        ('az_sw_en', 1),
        ('default_0', 4),
        ('ss_force_rst_n', 1),
        ('ss_adj_set_p', 1),
        ('ss_adj_set_n', 1),
        ('ss_adj_rst_p', 1),
        ('ss_adj_rst_n', 1),
        ('ss_verify_p', 1),
        ('ss_verify_n', 1),
        ('ss_verify_both', 1),
        ('adc_inv_en', 1),
        ('all16th_acc_en', 1),
        ('default_1', 2),
        ('pekb_en0', 1),
        ('pekb_en1', 1),
        ('default_2', 2),
        ('addr', 22),
        ('prog_mask', 8),
        ('first_az', 1),
        ('default_3', 7)
    )

class reg_pe_ctl_4(Reg64):
    FIELDS = (
        ('default_0', 1),
        ('shift0_num', 4),
        ('default_1', 2),
        ('shift1_num', 4),
        ('default_2', 9),
        ('dout_type0', 2),
        ('dout_type1', 2),
        ('default_3', 4),
        ('wl_verify_cal_voltage', 8),
        ('bl_verify_cal_voltage', 8),
        ('verify_bl_data', 4),
        ('change_adc_range_cycle', 14),
        ('read_dout_type', 1),
        ('default_4', 1)
    )



