from CIMA.uvm.regs import *

# a = reg_credit_init_data_mid_final()
# a.crdt_data_mid_init_s = 10
# b = hex(a.bits)
# c = a.to_verilog_hex()


# a1 = reg_vc_addr()
# b = a1.to_verilog_hex()

# a2 = reg_sub_array_info_addr()
# c = a2.to_verilog_hex()

# a2 = reg_slave_addr()
# a2.bits_offset_0 = 2
# a2.bits_offset_1 = 3
# a2.bits_offset_2 = 4
# c = a2.get_field_dict()

# b = {}
# b[1] = c

class reg_IO_offset_addr(Reg64):
    FIELDS = (
        ('offset', 64),
    )
    ea = 0
    initial_addr = 0x4000000

class reg_hosti_write_and_read_vld_addr(reg_IO_offset_addr):
    def __init__(self, offset = 0x4f8):
        default_value = dict(offset = offset + self.initial_addr)
        super().__init__(**default_value)

a = reg_hosti_write_and_read_vld_addr()
b = a.to_verilog_hex()
print(b)


