from devices.rram import *

class CIS(RramDevice):

    kind = 'rram-cis'

    profile = {
        'in_channel': 576,   # 1152/2
        'out_channel': 512,  # 512
        'in_bits': 2,        # [-1, 1] sint2
        'out_bits': 4,       # [-7, 7] sint4
        'weight_bits': 4,    # [-7, 7] sint4
        'adc_num':64,
        'signed': True,
    }
