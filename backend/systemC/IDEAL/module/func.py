from .base import BaseModule

class Conv2d(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, bitwise_mode=0, source_list=None,
                 out_mode=0, out_sequence=None, shift=0, adc_gear=0):
        self.op_code = 'Conv'
        super().__init__(node_id, self.op_code , ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, bitwise_mode, source_list, out_mode,
                         out_sequence, shift, adc_gear)

class FC(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, bitwise_mode=0, source_list=None,
                 out_mode=0, out_sequence=None, shift=0, adc_gear=0):
        self.op_code = 'FC'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, bitwise_mode, source_list, out_mode,
                         out_sequence, shift, adc_gear)
