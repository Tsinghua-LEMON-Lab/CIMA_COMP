from .base import BaseModule

class Conv2d(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, bitwise_mode=0, source_list=None,
                 out_sequence=None, in_linebuffer_width = [0, 0],
                 credit_len = 0, pe_index=None, valid = True, relu = 0, len = 0):
        self.op_code = 'R_MVM'
        super().__init__(task_id, self.op_code , ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, bitwise_mode, source_list,
                         out_sequence, in_linebuffer_width = in_linebuffer_width,
                         credit_len = credit_len, pe_index=pe_index,valid = valid , relu = relu,
                         len = len)

class FC(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, bitwise_mode=0, source_list=None,
                  out_sequence=None, in_linebuffer_width = [0, 0],
                 credit_len = 0, pe_index=None, valid = True,  relu = 0, len = 0):
        self.op_code = 'R_MVM'
        super().__init__(task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, bitwise_mode, source_list,
                         out_sequence, in_linebuffer_width = in_linebuffer_width,
                         credit_len = credit_len, pe_index=pe_index, valid = valid, relu = relu,
                         len = len)
