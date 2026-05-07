from .base import BaseModule

class AvgPool(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, source_list=None, in_linebuffer_width = [0, 0],
                 credit_len = 0, out_sequence=None, len=0):
        self.op_code = 'AvgPool'
        super().__init__(task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size,  source_list = source_list,
                         in_linebuffer_width = in_linebuffer_width, credit_len = credit_len,
                         out_sequence = out_sequence, len=len ,valid=1)

class MaxPool(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, source_list=None,
                 in_linebuffer_width = [0, 0], credit_len = 0,  len=0,
                 out_sequence=None):
        self.op_code = 'MaxPool'
        super().__init__(task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, source_list = source_list,in_linebuffer_width = in_linebuffer_width,
                         credit_len = credit_len, len=len, out_sequence = out_sequence,valid=1)
