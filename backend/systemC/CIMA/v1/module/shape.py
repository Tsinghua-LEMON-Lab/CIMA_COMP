from .base import BaseModule

class Concat(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_sequence=None, in_linebuffer_width = [0, 0],
                 credit_len = 0,relu=0,  len = 0,):
        self.op_code = 'Concat'
        super().__init__(task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_sequence = out_sequence,
                         in_linebuffer_width = in_linebuffer_width,
                         credit_len = credit_len, len = len,relu=relu, valid=True)


class Split(BaseModule):

    def __init__(self, *, task_id=0, ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_sequence=None, in_linebuffer_width = [0, 0],
                 credit_len = 0,relu=0,  len = 0,):
        self.op_code = 'Split'
        super().__init__(task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_sequence = out_sequence,
                         in_linebuffer_width = in_linebuffer_width,
                         credit_len = credit_len, len = len,relu=relu, valid=True)
