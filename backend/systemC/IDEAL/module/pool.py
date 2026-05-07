from .base import BaseModule

class AvgPool(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, source_list=None,
                 out_mode=0, out_sequence=None,):
        self.op_code = 'Avgpool'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size,  source_list = source_list, out_mode= out_mode,
                         out_sequence = out_sequence)

class MaxPool(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0, ifm_col=0,
                 ifm_channel=0, ofm_channel=0, stride=0, padding=0,
                 kernel_size=0, source_list=None,
                 out_mode=0, out_sequence=None):
        self.op_code = 'Maxpool'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         stride, padding, kernel_size, source_list = source_list, out_mode = out_mode,
                         out_sequence = out_sequence)
