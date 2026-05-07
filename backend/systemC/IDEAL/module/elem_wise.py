from re import L
from .base import BaseModule

class Add(BaseModule):

    def __init__(self, *, node_id=0,  ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_mode=0, out_sequence=None):
        self.op_code = 'Add'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_mode= out_mode, out_sequence = out_sequence)

class Relu(BaseModule):

    def __init__(self, *, node_id=0,  ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_mode=0, out_sequence=None,):
        self.op_code = 'Relu'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_mode= out_mode, out_sequence = out_sequence)

class Avg(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_mode=0, out_sequence=None,):
        self.op_code = 'Avg'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_mode= out_mode, out_sequence = out_sequence)
