from .base import BaseModule

class Concat(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_mode=0, out_sequence=None,):
        self.op_code = 'Concat'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_mode= out_mode, out_sequence = out_sequence)

class Resize(BaseModule):

    def __init__(self, *, node_id=0, ifm_row=0,
                 ifm_col=0, ifm_channel=0, ofm_channel=0,
                 source_list=None, out_mode=0, out_sequence=None,):
        self.op_code = 'Resize'
        super().__init__(node_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = source_list, out_mode= out_mode, out_sequence = out_sequence)
