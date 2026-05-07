from platform import node
from .base import BaseModule

class InBuffer(BaseModule):

    def __init__(self, layer_obj, index):
        self.layer_obj = layer_obj
        self.node_id = index
        self.op_code = 'InBuffer'
        self.source_list= [0]
        self.parse_para()

    def parse_para(self):
        input_ = self.layer_obj.inputs
        img_ = input_[self.task_id]
        ifm_row = img_.height
        ifm_col = img_.width
        ifm_channel = img_.channel
        super().__init__(self.task_id, self.op_code, ifm_row, ifm_col, ifm_channel,
                         source_list = self.source_list)

class OutBuffer(BaseModule):

    def __init__(self, *, task_id, layer_obj, source_list, in_linebuffer_width = [0,0],
                        credit_len = 0):
        self.op_code = 'R_MVM'
        self.layer_obj = layer_obj
        self.task_id = task_id
        self.source_list = source_list
        self.in_linebuffer_width = in_linebuffer_width
        self.credit_len = credit_len
        self.parse_para()

    def parse_para(self):
        input_ = self.layer_obj.inputs
        if len(input_) > 1:
            raise ValueError(f"Message translated to English.")
        img_ = input_[0]
        ifm_row = img_.height
        ifm_col = img_.width
        ifm_channel = img_.channel
        ofm_channel = img_.channel
        super().__init__(self.task_id, self.op_code, ifm_row, ifm_col, ifm_channel, ofm_channel,
                         source_list = self.source_list, in_linebuffer_width = self.in_linebuffer_width,
                         credit_len = self.credit_len, valid=1)
