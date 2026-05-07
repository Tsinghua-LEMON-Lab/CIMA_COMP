from .base import BaseModule

class Conv2d(BaseModule):

    def __init__(self, *, op_id=0, target_id=0, array_id=0, adc_select='0000000000000000', array_row_start=0,
                 relu=0, bias_addr_offset=0, bias_en=0, stride=0, padding=0,
                 out_channel = 0, in_channel =0 , kernel_size = 0, out_feature_map_col=0,
                 out_feature_map_row=0, in_feature_map_col = 0, in_feature_map_row=0,
                 func_number=2, func_mode='e'):

        super().__init__(op_id, target_id, array_id, adc_select, array_row_start,
                        relu, bias_addr_offset, bias_en, stride, padding,
                        out_channel, in_channel, kernel_size,  out_feature_map_col,
                        out_feature_map_row, in_feature_map_col, in_feature_map_row,
                        func_number, func_mode)

class FC(BaseModule):

    def __init__(self, *, op_id=0, target_id=0, array_id=0, adc_select='0000000000000000', array_row_start=0,
                 relu=0, bias_addr_offset=0, bias_en=0, stride=0, padding=0,
                 out_channel = 0, in_channel =0 , kernel_size = 0, out_feature_map_col=1,
                 out_feature_map_row=0, in_feature_map_col = 1, in_feature_map_row=0,
                 func_number=3, func_mode='b'):

        super().__init__(op_id, target_id, array_id, adc_select, array_row_start,
                        relu, bias_addr_offset, bias_en, stride, padding,
                        out_channel, in_channel, kernel_size,  out_feature_map_col,
                        out_feature_map_row, in_feature_map_col, in_feature_map_row,
                        func_number, func_mode)

class Pool(BaseModule):
    def __init__(self, *, op_id=0, target_id=0, array_id=0, adc_select='0000000000000000', array_row_start=0,
                 relu=1, bias_addr_offset=0, bias_en=0, stride=0, padding=0,
                 out_channel = 0, in_channel =0 , kernel_size = 0, out_feature_map_col=0,
                 out_feature_map_row=0, in_feature_map_col = 0, in_feature_map_row=0,
                 func_number=4, func_mode='e'):

        super().__init__(op_id, target_id, array_id, adc_select, array_row_start,
                        relu, bias_addr_offset, bias_en, stride, padding,
                        out_channel, in_channel, kernel_size,  out_feature_map_col,
                        out_feature_map_row, in_feature_map_col, in_feature_map_row,
                        func_number, func_mode)
