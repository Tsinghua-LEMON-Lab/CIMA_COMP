

class BaseModule:

    def __init__(self, op_id=0, target_id=0, array_id=0, adc_select='0000000000000000', array_row_start=0,
                 relu=0, bias_addr_offset=0, bias_en=0, stride=0, padding=0,
                 out_channel = 0, in_channel =0 , kernel_size = 0, out_feature_map_col=0,
                 out_feature_map_row=0, in_feature_map_col = 0, in_feature_map_row=0,
                 func_number=2,func_mode='e'):
        '''
        '''
        self.op_id = op_id
        self.target_id = target_id
        self.array_id = array_id
        self.adc_select = adc_select
        self.array_row_start = array_row_start
        self.relu = relu
        self.bias_addr_offset = bias_addr_offset
        self.bias_en = bias_en
        self.stride = stride
        self.padding = padding
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.out_feature_map_col = out_feature_map_col
        self.out_feature_map_row = out_feature_map_row
        self.in_feature_map_col = in_feature_map_col
        self.in_feature_map_row = in_feature_map_row
        self.func_number = func_number
        self.func_mode = func_mode

    def gen_code(self):

        t1 = f'data_mem[{self.op_id}][231:0] <= '
        t2 = '{'
        t3 = f"1'd0,6'd{self.target_id},10'd0,3'd0,2'd{self.array_id},64'h{self.adc_select},10'd{self.array_row_start},8'd0,"
        t4 = f"3'd0,3'd0,8'd0,8'd0,5'd1,1'd{self.relu},7'd31,1'd0,16'd{self.bias_addr_offset},1'd{self.bias_en},16'd0,2'd{self.stride},"
        t5 = f"2'd{self.padding},6'd{self.out_channel},6'd{self.in_channel},3'd{self.kernel_size},8'd{self.out_feature_map_col},"
        t6 = f"8'd{self.out_feature_map_row},8'd{self.in_feature_map_col},8'd{self.in_feature_map_row},8'h{self.func_number}{self.func_mode}"
        t7 = '};'

        return t1 + t2 + t3 + t4 + t5 + t6 + t7


