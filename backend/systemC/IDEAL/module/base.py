

class BaseModule:

    def __init__(self, node_id=0, op_code=0, ifm_row=0, ifm_col=0, ifm_channel=0, ofm_channel=0,
                    stride=0, padding=0, kernel_size=0, bitwise_mode=0, source_list=None, out_mode=0,
                    out_sequence=None, shift=0, adc_gear=0):
        '''
        '''
        self.node_id = node_id
        self.op_code = op_code
        self.ifm_row = ifm_row
        self.ifm_col = ifm_col
        self.ifm_channel = ifm_channel
        self.ofm_channel = ofm_channel
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.bitwise_mode = bitwise_mode
        self.source_list = source_list
        self.out_mode = out_mode
        self.out_sequence = out_sequence
        self.shift = shift
        self.adc_gear = adc_gear
        self.match_mode_sequence()

    def match_mode_sequence(self):
        if self.out_mode == 0:
            self.out_mode_str = "{}"
            self.out_sequence_str = "{}"
        else:

            assert len(self.out_mode) == len(self.out_sequence)
            self.out_mode_str = "{"
            self.out_sequence_str = "{"
            for i in range(len(self.out_mode)):
                self.out_mode_str += str(self.out_mode[i])
                temp_ = "{"
                for j in range(len(self.out_sequence[i])):
                    temp_ += str(self.out_sequence[i][j])
                    temp_ += ","
                temp_ += "}"
                self.out_sequence_str += temp_
                self.out_sequence_str += ","

                self.out_mode_str += ","
            self.out_mode_str += "}"
            self.out_sequence_str += "}"

    def gen_code(self):

        t1 = "{     " + f"{self.node_id}  ,     {self.op_code} ,     {self.ifm_col}  ,        {self.ifm_row}  ,      {self.ifm_channel}  ,      {self.ofm_channel}  ,      {self.kernel_size}  ,      {self.stride}  ,      {self.padding}  ,      {self.bitwise_mode}  ,"
        t2 = "      {"
        for i in self.source_list:
            t2 += f"{i}" + ","
        t3 = "},"
        t4 = f"     {self.out_mode_str}   ,     {self.out_sequence_str}  ,     {self.shift}   ,        {self.adc_gear}" + '},'
        return t1 + t2 + t3 + t4


