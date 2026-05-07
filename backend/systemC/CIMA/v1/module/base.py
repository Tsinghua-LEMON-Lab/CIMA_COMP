class BaseModule:

    def __init__(self, task_id=0, op_code=0, ifm_row=0, ifm_col=0, ifm_channel=0, ofm_channel=0,
                stride=0, padding=0, kernel_size=0, bitwise_mode=0, source_list=[],
                out_sequence=[],  in_linebuffer_width = [0, 0], credit_len = 0, pe_index= 'NA',
                valid = False, relu = 0, len = 0):

        self.task_id = task_id
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
        self.out_sequence = out_sequence
        self.in_linebuffer_width = in_linebuffer_width
        self.credit_len = credit_len
        self.pe_index = pe_index
        self.valid = valid
        self.relu = relu
        # self.len = len


    def gen_code(self):

        if self.valid :
            line1 = '       1, \n'
            line2 = f'       {self.task_id}, pair<int,int>({self.in_linebuffer_width[0]}, {self.in_linebuffer_width[1]}), {self.credit_len}, \n'
            line3 = '       {\n'
            line4 = ''
            for sr in self.source_list:
                line4 += '         Src_struct{('
                # pair<int,int>(0,64)
                line4 += f'{sr[0]} * MESH_COL+{sr[1]}) * CORE_TASK_LIMIT + {sr[2]}, pair<int,int>(0, 64), 0'
                line4 += '}, \n'

            line5 = '       }, \n'
            line6 = f'       {self.op_code}, \n'
            line7 = '       {\n         Conv_struct{'
            line8 = f'{self.ifm_col}, {self.ifm_channel}, {self.ofm_channel}, {self.kernel_size}, {self.stride}, {self.padding}, {self.pe_index}, {self.bitwise_mode}, {self.relu}'
            line9 = '},\n       }, \n'
            line10 = '       {\n'
            line11 = ''
            for dr in self.out_sequence:
                line11 += f'         pair<int,int>(({dr[0]} * MESH_COL + {dr[1]}) * CORE_TASK_LIMIT + {dr[2]}, {dr[3]}), \n'

            line12 = '       },\n'
            # line13 = '       }'
            return line1 + line2 + line3 + line4 + line5 + line6 + \
                    line7 + line8 + line9 + line10 + line11 + line12
        else:
            return '        0, \n'


