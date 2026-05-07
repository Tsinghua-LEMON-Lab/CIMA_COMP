# CIMA device abstractions
from irtool.core.layer import OpLayer
from irtool.core.ref import is_valid_ref
from irtool.core.jsonable import to_json_obj, Jsonable
from irtool.core.type_util import is_integers, is_integer, mixin, \
                                       to_int_tuple, to_cls_obj
from irtool import BaseDevice
from irtool.devices import RramDevice, RramProfile
from itertools import product

class CIMADeviceMappingInfo(Jsonable):

    index = None
    device = None
    address = None

    def __init__(self, *, index, device, address=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('index', to_int_tuple(index, keep_scalar=True),
                      is_integers, not_none=True, min_val=0, ndims=[3])
        # self.set_attr('device', device, is_valid_ref, not_none=True)
        self.set_attr('device', device)
        self.set_attr('address', to_int_tuple(address, keep_scalar=True),
                      is_integers, min_val=0, ndims=[4])


class CIMAMappingInfo(Jsonable):

    '''
    Mapping configuration:
        - col_split_num / row_split_num:
            Total split factors along column/row.
        - col_repeat_num / row_repeat_num:
            Replication factors along column/row.
        - para_diff_array:
            Parallelism by duplicating weights across different arrays.
    '''

    col_split_num = 1
    row_split_num = 1
    col_repeat_num = 1
    row_repeat_num = 1
    para_diff_array = 1
    in_line_buffer_addr = None
    credit_len = None
    # output_tile_buffer_addr = None
    # in_buf_type = 0
    # out_buf_type = 0

    mappings = None

    def __init__(self, *, col_split_num = 1,  row_split_num = 1,
                 col_repeat_num = 1, row_repeat_num = 1, para_diff_array = 1,
                 in_line_buffer_addr = None, credit_len=0,
                #  output_tile_buffer_addr = None , in_buf_type = 0, out_buf_type =0,
                 mappings = None, **kwargs):

        self.set_attr('col_split_num', col_split_num, is_integer, min_val=1)
        self.set_attr('row_split_num', row_split_num, is_integer, min_val=1)
        self.set_attr('col_repeat_num', col_repeat_num, is_integer, min_val=1)
        self.set_attr('row_repeat_num', row_repeat_num, is_integer, min_val=1)
        self.set_attr('para_diff_array', para_diff_array, is_integer, min_val=1)
        self.set_attr('in_line_buffer_addr', in_line_buffer_addr)
        self.set_attr('credit_len', credit_len)
        # self.set_attr('output_tile_buffer_addr', output_tile_buffer_addr)
        # self.set_attr('in_buf_type', in_buf_type, is_integer, min_val=0)
        # self.set_attr('out_buf_type', out_buf_type, is_integer, min_val=0)

        if mappings is not None:
            if isinstance(mappings, (tuple, list)):
                d = {}
                for obj in mappings:
                    obj = to_cls_obj(obj, cls=CIMADeviceMappingInfo)
                    d[obj.index] = obj
                # d = {d.index: d for d in
                #      map(DeviceMappingInfo.from_json_obj, mappings)}
                assert len(d) == len(mappings)
                mappings = d
            assert tuple(sorted(mappings)) == tuple(product(
                    range(self.para_diff_array), range(self.row_split_num), range(self.col_split_num)))
            self.mappings = mappings


    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        if self.mappings is not None:
            d['mappings'] = list(self.mappings.values())
        return to_json_obj(d, **kwargs)

class CIMACalcInfo(Jsonable):

    ADC_quant_level = 0
    scale_shift_num = [0]
    scale = [1]
    offset = [0]
    accumulate_shift_num = 0
    data_type = '4bit'
    scale_first = True

    def __init__(self, *, ADC_quant_level = 0, scale_shift_num = [0],
                          scale = [1], offset = [0], accumulate_shift_num = 0,
                          data_type = '4bit', scale_first = True,
                          **kwargs):
        self.set_attr('ADC_quant_level', ADC_quant_level)
        self.set_attr('scale_shift_num', scale_shift_num)
        self.set_attr('scale', scale)
        self.set_attr('offset', offset)
        self.set_attr('accumulate_shift_num', accumulate_shift_num)
        self.set_attr('data_type', data_type)
        self.set_attr('scale_first', scale_first)

    def to_json_obj(self, **kwargs):
        d = dict(self.__dict__)
        return to_json_obj(d, **kwargs)

@mixin(OpLayer)
class MappedLayer(Jsonable):

    CIMA_mapping_info = None
    CIMA_calc_info = None

    def __init__(self, *, CIMA_mapping_info=None, CIMA_calc_info=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('CIMA_mapping_info', to_cls_obj(CIMA_mapping_info, CIMAMappingInfo))
        self.set_attr('CIMA_calc_info', to_cls_obj(CIMA_calc_info, CIMACalcInfo))


class CIMAXbDevice(RramDevice):

    kind = 'cima-xb'

    profile = dict(
        in_channel = 576,
        out_channel = 128,
        in_bits = 8,          # ?
        out_bits = 8,         # ?
        weight_bits = 4,      # ?
        signed = True,
    )

class CIMAPEClusterDevice(BaseDevice):

    kind = 'cima-pe-cluster'

    devices = {
        'xb':{
            'kind': 'cima-xb',
            'number': 16
        }
    }


#     kind = 'cima-adder'


class CIMADRAMDevice(BaseDevice):

    kind = 'cima-dram'

class CIMADMACDevice(RramDevice):

    kind = 'cima-dmac'

    profile = dict(
        in_channel = 256,
        out_channel = 64,
        in_bits = 8,          # ?
        out_bits = 8,         # ?
        weight_bits = 8,      # ?
        signed = True,
    )


class CIMANodeDevice(BaseDevice):

    kind = 'cima-node'

    devices = {
        'pe-cluster': {
            'kind': 'cima-pe-cluster',
            'number': 4
        },
        # 'adder':{
        #     'kind': 'cima-adder'
        # },
        # 'pooler':{
        #     'kind': 'cima-pooler'
        # },
        'dmac':{
            'kind': 'cima-dmac'
        }
    }



