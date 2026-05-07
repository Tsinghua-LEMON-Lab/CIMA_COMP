from systemC.IDEAL.gen_code import CodeGen
from device.c200 import C200CalcInfo
from onnx2ir.converter import ConvertONNX
from mapper import mapper
from device.c200 import C200CalcInfo
from device.c200 import MappedLayer #noqa
from placement import LowestLevelAlgorithm
import numpy as np

def load_csv(fn, dtype='int32'):
    assert fn
    return np.loadtxt(fn, dtype=dtype, delimiter=',', ndmin=2)

def save_csv(fn, data):
    assert fn
    np.savetxt(fn, data, delimiter=',', fmt='%d')

def txt2numpy(fn):

    with open(fn,'r') as f:
        data = f.readline()
        data_all = []
        while (data):
            data = data.strip()
            data1 = data.split(',')
            if data1[-1] == '':
                data1.remove(data1[-1])
            new_data = []
            for i in data1:
                new_data.append(int(i))
            data = f.readline()
            data_all.append(new_data)

    return np.array(data_all)

if __name__ == "__main__":

    cpu_layer = None

    device = [{
        'name':'c200-0',
        'kind':'rram-144k-cluster',
        'num':500,
        'ip':'192.168.2.98'
    }]

    model = 'model.onnx'

    onnx_obj = ConvertONNX(model,weight_half_level=6,
                        cpu_layer=cpu_layer)
    onnx_ir = onnx_obj.ir
    onnx_weight_data = onnx_obj.model_parser.weight_numpy
    onnx_weight_data_quant = onnx_obj.model_parser.weight_numpy_quant


    calc_info = C200CalcInfo(shift_expansion_mode='bit_shift',
                             output_half_level=31, adc_clamp= True,
                             adc_quant = True, noise_scale=0.04)

    map = mapper(ir=onnx_ir,device=device,
                cpu_layer=cpu_layer,
                calc_info=calc_info,
                runtime='simulation',)
    map.run()
    mapped_ir = map.ir

    mapped_ir.dump_json(file="resnet50_calc_info_ir.yaml")
    code = CodeGen(mapped_ir)
    code.run()
