from systemC.IDEAL.gen_code import CodeGen
from core.ir import load_ir
from device.base import CalcInfo
import math
import numpy

path = "test\\systemC\\IDEAL\\matmul"
ir = load_ir(file=path + "\\" + 'ir_mapped_matmul.yaml')
calc_info = CalcInfo()
for name, layer in ir.iter_layers():
    if layer.type == 'op':
        if layer.op.op_id in ['conv2d', 'linear', 'matmul','fc']:
            layer.calc_info = calc_info
ir.dump_json(file=path + "\\" + "fc_calc_info_ir.yaml")
code = CodeGen(ir,output=path + '\\' + 'systemc.h')
code.run()
weight_array= {}
for i in range(16):
    temp = math.floor(i / 4)
    if temp % 2 == 0:
        weight_array[i] = numpy.random.randint(-8,8,size=(14,16))
    else:
        weight_array[i] = numpy.random.randint(-8,8,size=(13,16))
code.gen_weight(weight_array,weight_file=path +'\\' + 'weight.txt')
