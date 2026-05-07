from systemC.IDEAL.gen_code import CodeGen
from device.c200 import C200CalcInfo
from mapper import mapper
from onnx2ir.converter import ConvertONNX

model_path = 'test\\systemC\\IDEAL\\resnet18\\full_arch\\'
model_name = 'resnet18.onnx'

device = [{
    'name':'c200-0',
    'kind':'rram-144k-cluster',
    'num':1000,
}]

onnx_obj = ConvertONNX(model_path + model_name)
onnx_ir = onnx_obj.ir

calc_info = C200CalcInfo()

map = mapper(ir=onnx_ir,device=device,calc_info=calc_info)
map.run()
mapped_ir = map.ir
mapped_ir.dump_json(file=model_path + 'resnet18.yaml')

code = CodeGen(mapped_ir)
code.run()
