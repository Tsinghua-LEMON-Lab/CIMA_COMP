from torch_script.gen_code import TorchCodeGen
from core.ir import load_ir
from device.c200 import C200CalcInfo
from onnx2ir.converter import ConvertONNX
from mapper import mapper
from placement import OneOnOne

path = "torch_script\\"
model = 'YOLO.onnx'

device = [{
    'name':'c200-0',
    'kind':'rram-144k-cluster',
    'num':25,
    'ip':'192.168.2.98'
}]
calc_info = C200CalcInfo(shift_expansion_mode='bit_pulse', output_half_level=31)

onnx_obj = ConvertONNX(model, fix_layer_name=True)
onnx_ir = onnx_obj.ir
onnx_weight_data = onnx_obj.model_parser.weight_numpy

map = mapper(ir=onnx_ir,device=device,
            calc_info=calc_info,
            place_strategy=OneOnOne,
            runtime='simulation')
map.run()
mapped_ir = map.ir
ir = mapped_ir
ir.dump_json(file=path + "\\" + "yolo_ir.yaml")

code = TorchCodeGen(ir)
code.to_code(code.gen_layers(), file=path+"script\\1.py")
