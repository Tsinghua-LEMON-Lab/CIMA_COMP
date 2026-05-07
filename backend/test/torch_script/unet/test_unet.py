from torch_script.gen_code_1 import CodeGen
from core.ir import load_ir
from device.c200 import C200CalcInfo
from onnx2ir.converter import ConvertONNX
from mapper import mapper
from placement import OneOnOne

path = "test\\torch_script\\unet\\"
model = path + 'unet.onnx'

device = [{
    'name':'c200-0',
    'kind':'rram-144k-cluster',
    'num':22,
    'ip':'192.168.2.98'
}]
calc_info = C200CalcInfo(shift_expansion_mode='bit_shift', output_half_level=31)

onnx_obj = ConvertONNX(model,weight_half_level=6)
onnx_ir = onnx_obj.ir
onnx_weight_data = onnx_obj.model_parser.weight_numpy

map = mapper(ir=onnx_ir,device=device,
            calc_info=calc_info,
            place_strategy=OneOnOne,
            runtime='simulation')
mapped_ir = map.ir
ir = mapped_ir

ir.dump_json(file=path + "\\" + "unet_calc_info_ir.yaml")
code = CodeGen(ir)
code.run(output_file=path+"\\LBL_code\\"+"nn.py", layer_by_layer_train=True)
