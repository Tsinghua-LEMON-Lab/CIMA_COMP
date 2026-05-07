from onnx2ir.converter import ConvertONNX
from mapper import mapper
from cis.gen_code import CodeGen
from systemC.IDEAL.build import *
from placement.LLA import LowestLevelAlgorithm
from .cis import CIS # noqa

test_path = 'test\\cis\\FA'

cpu_layer = None

model = test_path + '\\' + 'FA_FP32.onnx'

onnx_obj = ConvertONNX(model)
onnx_ir = onnx_obj.ir

device = {
    'name':'CIS_v1',
    'kind':'rram-cis',
    'num':1
}

# mapping
map = mapper(ir=onnx_ir,device=device,place_strategy=LowestLevelAlgorithm)
mapped_ir = map.ir

# mapped_ir.dump_json(file=test_path + '\\' +'FA_mapped_ir.yaml')

codegen = CodeGen(mapped_ir)
codegen.run(output_file=test_path + '\\' + 'FA_code.txt')
