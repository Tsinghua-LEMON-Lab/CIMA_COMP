from torch_script.gen_code_1 import CodeGen
from core.ir import load_ir
from device.c200 import C200CalcInfo
import math
import numpy

path = "test\\torch_script\\matmul"
ir = load_ir(file=path + "\\" + 'ir_mapped_matmul.yaml')
calc_info = C200CalcInfo()
for name, layer in ir.iter_layers():
    if layer.type == 'op':
        if layer.op.op_id in ['conv2d', 'linear', 'matmul','fc']:
            layer.calc_info = calc_info
ir.dump_json(file=path + "\\" + "fc_calc_info_ir.yaml")
code = CodeGen(ir)
code.run(output_file=path + '\\' + "test.py")
