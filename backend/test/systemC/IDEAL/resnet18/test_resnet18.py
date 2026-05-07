from systemC.IDEAL.gen_code import CodeGen
from core.ir import load_ir
from device.c200 import C200CalcInfo

ir = load_ir(file='resnet18_0_mapped_ir.yaml')
calc_info = C200CalcInfo()
for name, layer in ir.iter_layers():
    if layer.type == 'op':
        if layer.op.op_id in ['conv2d', 'linear', 'matmul']:
            layer.calc_info = calc_info
ir.dump_json(file="resnet18_calc_info_ir.yaml")
code = CodeGen(ir)
code.run()
