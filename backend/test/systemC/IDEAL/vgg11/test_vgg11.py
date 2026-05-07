from systemC.IDEAL.gen_code import CodeGen
from core.ir import load_ir
from device.c200 import C200CalcInfo

path = "test\\systemC\\IDEAL\\vgg11"
ir = load_ir(file=path + "\\" + 'VGG11_mapped_ir.yaml')
calc_info = C200CalcInfo()
for name, layer in ir.iter_layers():
    if layer.type == 'op':
        if layer.op.op_id in ['conv2d', 'linear', 'matmul']:
            layer.c200_calc_info = calc_info
ir.dump_json(file=path + "\\" + "VGG11_calc_info_ir.yaml")
code = CodeGen(ir)
code.run(output_file=path)
