from systemC.CIMA.v2.gen_code import CodeGen
from core.ir import load_ir
from device.CIMA import *
from fused_op.op import *
ir = load_ir(file='resnet18_mapped_ir_relu_fused_a_search.yaml')
code = CodeGen(ir)
code.run(output_file='resnet18_a_search.json')
