from systemC.CIMA.v1.gen_code import CodeGen
from core.ir import load_ir
from device.CIMA import *
from fused_op.op import *
ir = load_ir(file='fashionmnist_mapped_ir_relu_fused_2.yaml')
code = CodeGen(ir)
code.run()
