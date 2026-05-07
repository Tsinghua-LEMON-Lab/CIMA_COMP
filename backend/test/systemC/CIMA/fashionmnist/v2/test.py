from systemC.CIMA.v2.gen_code import CodeGen
from core.ir import load_ir
from device.CIMA import *
from self_defined_op.fused_op import *

ir = load_ir(file='fashionmnist_mapped_ir_relu_fused_2.yaml')
code = CodeGen(ir)
code.run()
