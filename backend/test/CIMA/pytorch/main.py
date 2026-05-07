
# CIM toolchain
from core import load_ir
from self_defined_op.cima_op import *  # noqa
from self_defined_op.fused_op import *  # noqa

from CIMA.pytorch.gen_code import TorchRTCodeGen

if __name__ == "__main__":

    # load ir
    ir = load_ir(file = 'ir\\yolov5_all_layers_mapped_params.yaml')
    # code generation
    code = TorchRTCodeGen(ir, module_name=f'Yolov5_pytorch_rt')
    code.to_code(generator=code.gen_layers(), file=f'yolov5_pytorch_rt.py')
