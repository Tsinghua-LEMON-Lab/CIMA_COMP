from CIMA.uvm.gen_code import UVMCodeGen
from systemC.CIMA.v2.gen_code import CodeGen
from core import load_ir
from device.CIMA import * # noqa
from self_defined_op import * # noqa

def fashionmnist():
    file = 'fashionmnist\\fashionmnist_workload_balance.json'
    code_uvm = UVMCodeGen(file, module_name=f'fashionmnist')
    code_uvm.to_code(generator=code_uvm.gen_layers(), file=f'fashionmnist\\fashionmnist_uvm_codes.txt')
    code_uvm.gen_register_raw_data(raw_config_file=f'fashionmnist\\register_raw_data.json')

def yolov5():
    file = 'yolov5\\yolov5_all_layers_mapped_params_head_split_wo_Conv_0.yaml'
    ir = load_ir(file = file)
    code_systemc = CodeGen(ir)
    code_systemc.run(output_file=f'yolov5\\yolov5_all_layers_mapped_params_head_split_wo_Conv_0.json', run_time = 10 * 10**(6))
    # uvm codes
    uvm_file = f'yolov5\\yolov5_all_layers_mapped_params_head_split_wo_Conv_0.json'
    code_uvm = UVMCodeGen(uvm_file, module_name=f'Yolov5')
    code_uvm.to_code(generator=code_uvm.gen_layers(), file=f'yolov5\\yolov5_uvm_codes_wo_Conv_0.txt')
    code_uvm.gen_register_raw_data(raw_config_file=f'yolov5\\register_raw_data.json')

if __name__ == "__main__":
    fashionmnist()
    # yolov5()
