from core import load_ir
import json
from CIMA.sim_in_loop_dmem_optim import CIMA_DMEM_Optimizer

if __name__ == "__main__":

    path = f'test\\CIMA\\resnet18\\'
    # ir
    ir_name = f'resnet18.yaml'
    sc_config = f'resnet18_systemc_dmem_mini_gen_v2_single_frame_alpha_1_0002_no_ceil.json'
    ir = load_ir(file = path+ f'{ir_name}')
    #
    with open(path+f'{sc_config}', 'r') as f:
        sc_config = json.load(f)

    ##
    dmem_optimizer = CIMA_DMEM_Optimizer(ir, sc_config, path + f'\\performance_optim\\')
    dmem_optimizer.run(module_name='resnet18_distance_cal_num')
