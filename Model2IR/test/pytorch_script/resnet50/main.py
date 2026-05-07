from torch_script.gen_code import TorchCodeGen
from core import load_ir
from helper import *
from mapper import *
import json
from pytorch2ir.helper import convert_module_to_object
from pytorch2ir.converter import ConvertPytorchModule

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

def make_json(program_dict, output_file):
    json_str = json.dumps(program_dict, cls = MyEncoder, indent=4)
    with open(output_file, 'w') as json_file:
        json_file.write(json_str)

if __name__ == "__main__":

    # device
    devices = [{
        'name':'c200-0',
        'kind':'rram-144k-cluster',
        'num':1000,
        'profile': {
            'in_channel': 576,   # 1152/2
            'out_channel': 128 * 4,  # 128
            'in_bits': 2,        # [-1, 1] sint2
            'out_bits': 4,       # [-7, 7] sint4
            'weight_bits': 4,    # [-7, 7] sint4
            'signed': True,
        }
    }]


    file_name = 'resnet50'

    pytorch_file = f'resnet50_ideal_torch_script_full_layers_AddFirst.py'
    pt_module_name = f'resnet50IdealNet'
    model = convert_module_to_object(pytorch_file, pt_module_name)
    input_shape = [1,3,224,224]
    pt_obj = ConvertPytorchModule(model, input_shape=input_shape, fix_layer_name=True)
    #
    path = f''
    pt_ir = pt_obj.ir
    pt_ir.dump_json(file=path + f'{file_name}_onnx_ir.yaml')
    updated_name_dict = pt_obj.updated_name_dict
    make_json(updated_name_dict, path + f'updated_model_name.json')

    # wo_bn_layers = None
    # code generation
    code = TorchCodeGen(pt_ir, module_name=f'{file_name}IdealNet')
    code.to_code(generator=code.gen_layers(), file=path + f'{file_name}_ideal_torch_script_full_layers_AddFirst.py')

    # mapping
    map = mapper(ir=pt_ir, device = devices,
                weight_format = 'HWC',
                place_strategy = OneOnOne,
                relu_fuse= False,
                pool_fuse= False,
                split_fuse = False,
                silu_fuse= False,
                adaptive_split_ir = True,
                operator_replace = True,
                )
    map.run()

    # gen code
    mapped_ir = map.ir
    mapped_ir.dump_json(file= path + f'{file_name}_Hardware_adaptive_ir_torch_AddFirst.yaml')
    # code generation
    code = TorchCodeGen(mapped_ir, module_name=f'{file_name}AdaptiveNet')
    code.to_code(generator=code.gen_layers(), file=path + f'{file_name}_Hardware_adaptive_torch_script_full_layers_AddFirst_shared_activation.py')

