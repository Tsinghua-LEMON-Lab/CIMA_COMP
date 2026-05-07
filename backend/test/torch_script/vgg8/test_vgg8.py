from torch_script.gen_code_1 import CodeGen
from core.ir import load_ir
from onnx2ir.converter import ConvertONNX
from mapper import mapper
from pytorch2ir.converter import ConvertPytorchModule
from pytorch2ir.helper import convert_module_to_object

device = [{
        'name':'c200-0',
        'kind':'rram-144k-cluster',
        'num':128,
        'ip':'192.168.2.98'
    }]

path = "test\\torch_script\\vgg8\\"
# onnx_obj = ConvertONNX(path+'vgg8.onnx')
# onnx_ir = onnx_obj.ir
# map = mapper(ir=onnx_ir,device=device)
# map.run()
# mapped_ir = map.ir
# code = CodeGen(mapped_ir)
output_file = path + '\\' + "nn_1.py"
# code.run(output_file=output_file)

pt_module = convert_module_to_object(output_file, 'MappedNet')
pt_obj = ConvertPytorchModule(pt_module, input_shape=[1,3,32,32])
pt_ir = pt_obj.ir
map = mapper(ir=pt_ir,device=device)
map.run()
mapped_ir = map.ir
mapped_ir.dump_json(file=path + 'vgg8_split_mapped_layer.yaml')
