from pathlib import Path
from importlib import import_module
from .parser import PytorchModuleParser
from ..onnx2ir.helper import *
import torch
from ..onnx2ir.passop import *
from ..onnx2ir.shape_operation import *
from ..onnx2ir.converter import ConvertONNX

class ConvertPytorchModule(ConvertONNX):
    '''
    Convert a PyTorch module to ONNX first, then convert ONNX to IR.
    '''
    def __init__(self, pytorch_module = None, input_shape=None, weight_file=None, ir_file=None,
                 weight_half_level=None, weight_scale = None, fix_layer_name=False, data_range_specify=None,
                 data_clamp_std = 0, store_intermediate_model = False,
                 specify_input_layer = None, specify_output_layer = None,
                 BatchSize=None):

        self.pytorch_module = pytorch_module
        self.input_shape = input_shape
        self.weight_file = weight_file
        self.ir_file = ir_file
        #
        onnx_model = self._convert_onnx()

        super().__init__(onnx_file = onnx_model, ir_file=ir_file, weight_half_level=weight_half_level,
                        weight_scale = weight_scale, fix_layer_name=fix_layer_name, data_range_specify=data_range_specify,
                        data_clamp_std = data_clamp_std, store_intermediate_model = store_intermediate_model,
                        specify_input_layer = specify_input_layer, specify_output_layer = specify_output_layer,
                        BatchSize=BatchSize)
        self._convert()

    def _convert_onnx(self):

        if self.pytorch_module == None:
            raise ValueError("Missing input: please provide a PyTorch module.")

        # Export to ONNX.
        # module = import_module(self.pytorch_module)
        pt_model = self.pytorch_module
        if self.input_shape == None:
            raise ValueError("Missing input_shape: please provide the model input shape.")
        x = torch.randn(self.input_shape, requires_grad=True)
        file_path = os.getcwd() + '\\temp\\'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.onnx.export(pt_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  file_path + f"temp.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])

        model = onnx.load(file_path + "temp.onnx")
        return model

