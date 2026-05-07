from ..onnx2ir.parser import *

class PytorchModuleParser(OnnxParser):

    def __init__(self, onnx_model, weight_half_level=None,
                 weight_scale=None, data_clamp_std = 0, data_range_specify = None):
        super().__init__(onnx_model, weight_half_level=weight_half_level,
                 weight_scale=weight_scale, data_clamp_std = data_clamp_std, data_range_specify = data_range_specify)


