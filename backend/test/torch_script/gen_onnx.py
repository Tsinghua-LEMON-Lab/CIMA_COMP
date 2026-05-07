import torch
from test.torch_script.resnet18.nn import MappedNet

x = torch.randn(1,3,32,32)
net = MappedNet()
torch.onnx.export(net, x, f"test_resnet18.onnx",
                  input_names = ['input'],
                  output_names = ['output'],
                  opset_version=11)
