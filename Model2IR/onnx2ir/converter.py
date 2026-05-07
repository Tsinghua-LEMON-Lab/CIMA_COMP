from pathlib import Path
from .parser import OnnxParser
from .passop import *
from .shape_operation import *
import copy
from irtool.tools import flatten_layers, layer_graph # noqa

class ConvertONNX(object):

    def __init__(self, onnx_file = None, ir_file=None, weight_half_level=None,
                weight_scale = None, fix_layer_name=False, data_range_specify=None,
                data_clamp_std = 0, store_intermediate_model = False,
                specify_input_layer = None, specify_output_layer = None,
                BatchSize=None):
        '''
        onnx_file: ONNX model file (path) or ModelProto
        ir_file: output IR filename (e.g. *.yaml)
        '''
        self.onnx_file = onnx_file
        self.ir_file = ir_file
        self.weight_half_level = weight_half_level
        # self.cpu_layer = cpu_layer
        self.weight_scale = weight_scale
        self.fix_layer_name = fix_layer_name
        self.store_intermediate_model = store_intermediate_model
        self.data_range_specify = data_range_specify
        self.data_clamp_std = data_clamp_std
        self.specify_input_layer = specify_input_layer
        self.specify_output_layer = specify_output_layer
        self.BatchSize = BatchSize
        self._convert()

    def _convert(self):

        if self.onnx_file == None:
            raise ValueError("Missing input. Please provide an ONNX model file (path) or ModelProto.")

        # Parse model.
        if not isinstance(self.onnx_file, onnx.onnx_ml_pb2.ModelProto):
            model = onnx.load(self.onnx_file)
        else:
            model = self.onnx_file

        model, self.updated_name_dict = load_onnx_model(model,fix_layer_name=self.fix_layer_name,store_intermediate_model=self.store_intermediate_model)
        self.model_parser = OnnxParser(model,self.weight_half_level,
                                       self.weight_scale,self.data_clamp_std,
                                       self.data_range_specify)

        # Build IR.
        self.ir = make_ir()

        # Redundant layer list: if input/output layers are specified, nodes outside the slice are redundant.
        self.all_reduntant_layers = []
        if self.specify_input_layer != None:
            self.model_parser.inputs.clear()
            for sil in self.specify_input_layer:
                self.model_parser.inputs.append(self.model_parser.nodes[sil].input[0])
            input_node_name = [i for i in self.model_parser.inputs]
            # Mark all layers before the specified inputs.
            self.get_pre_layers_specific(input_node_name)

        # Graph input layer.
        g_inputs = []

        for input_name in self.model_parser.inputs:
            input_value_info = self.model_parser.value_infos[input_name]
            input_shape = dim_to_list(input_value_info.type.tensor_type.shape.dim)
            if len(input_shape) == 4:
                in_channel, in_height, in_width = input_shape[1:]
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            elif len(input_shape) == 2:
                in_channel = input_shape[1]
                in_height, in_width = 1, 1
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            elif len(input_shape) == 3:
                in_channel = input_shape[0]
                in_height = input_shape[1]
                in_width = input_shape[2]
                # in_height, in_width = 1, 1
                temp_d = dict(channel=in_channel,height=in_height,width=in_width,channel_last=True)
            else:
                raise ValueError(f"Unsupported input rank/shape: {input_shape}.")
            g_inputs.append(temp_d)
        self.ir.add_layer('graph_input',type='input',inputs=g_inputs)


        if self.specify_output_layer != None:
            out_node_names = []
            for ol in self.specify_output_layer:
                for o in self.model_parser.nodes[ol].output:
                    out_node_names.append(o)
            # Mark all layers after the specified outputs.
            self.get_next_layers_specific(out_node_names)
        # Detect whether the graph contains loop ops.
        HasLoopOp = False

        # Convert internal nodes.
        MakeIR = MakeIROp()
        for node_name in self.model_parser.nodes.keys():
            if node_name in self.all_reduntant_layers:
                continue
            node = self.model_parser.nodes[node_name]
            op_type = node.op_type
            if op_type in ['LSTM']:
                HasLoopOp = True
            func = getattr(MakeIR,op_type,None)
            if func == None:
                raise ValueError(f"Operator is not implemented in MakeIROp: {op_type!r}")
            if op_type == 'Reshape':
                func(self.ir, self.model_parser, node_name, self.BatchSize)
            else:
                func(self.ir, self.model_parser, node_name)


        output_layer_name = []
        if self.specify_output_layer != None:
            for sl in self.specify_output_layer:
                output_layer_name.append(self.model_parser.nodes[sl].output[0])
        else:
            output_layer_name = self.model_parser.graph_output

        # Graph output layer.
        g_outputs = []
        for out_name in output_layer_name:
            #     out_node_name = self.model_parser.nodes[out_name].output[0]
            # else:
            #     out_node_name = out_name
            out_value_info = self.model_parser.value_infos[out_name]
            out_shape = dim_to_list(out_value_info.type.tensor_type.shape.dim)
            ref_name = self.model_parser.predecessors[out_name][0].name
            if len(out_shape) == 4:
                out_channel,out_height,out_width = out_shape[1:]
                temp_d = dict(ref=ref_name, channel=out_channel, height=out_height, width=out_width,channel_last=True)
            elif len(out_shape) == 2:
                out_channel = out_shape[1]
                temp_d = dict(ref=ref_name, channel=out_channel, height=1, width=1, channel_last=True)
            elif len(out_shape) == 3:
                out_channel,out_height,out_width = out_shape[0],out_shape[1],out_shape[2]
                temp_d = dict(ref=ref_name, channel=out_channel, height=out_height, width=out_width, channel_last=True)
            else:
                raise ValueError(f"Unsupported output rank/shape: {out_shape}.")
            g_outputs.append(temp_d)
        self.ir.add_layer('graph_output',type='output',inputs=g_outputs)

        # Loop FuseOp
        if HasLoopOp:
            self.LoopFuseOp()

        # remove reduntant layer
        while True:
            meanless_layers = []
            next_layers_dict = self.get_ir_next_layer(self.ir.layers)
            for key, layer in self.ir.layers.items():
                if layer.type == 'op' and key not in next_layers_dict.keys():
                    meanless_layers.append(key)
            if meanless_layers != []:
                for k in meanless_layers:
                    self.ir.layers.pop(k)
            else:
                break
        # flatten IR
        self.ir.layers = self.ir.flatten_layers()

    def LoopFuseOp(self):

        layers_info = copy.deepcopy(self.ir.layers)
        next_layer_dict = self.get_ir_next_layer(layers_info)

        for name, layer in layers_info.items():
            # Check whether this loop can be fused with a fixed op pattern.
            can_fuse_loop = False
            fused_layer_name = []
            if layer.type == 'loop':
                next_layers = next_layer_dict[name]

                # Only support the case where the next layer is a single op (1st op).
                if len(next_layers) == 1:
                    nl = next_layers[0]
                    if layers_info[nl].type == 'op' and layers_info[nl].op.op_id == 'squeeze':
                        fused_layer_name.append(nl)
                        # Next-next layer (2nd op).
                        next_layers_2nd = next_layer_dict[nl]
                        if len(next_layers_2nd) == 1:
                            nl_2nd = next_layers_2nd[0]
                            if layers_info[nl_2nd].type == 'op' and layers_info[nl_2nd].op.op_id == 'gather':
                                fused_layer_name.append(nl_2nd)
                                can_fuse_loop = True

                # Apply fusion.
                if can_fuse_loop:
                    # Update ref name for the 3rd layer.
                    nl_3nd = next_layer_dict[nl_2nd]
                    for n3 in nl_3nd:
                        nl_3nd_layers_info = self.ir.layers[n3]
                        nl_3nd_layers_info.inputs[0].ref = f'{name}:0'
                    # Remove fused layers.
                    for fln in fused_layer_name:
                        self.ir.layers.pop(fln)

        # Sort layers.
        self.ir.layers = dict(self.ir.iter_layers(deep=False, sorted=True))

    def get_next_layers_specific(self, out_node_names):

        for on in out_node_names:

            if on not in self.model_parser.successors.keys():
                continue

            # Special-case: if a successor is one of ['Conv_342','Conv_379','Conv_416'],
            # only remove nodes after that one.
            OnlyAddOne = False
            t = 0
            for node in self.model_parser.successors[on]:
                if node.name in ['Conv_342', 'Conv_379', 'Conv_416']:
                    OnlyAddOne = True
                    break
                t += 1

            if OnlyAddOne:
                node = self.model_parser.successors[on][0]
                if node.name not in self.all_reduntant_layers:
                    self.all_reduntant_layers.append(node.name)
                if node.name not in self.model_parser.graph_output:
                    onn_ = []
                    for i in node.output:
                        onn_.append(i)
                    if onn_ != []:
                        self.get_next_layers_specific(onn_)
            else:
                for node in self.model_parser.successors[on]:
                    if node.name not in self.all_reduntant_layers:
                        self.all_reduntant_layers.append(node.name)
                    else:
                        continue
                    # IsOutput = True
                    # input()
                    if node.name not in self.model_parser.graph_output:
                        onn_ = []
                        for i in node.output:
                            onn_.append(i)
                        if onn_ != []:
                            self.get_next_layers_specific(onn_)


    def get_pre_layers_specific(self, input_node_names):
        while input_node_names != []:
            node_name = input_node_names.pop(0)
            # Get predecessor nodes.
            for pre_node in self.model_parser.predecessors[node_name]:
                if pre_node.name not in self.all_reduntant_layers:
                    self.all_reduntant_layers.append(pre_node.name)
                # Walk further upstream unless this is a graph input / constant / parameter.
                IsInput = True
                inn_ = []
                for i in pre_node.input:
                    if i not in self.model_parser.constant.keys() and i not in self.model_parser.parameters.keys() and i not in self.model_parser.graph_input:
                        inn_.append(i)
                        IsInput = False
                if not IsInput and inn_ != []:
                    for i in inn_:
                        if i not in input_node_names:
                            input_node_names.append(i)
                else:
                    break
            # input()
        #             self.all_reduntant_layers.append(node.name)
        #         IsInput = True
        #         inn_ = []
        #                 inn_.append(i)
        #                 IsInput = False
        #             self.get_pre_layers_specific(inn_)

    def dump(self):
        if self.ir_file == None:
            # using the ONNX filename stem.
            file_path = os.getcwd()
            ir_file = Path(file_path +'\\'+ self.onnx_file.split('\\')[-1].split('.')[0] + '.yaml')
        else:
            ir_file =  self.ir_file
        self.ir.dump(file=ir_file)

    def gen_calc_info(self, calc_info_obj = None):
        '''
        input : calc_info_obj
        '''
        calc_info = {}
        weight_quant_scale = self.model_parser.weight_quant_scale
        if weight_quant_scale != {}:
            for key in weight_quant_scale.keys():
                layer_name = key.split('.')[0]
                if isinstance(calc_info_obj, dict):
                    calc_ = calc_info_obj[layer_name]
                    calc_.weight_scale = float(weight_quant_scale[key])
                    calc_info[layer_name] = calc_
                else:
                    raise ValueError(f"calc_info_obj type error: {type(calc_info_obj)}. Expected dict.")
        return calc_info

    def get_ops(self):

        # Count effective ops: multiply and add are counted separately.
        MAC_count = 0
        Relu_count = 0
        # Track total feature map size.
        Output_size = 0

        MVM_parameters_count = 0
        for node_name, node in self.model_parser.nodes.items():
            if node_name in self.all_reduntant_layers:
                continue
            if node.op_type in ['Conv', 'MatMul', 'Gemm', 'ConvTranspose']:

                # Compute times: product of output spatial size.
                output_shape = dim_to_list(self.model_parser.value_infos[node.output[0]].type.tensor_type.shape.dim)
                if len(output_shape) == 4:
                    calc_times = output_shape[2] * output_shape[3]
                    # Output_size = max(Output_size, output_shape[2] * output_shape[3] * output_shape[1])
                    Output_size += output_shape[2] * output_shape[3] * output_shape[1]

                elif len(output_shape) == 2:
                    calc_times = 1
                    # Output_size = max(Output_size, output_shape[1])
                    Output_size += output_shape[1]
                else:
                    raise ValueError(f"Unsupported output rank/shape: {output_shape}")

                # Per-inference compute: prod(weight_shape) * 2 (mul+add).
                weight_shape =  dim_to_list(self.model_parser.value_infos[node.input[1]].type.tensor_type.shape.dim)
                calc_numbers = np.prod(np.array(weight_shape))
                MVM_parameters_count += calc_numbers / 10 **(3)

                # Layer total ops: per-inference ops * compute times.
                MAC_count += (calc_numbers / 10 **(3)) * (calc_times / 10 **(3)) * 2


            if node.op_type == 'Relu':
                input_shape = dim_to_list(self.model_parser.value_infos[node.input[0]].type.tensor_type.shape.dim)
                # Ops: input feature map size.
                Relu_count += np.prod(np.array(input_shape))

        print(f'Total Conv/FC MAC operations: {round(MAC_count / 10**(3), 4)} GOPs')
        print(f'Total Conv/FC Weight parameters: {round(MVM_parameters_count / 10**(3), 4)} M')
        print(f'Max Feature Map Size: {round(Output_size / 10**(6), 4)} M')

        return MAC_count / 10**(3), MVM_parameters_count / 10**(3), Output_size / 10**(6)


    def get_ir_pre_layer(self, layers):
        ''''
        Build mapping: current layer name -> list of predecessor layer names.
        input: {layer_name:layer_object}
        return: {current_layer_name: pre_layer_name}
        '''
        prefix_layer = {}
        for name,layer in layers.items():
            if layer.type not in ['input'] and not (layer.type == 'op' and layer.op.op_id in ['constant']):
                prefix_layer[name] =  []
                for i in layer.inputs:
                    if 'graph_input' not in i.ref:
                        ref = i.ref
                        if ':' in ref:
                            ref = ref.split(':')[0]
                        prefix_layer[name].append(ref)
                    else:
                        prefix_layer[name].append(i.ref)
        return prefix_layer

    def get_ir_next_layer(self, layers):
        '''
        Build mapping: current layer name -> list of successor layer names.
        input: {layer_name:layer_object}
        return: {current_layer_name: next_layer_name}
        '''
        next_layer = {}
        pre_layer = self.get_ir_pre_layer(layers)

        for k,v in pre_layer.items():
            #             next_layer[name] = []
            #         next_layer[name].append(k)
            #     continue

            for name in v:
                if name not in next_layer.keys():
                    next_layer[name] = []
                next_layer[name].append(k)

        return next_layer
