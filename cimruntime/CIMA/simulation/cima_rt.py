from ...numpy import NumpyRuntime
from ...cimtensor import CIMNumpyTensor as CNT
from .utils import *
from mapper.device.CIMA import MappedLayer # noqa
from typing import Callable
from irtool.core import BaseIR
from irtool.tools import flatten_layers  # noqa
from irtool.runtime.utils import *
import copy
import warnings

class CIMANumpyRT(NumpyRuntime):

    name = "CIMA"

    def __init__(self, *, activation_lut = None, weight_noise = 0., output_noise=0.):
        '''
        ================== Integer-only dataflow (CIMA simulation pipeline) ==================
        activation_lut: numpy array. Activation lookup table for complex activations (silu/sigmoid/tanh).
        '''
        super().__init__()
        self.activation_lut = activation_lut
        self.weight_noise = weight_noise
        self.output_noise = output_noise

    def run_ir(self, ir_, inputs, weights=None, log_info=False, *, outputs=None, callback=None, **kwargs):
        # Sort layers.
        ir = copy.deepcopy(ir_)
        ir.layers = dict(ir.iter_layers(deep=False, sorted=True))

        assert isinstance(ir, BaseIR), f'invalid IR type={type(ir)}'
        layers = ir.flatten_layers()
        inp, oup = ir.get_io_layers(layers)
        inl, oul = layers[inp], layers[oup]

        if isinstance(inputs, dict):
            data = {k: tuple(v) if isinstance(v, (tuple, list)) else (v,)
                    for k, v in inputs.items() if v is not None}
        elif isinstance(inputs, (tuple, list)):
            assert len(inputs) == len(inl.inputs)
            data = {inp: tuple(inputs)}
        else:
            data = {inp: (inputs,)}

        for k, v in data.items():
            assert k in layers, f'invalid input name {k!r}'
            assert isinstance(v, (tuple, list)), \
                f'invalid inputs type {type(v)}'
        ons = None
        if outputs is not None:
            if isinstance(outputs, str):
                ons = set(outputs)
            elif outputs is True:
                ons = set(layers.keys()) - {inp, oup}
            elif isinstance(outputs, (tuple, list, set)):
                ons = set(outputs)
            for k in ons:
                assert k in layers, f'invalid output name {k!r}'


        if callback is not None:
            assert isinstance(callback, Callable), \
                f'invalid callback {type(callback)}'

        # input()
        self.layer_calc_time = {}
        self.log_info = log_info

        for name, layer in layers.items():

            if self.log_info:
                print(f"Running layer: {name} ===>")

            if name in data:
                continue    # layer is done

            if layer.type == 'op' and layer.op.op_id in ['constant']:
                data[name] = [np.array(layer.op.value)]
                continue

            if any(dd.parse_ref()[0] not in data for dd in layer.inputs):
                continue    # layer can't be run

            if name == oup:
                data[name] = []
                for o in layer.inputs:
                    name_ = o.parse_ref()[0]
                    assert name_ in data.keys()
                    data[name].append(data[name_])

                break       # output layer

            IsReuseLayer = False
            # resue layer
            if layer.type == 'reuse':
                reuse_layer_name = name
                name = layer.layer
                # Replace reuse-layer inputs.
                reuse_layer = ir.layers[name]
                reuse_layer.inputs = layer.inputs
                layer = reuse_layer
                IsReuseLayer = True

            x = []
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                x.append(data[nm][0 if idx is None else idx])

            ats = layer.op.get_attrs()
            for ats_n in ['with_batch', 'channel_pos']:
                if ats_n in ats.keys():
                    ats.pop(ats_n)

            wts = dict()
            device_info = dict()
            layer_info = dict()

            if layer.CIMA_mapping_info != None:
                device_info.update(dict(device_info=ir.devices))
                layer_info.update(dict(layer_info=ir.layers[name], layer_name = name))

            if layer.op.op_id in ['conv2d','matmul','fc', 'linear','conv_transpose2d', 'fused_conv2d', 'fused_fc']:

                if layer.op.bias:
                    bias_name = f"{name}.bias"
                    if bias_name not in weights.keys():
                        warnings.warn(f"Missing bias '{bias_name}' in weights; layer '{name}' will run without bias.")
                    wts['bias'] = weights.get(bias_name)

                for k in layer.op.weights:
                    wn = f'{name}.{k}'
                    if k not in layer.op.optional_weights:
                        assert wn in weights, f'missing weight {wn}'
                    wts[k] = weights.get(wn)

                # Load weights.
                wn = f'{name}.weight'
                wts['weight'] = weights.get(wn)

            elif layer.op.op_id in ['batch_norm2d']:

                for k in layer.op.weights:
                    ats[k] = np.array(ats[k])

            # record weight scale
            if layer.CIMA_mapping_info == None and layer.CIMA_calc_info != None:
                layer_info.update(dict(layer_info=ir.layers[name]))

            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=None, **kwargs)
            # record time
            time1 = time.time()

            # Run current layer.
            y = self.run_layer(layer, *x, **wts, **ats, **device_info, **layer_info)

            # input()
            time2 = time.time()
            self.layer_calc_time[name] = round(time2 - time1, 4)

            if not isinstance(y, (tuple, list)):
                y = (y,)
            if callback is not None:
                callback(name, layer=layer, inputs=x, weights=wts,
                         attrs=ats, outputs=y, **kwargs)

            if IsReuseLayer:
                data[reuse_layer_name] = tuple(y)
            else:
                data[name] = tuple(y)

            if ons is not None and all(k in data for k in ons):
                break       # all outputs are ready

        if ons is not None:
            res = {}
            for k in ons:
                v = data[k]
                res[k] = v[0] if len(v) == 1 else v
            if isinstance(outputs, str):
                res = iter(res.values()).next()
        else:
            res = []
            for dd in oul.inputs:
                nm, idx = dd.parse_ref()
                res.append(data[nm][0 if idx is None else idx])
            if len(res) == 1:
                res = res[0]
        return res

    def run_op(self, op_id, *args, **kwargs):
        fn = getattr(self, f'fn_{op_id}', None)
        assert isinstance(fn, Callable), f'fn_{op_id} is not a function'
        return fn(*args, **kwargs)

    def run_layer(self, layer, *args, **kwargs):
        return self.run_op(layer.op.op_id, *args, **kwargs)

    # matmuls
    def fn_matmul(self, x, **kwargs):
        '''
        =================================================
                      CIMA integer-only simulation dataflow (MatMul)
        =================================================
        input:
          x: numpy, integer input with dtype '4bit'/'8bit'
        weight / bias:
          weight/bias are also integer values (pre-quantized externally during training/export)
        output:
          output is quantized to the target hardware dtype ('4bit'/'8bit')
        =================================================
        '''

        # Decide whether to run on CIM.
        if 'device_info' not in kwargs.keys():
            # Software compute path.
            output = super().fn_fc(x,**kwargs)
            return output
        else:
            # Step 1. Gather runtime params.
            layer_info = kwargs['layer_info']
            ADC_qunat_level = layer_info.CIMA_calc_info.ADC_qunat_level
            scale_shift_num = layer_info.CIMA_calc_info.scale_shift_num
            scale = layer_info.CIMA_calc_info.scale
            offset = layer_info.CIMA_calc_info.offset
            accumulate_shift_num = layer_info.CIMA_calc_info.accumulate_shift_num
            data_type = layer_info.CIMA_calc_info.data_type

            # Step 2. Load weights.
            weight_data = kwargs['weight']

            # Step 3. CIMA simulated core compute.
            output = CIMA_analog_MAC(x, weight_data, dtype=data_type, ADC_qunat_level=ADC_qunat_level,
                                     scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                     accumulate_shift_num=accumulate_shift_num)

            return output


    fn_linear = fn_matmul
    fn_fc = fn_matmul
    fused_fc = fn_matmul

    # conv

    def fn_conv2d(self, x, **kwargs):
        '''
        =================================================
                      CIMA simulation dataflow (Conv2d)
        =================================================
        input:
          x: numpy, integer input with dtype '4bit'/'8bit'
        weight / bias:
          weight/bias are also integer values (pre-quantized externally during training/export)
        output:
          output is quantized to the target hardware dtype ('4bit'/'8bit')
        =================================================
        '''

        if 'device_info' not in kwargs.keys():
            # B H W C => B C H W
            x = x.transpose(0, 3, 1, 2)
            output = super().fn_conv2d(x,**kwargs)
            return output

        else:

            # Step 1. Gather runtime params.
            layer_info = kwargs['layer_info']
            ADC_quant_level = layer_info.CIMA_calc_info.ADC_quant_level
            scale_shift_num = layer_info.CIMA_calc_info.scale_shift_num
            scale = np.array(layer_info.CIMA_calc_info.scale)
            offset = np.array(layer_info.CIMA_calc_info.offset)
            accumulate_shift_num = layer_info.CIMA_calc_info.accumulate_shift_num
            data_type = layer_info.CIMA_calc_info.data_type

            # Step 2. Read input shape. Simulation assumes HWC layout and batched inputs.
            assert len(x.shape) == 4, "Expected batched input with rank=4."
            batch, input_rows, input_cols, channel = x.shape
            # Step 3. Gather op attributes.
            padding = layer_info.op.padding
            kernel_size = layer_info.op.kernel
            # stride = layer_info.op.stride
            if 'stride' in kwargs.keys():
                stride = kwargs['stride']
            else:
                stride = layer_info.op.stride
            out_feature_size_rows = int((input_rows + padding + padding - kernel_size) / stride + 1)
            out_feature_size_cols = int((input_cols + padding + padding - kernel_size) / stride + 1)
            # Step 4. Input re-layout (expects HWC, batched).
            array_input = feature_map_to_input_np_HWC(x, stride = stride, kernel_size = kernel_size,
                                        padding = padding, multi_batch = True)

            # weight
            weight_data = kwargs['weight']

            # Choose compute device by name (RRAM vs DMAC).
            for k,v in layer_info.CIMA_mapping_info.mappings.items():
                device_name = v.device
                break
            if self.log_info:
                print(f'Compute location: {device_name}')
            if 'dmac' in device_name:
                output = CIMA_digital_MAC(array_input, weight_data, scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                        accumulate_shift_num=accumulate_shift_num)
            elif 'cima-xb' in device_name:
                # Step 5. CIMA simulated core compute.
                output = CIMA_analog_MAC(array_input, weight_data, dtype=data_type, ADC_quant_level=ADC_quant_level,
                                        scale=scale, offset=offset, scale_shift_num=scale_shift_num,
                                        accumulate_shift_num=accumulate_shift_num,
                                        conductance_noise=self.weight_noise, ADC_noise=self.output_noise)
            else:
                raise ValueError(f"Unsupported device kind: {device_name!r}")

            # Step 6. Restore output shape.
            output = output_to_feature_map(output, out_feature_size_rows, out_feature_size_cols, multi_batch=True)

            # Output defaults to CHW and is converted back to HWC.
            output = output.transpose(0,2,3,1)

            return output

    # fn_fused_conv2d = fn_conv2d
    # convtranspose2d

    def fn_conv_transpose2d(self, x, **kwargs):
        '''
        =================================================
                      CIMA simulation dataflow (ConvTranspose2d)
        =================================================
        input:
          x: numpy, integer input with dtype '4bit'/'8bit'
        weight / bias:
          weight/bias are also integer values (pre-quantized externally during training/export)
        output:
          output is quantized to the target hardware dtype ('4bit'/'8bit')
        =================================================
        '''

        if 'device_info' not in kwargs.keys():
            output = super().fn_conv_transpose2d(x,**kwargs)
            return output
        else:
            in_ = x
            # Gather params.
            group = kwargs['group']
            stride = kwargs['stride']
            padding = kwargs['padding']
            dilation = kwargs['dilation']
            output_padding = kwargs['output_padding']
            auto_pad = kwargs['auto_pad']

            weight_shape = kwargs['layer_info'].weights['weight'].shape
            kci, co, *kernel = weight_shape

            # Transform input.
            ndim = 2

            if self.channel_last:
                ba, *xd, ci = in_.shape
            else:
                ba, ci, *xd = in_.shape
            assert ci == kci * group, \
                f'invalid input shape {in_.shape} with kernel {weight_shape}'
            os, (k, s, p, d, dk, dp, di) = \
                conv_t_shapes(xd, kernel, stride, padding, dilation,
                            output_padding, auto_pad)
            if di != xd:
                xp = np.zeros(self.to_axes(ba, ci, di), dtype=in_.dtype)
                oi = tuple(slice(dp[i], xd[i] * s[i] + dp[i], s[i])
                        for i in range(ndim))
                xp[self.to_slices(oi)] = in_
                in_ = xp

            # Reuse conv2d implementation.
            if stride != 1:
                kwargs['stride'] = 1

            return self.fn_conv2d(in_, **kwargs)


    def rand(self, shape):
        data = self._be.random.randint(low=0, high=self.half_level, size=shape)
        return CNT.to_cimtensor(data=data,multi_batch=self.multi_batch)

    def fn_identity(self, x, **kwargs):
        return x
    # transes

    def fn_concat(self, *x, **kwargs):
        axis = kwargs['layer_info'].op.axis
        # CIMA inference uses BHWC layout (different from PyTorch BCHW).
        if axis == 1:
            axis = 3
        output = CIMA_concat(*x, axis=axis)

        layer_info = kwargs['layer_info']
        data_type = layer_info.CIMA_calc_info.data_type
        op_type = layer_info.op.op_id

        # Apply activation if present.
        if op_type == 'fused_concat':
            layer_name = kwargs['layer_name']
            if layer_info.op.silu != None:

                assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
                lut = self.activation_lut[layer_name]
                if data_type == '4bit':
                    output_query = output + 8
                elif data_type == '8bit':
                    output_query = output + 128
                else:
                    raise ValueError(f"Unsupported data_type: {data_type!r}")
                output_query = output_query.astype(np.int32)
                output = lut[output_query]
            elif layer_info.op.relu != None:
                output = self._be.clip(output, 0)

            if layer_info.op.split != None:
                # attr
                axis = kwargs['layer_info'].op.split.axis
                # CIMA inference uses BHWC layout (different from PyTorch BCHW).
                if axis == 1:
                    axis = 3
                split = kwargs['layer_info'].op.split.split
                output = super().fn_split(output, axis=axis, split=split)
                re_ = []
                for d in output:
                    if len(d.shape) == 3:
                        d = d.squeeze()
                    re_.append(d)
                output = re_

        return output

    fn_fused_concat = fn_concat

    def fn_add(self, *x, **kwargs):

        layer_info = kwargs['layer_info']
        data_type = layer_info.CIMA_calc_info.data_type
        op_type = layer_info.op.op_id

        output = CIMA_add(*x, dtype=data_type)

        # Apply activation if present.
        if op_type == 'fused_add':
            layer_name = kwargs['layer_name']
            if layer_info.op.silu != None:
                assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
                lut = self.activation_lut[layer_name]
                if data_type == '4bit':
                    output_query = output + 8
                elif data_type == '8bit':
                    output_query = output + 128
                else:
                    raise ValueError(f"Unsupported data_type: {data_type!r}")
                output_query = output_query.astype(np.int32)
                output = lut[output_query]
            elif layer_info.op.relu != None:
                output = self._be.clip(output, 0)
            else:
                raise ValueError(f"Fused layer is not implemented: {layer_name!r}")
        # Align dtype to integer.
        output = output.astype(np.int32)
        return output

    fn_fused_add = fn_add

    def fn_mul_add(self, x, **kwargs):
        # attributes
        scale = int(kwargs['layer_info'].CIMA_calc_info.scale)
        scale_shift_num = int(kwargs['layer_info'].CIMA_calc_info.scale_shift_num)
        offset = kwargs['layer_info'].CIMA_calc_info.offset
        dtype = kwargs['layer_info'].CIMA_calc_info.data_type
        # mul add
        x = x.astype(np.int32)
        re = CIMA_mul_add(x, scale=scale, scale_shift_num=scale_shift_num, offset=offset, dtype=dtype)
        return re

    # activate
    def fn_silu(self, x, **kwargs):
        layer_name = kwargs['layer_name']
        data_type = kwargs['layer_info'].CIMA_calc_info.data_type
        assert layer_name in self.activation_lut.keys(), f'{layer_name} not in {self.activation_lut.keys()}!!!'
        lut = self.activation_lut[layer_name]
        #
        output =  CIMA_silu(x, lut, data_type=data_type)
        return output

    def fn_relu(self, x, **kwargs):
        output = self._be.clip(x, 0)
        return output

    # poolings

    def fn_avg_pool2d(self, x, **kwargs):
        pass

    fn_avgpool2d = fn_avg_pool2d


    def fn_max_pool2d(self, x, **kwargs):
        stride = kwargs['layer_info'].op.stride
        padding = kwargs['layer_info'].op.padding
        kernel = kwargs['layer_info'].op.kernel
        # Convert x layout from BHWC to BCHW.
        x = x.transpose(0, 3, 1, 2)
        output = self._pool(self._be.amax, 2, x, stride=stride, padding=padding, kernel=kernel, ceil_mode=0, auto_pad=0, dilation=1)
        # Convert output layout back to BHWC.
        output = output.transpose(0, 2, 3, 1)
        return output

    fn_maxpool2d = fn_max_pool2d


    # resize
    def fn_resize(self, x, **kwargs):
        scale = kwargs['layer_info'].op.scale[2:]
        # Convert x layout from BHWC to BCHW.
        x = x.transpose(0, 3, 1, 2)
        re = super().fn_resize(x, size=None, scale=scale, mode='nearest')
        # Convert output layout back to BHWC.
        re = re.transpose(0, 2, 3, 1)
        return re

    # split
    def fn_split(self, x, **kwargs):
        # attr
        axis = kwargs['layer_info'].op.axis
        # CIMA inference uses BHWC layout (different from PyTorch BCHW).
        if axis == 1:
            axis = 3
        split = kwargs['layer_info'].op.split
        data = x
        re = super().fn_split(data, axis=axis, split=split)
        re_ = []
        for d in re:
            if len(d.shape) == 3:
                d = d.squeeze()
            re_.append(d)
        return re_

