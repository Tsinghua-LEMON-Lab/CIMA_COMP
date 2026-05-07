import time
import numpy as np

def CIMA_8bit_to_4bit(a):

    assert a <= 127 and a>=-128
    complement_code = {'1000':-8, '1001':-7, '1010': -6, '1011': -5, '1100': -4, '1101': -3, '1110': -2, '1111': -1}
    batch, h, w = a.shape
    batch_high_4bit = []
    batch_low_4bit = []

    for k in range(batch):
        b_high_4bit = []
        b_low_4bit = []
        for i in range(h):
            b_high_4bit_partial = []
            b_low_4bit_partial = []
            for j in range(w):
                # 1) Convert 'a' to 8-bit two's-complement binary string.
                b_ = bin(int(a[i][j]) & 0xff)[2:]
                b_ = list(b_)
                while len(b_) < 8:
                    b_.insert(0, '0')
                b_ = ''.join(b_)
                high_bits = b_[0:4]
                low_bits = b_[4:]
                # 2) For negative numbers, split into high/low 4-bit parts:
                #    - high 4-bit is interpreted as signed (two's complement)
                #    - low 4-bit is interpreted as unsigned, then mapped to signed by (low - 8)
                if a[i][j] < 0:
                    negative_high = complement_code[high_bits]
                    b_high_4bit_partial.append(negative_high)
                else:
                    b_high_4bit_partial.append(int(f'0b{high_bits}', base=2))
                b_low_4bit_partial.append(int(f'0b{low_bits}', base=2) - 8)
            # High/low 4-bit tensors.
            b_high_4bit.append(b_high_4bit_partial)
            b_low_4bit.append(b_low_4bit_partial)
        batch_high_4bit.append(b_high_4bit)
        batch_low_4bit.append(b_low_4bit)

    batch_high_4bit = np.array(batch_high_4bit, dtype=np.int32)
    batch_low_4bit = np.array(batch_low_4bit, dtype=np.int32)

    return batch_high_4bit, batch_low_4bit

def CIMA_array_MAC(input_data, weight_data, *,
                    DAC_noise = 0, conductance_noise = 0,
                    ADC_noise = 0, ADC_offset = 0, ADC_quant_level = 0,
                    scale = 1, offset = 0, scale_shift_num = 0,
                    max_conductance = 36, max_voltage = 0.0957,
                    max_current = None):
    '''
    Analog array inference flow
    '''
    assert max_current is not None, "Missing ADC current range table (max_current)."
    assert input_data.max() <= 7 and input_data.min() >= -8

    # Map input/weight integers to voltages/conductances.
    input_voltage = input_data / 7 * max_voltage
    input_voltage = input_voltage.astype(np.float32)

    weight_conductance = weight_data / 127 * max_conductance
    weight_conductance = weight_conductance.astype(np.float32)

    # Add input DAC noise (voltage domain).
    if DAC_noise != 0:
        DAC_noise = np.abs(input_voltage).max() * DAC_noise
        input_voltage = input_voltage + DAC_noise * np.random.randn(*input_voltage.shape).clip(-3.0, 3.0)

    # Add conductance noise (weight domain).
    # conductance_noise = 2 * max_conductance * 0.05  # weight 5% noise
    if conductance_noise != 0:
        conductance_noise = np.abs(weight_conductance).max() * conductance_noise
        weight_conductance = weight_conductance + conductance_noise * np.random.randn(*weight_conductance.shape).clip(-3.0, 3.0)

    # Compute output current.
    output_current = input_voltage @ weight_conductance

    # Add ADC input noise (current domain).
    # ADC_noise = 2 * max_current[ADC_qunat_level] * 0.05  # output 5% noise
    if ADC_noise != 0:
        ADC_noise = np.abs(output_current).max() * ADC_noise
        output_current = output_current + ADC_noise * np.random.randn(*output_current.shape).clip(-3.0, 3.0)

    # Quantize current to int8 code.
    output_quant = output_current / max_current[ADC_quant_level] * 127
    output_quant = np.around(output_quant).astype(np.int32)

    # ADC quantization offset error.
    if ADC_offset != 0:
        ADC_offset = np.round( np.random.randn(*output_current.shape).clip(-3.0, 3.0) * ADC_offset)
        output_quant = output_quant + ADC_offset

    # Clamp to int8 range.
    output_quant = np.clip(output_quant, a_min=-128, a_max=127)

    # BN calibration: scale + right shift.
    output_quant = (output_quant * scale).astype(np.int32)
    output_quant = output_quant >> scale_shift_num

    # BN calibration: offset.
    output_quant = output_quant + offset

    return output_quant

def CIMA_analog_MAC(input_data, weight_data, *, dtype='4bit',
                    DAC_noise = 0., conductance_noise = 0.,
                    ADC_noise = 0., ADC_offset = 0, ADC_quant_level = 0,
                    scale = 1, offset = 0, scale_shift_num = 0, accumulate_shift_num = 0,
                    max_conductance = 36, max_voltage = 0.0957):
    '''
    input_data: int8/int4. dtype='4bit' => range [-8, 7]; dtype='8bit' => range [-128, 127].
    weight_data: int8 in [-128, 127]. During inference it is mapped to conductance, where 127 corresponds to max_conductance.
    dtype: str. Controls input/output bit-width ('4bit' or '8bit').
    DAC_noise: float. Voltage read noise scale (V), interpreted as std multiplier.
    conductance_noise: float. Conductance noise scale (uS), interpreted as std multiplier.
    ADC_noise: float. ADC front-end noise scale (uA), interpreted as std multiplier.
    ADC_offset: int. ADC code offset error magnitude.
    ADC_quant_level: int. ADC current full-scale range selection: 0..6 (0:32uA, 1:40uA, 2:64uA, 3:80uA, 4:120uA, 5:160uA, 6:200uA).
    scale: int. BN calibration scale.
    offset: int. BN calibration offset.
    scale_shift_num: int. Right shift after scaling (0..15).
    accumulate_shift_num: int. Shift after accumulation; >0 right shift (<=23), <0 left shift (>=-7).
    max_conductance: float. Max mapped conductance (uS).
    max_voltage: float. Max input voltage (V) corresponding to input code 7.
    '''

    assert accumulate_shift_num <= 23 and accumulate_shift_num >= -7
    assert scale_shift_num >= 0 and scale_shift_num <= 15

    if len(input_data.shape) == 3:
        # Input layout transform: B, OC, IC*H*W -> B, IC*H*W, OC
        input_data = input_data.transpose(0, 2, 1)

    # ADC current range table.
    max_current = {0: 32, 1:40, 2:64, 3:80, 4:120, 5:160, 6:200}

    if dtype == '4bit':
        assert input_data.max() <= 7 and input_data.min() >= -8, f'{input_data.max(), input_data.min()}'
        output_quant = CIMA_array_MAC(input_data, weight_data,
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise,
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num,
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)
        # Post-accumulation shift.
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        elif accumulate_shift_num < 0:
            output_quant = output_quant << abs(accumulate_shift_num)

        # Clamp output to 4-bit range.
        output_quant = np.clip(output_quant, a_min=-8, a_max=7)

    elif dtype == '8bit':
        assert input_data.max() <= 127 and input_data.min() >= -128
        input_data_high, input_data_low = CIMA_8bit_to_4bit(input_data)

        # High 4-bit MAC.
        output_quant_hight = CIMA_array_MAC(input_data_high, weight_data,
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise,
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num,
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)
        # Low 4-bit MAC.
        output_quant_low = CIMA_array_MAC(input_data_low, weight_data,
                                        DAC_noise = DAC_noise, conductance_noise = conductance_noise,
                                        ADC_noise = ADC_noise, ADC_offset = ADC_offset, ADC_quant_level = ADC_quant_level,
                                        scale = scale, offset = offset, scale_shift_num = scale_shift_num,
                                        max_conductance = max_conductance, max_voltage = max_voltage, max_current = max_current)

        # Combine high/low parts.
        output_quant = output_quant_hight * 16 + output_quant_low

        # Post-accumulation shift.
        if accumulate_shift_num > 0:
            output_quant = output_quant >> accumulate_shift_num
        else:
            output_quant = output_quant << abs(accumulate_shift_num)

        # Clamp output to 8-bit range.
        output_quant = np.clip(output_quant, a_min=-128, a_max=127)

    else:
        raise ValueError(f"Unsupported bit-width for CIMA analog MAC: {dtype!r}")

    return output_quant

def CIMA_digital_MAC(input_data, weight_data, *, dtype = '8bit', scale=1, offset=0, scale_shift_num=0, accumulate_shift_num = 0):
    assert dtype == '8bit'
    assert input_data.max() <= 127 and input_data.min() >= -128
    assert weight_data.max() <= 127 and weight_data.min() >= -128
    if len(input_data.shape) == 3:
        # Input layout transform: B, OC, IC*H*W -> B, IC*H*W, OC
        input_data = input_data.transpose(0, 2, 1)
    # Compute.
    output_data = input_data @ weight_data
    # BN calibration: scale + right shift.

    output_data = (output_data * scale).astype(np.int32)
    output_data = output_data >> scale_shift_num

    # BN calibration: offset.
    output_data = output_data + offset
    output_data = output_data.astype(np.int32)

    # Post-accumulation shift.
    if accumulate_shift_num > 0:
        output_quant = output_data >> accumulate_shift_num
    else:
        output_quant = output_data << abs(accumulate_shift_num)

    # Clamp output to 8-bit range.
    output_quant = np.clip(output_quant, a_min=-128, a_max=127)

    return output_quant

def CIMA_add(*args, dtype='4bit'):
    input_data = 0
    # Accumulate.
    for i in args:
        input_data += i
    # Clamp output.
    if dtype == '8bit':
        input_data = np.clip(input_data, a_min=-128, a_max=127)
    elif dtype == '4bit':
        input_data = np.clip(input_data, a_min=-8, a_max=7)
    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return input_data

def CIMA_silu(x, lut, data_type='4bit'):
    if data_type == '4bit':
        output_query = x + 8
    elif data_type == '8bit':
        output_query = x + 128
    else:
        raise ValueError(f"Unsupported data_type: {data_type!r}")
    output_query = output_query.astype(np.int32)
    output = lut[output_query]
    # Align dtype to integer.
    output = output.astype(np.int32)
    return output

def CIMA_relu():
    pass

def CIMA_concat(*args, axis=1):
    data = np.concatenate(args, axis=axis)
    return data

def CIMA_mul_add(input_data, *, scale=1, scale_shift_num=0, offset=0, dtype='4bit'):

    output_data = input_data * scale
    output_data = output_data >> scale_shift_num
    output_data = output_data + offset

    if dtype == '8bit':
        output_data = np.clip(output_data, a_min=-128, a_max=127)
    elif dtype == '4bit':
        output_data = np.clip(output_data, a_min=-8, a_max=7)
    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return output_data

def CIMA_data_conversion():
    pass

# Convert feature_map to the next array input (RRAM input packing).
def feature_map_to_input_np_HWC(feature_map, kernel_size, stride, padding, repeat = None, multi_batch=False):
    # feature_map shape = [W_in, H_in, C_in,]
    # array_input shape = [W_out * H_out, C_out]
    if multi_batch:
        if len(feature_map.shape) != 4:
            raise ValueError(
                f"Unsupported feature_map shape: {feature_map.shape}. Expected 4D [b,c,h,w] for multi_batch."
            )
        # Convert to CHW for the packing implementation below.
        feature_map = feature_map.transpose(0,3,1,2)

        batch = feature_map.shape[0]
        array_input = []
        for i in range(batch):
            temp_input = feature_map[i,:,:,:]
            temp_array_input = convert_input_HWC(temp_input, kernel_size,padding,stride)
            temp_array_input = np.expand_dims(temp_array_input,axis=0)
            array_input.append(temp_array_input)
        array_input = np.concatenate(array_input,axis=0)
        assert (len(array_input.shape) == 3)
    else:
        # Convert HWC -> CHW for the packing implementation below.
        feature_map = feature_map.transpose(2,0,1)
        array_input = convert_input_HWC(feature_map,kernel_size,padding,stride)
    if repeat:
        raise ValueError("repeat must be handled outside this function.")
    return array_input

# Convert 3D input [in_channel,height,width] -> 2D array_input [array_height,array_width]
def convert_input_HWC(feature_map,kernel_size,padding,stride):
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    in_channels = feature_map.shape[0]
    feature_in_w = feature_map.shape[1]
    feature_in_h = feature_map.shape[2]
    feature_out_w = int((feature_in_w - kernel_size + 2 * padding) / stride + 1)
    feature_out_h = int((feature_in_h - kernel_size + 2 * padding) / stride + 1)
    feature_map = feature_map_padding(feature_map, padding)
    input_rows = kernel_size ** 2 * in_channels
    output_rows = feature_out_w * feature_out_h
    array_input = np.zeros([input_rows, output_rows])
    idx = 0
    for i in range(feature_out_w):
        for j in range(feature_out_h):
            slide_window = feature_map[:, i * stride:i * stride + kernel_size,
                        j * stride:j * stride + kernel_size]
            # Swap axes to make channel last for flattening.
            slide_window = slide_window.transpose(1,2,0)
            array_input[:, idx] = slide_window.reshape(-1)
            idx += 1
    return array_input

# Pad feature_map.
def feature_map_padding(feature_map, padding):
    # feature_map shape: C, W, H
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    feature_map_pad = np.pad(feature_map, ((0, 0), (padding, padding), (padding, padding)), mode = 'constant')
    return feature_map_pad

# Convert per-array output back to feature_map form.
def output_to_feature_map(out_put, out_h, out_w, multi_batch=False):
    # out_put shape = [W_out * H_out, C_out]
    # feature_map shape = [C_out, W_out, H_out]
    if multi_batch:
        batch = out_put.shape[0]
        channels = out_put.shape[2]
        feature_map = out_put.transpose(0, 2, 1).reshape([batch, channels, out_h, out_w])
    else:
        channels = out_put.shape[1]
        feature_map = out_put.transpose(1, 0).reshape([channels, out_h, out_w])
    return feature_map
