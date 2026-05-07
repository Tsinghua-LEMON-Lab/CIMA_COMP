
import json
import numpy as np

class GodeGen:

    def __init__(self) -> None:
        pass

class CompilerToHARNS(object):
    def __init__(self, program_dict_path, SystemCfile, resnet_pair = None):
        self.program_dict_path = program_dict_path
        self.SystemCfile = SystemCfile
        self.interface = {}
        self.resnet_pair = resnet_pair
        self.load_ir()
        self.device_parser()
        self.layer_parser()
        self.outputfile()

    def load_ir(self):
        with open(self.program_dict_path,'r',encoding='utf8') as fp:
            self.program_dict = json.load(fp)

    def device_parser(self):
        chip_devices = self.program_dict['devices']
        NUM_LAYERS = []
        # NUM_ARRAYS = []
        for i in range(len(chip_devices)):
            tile_devices = chip_devices[i]['devices']
            for m in tile_devices:
                attributes = m['attributes']
                # NUM_LAYERS += len(attributes[1]['value_s'])
                layer_num = len(attributes[1]['value_s'])
                if self.resnet_pair != None:
                    for n_ in attributes[1]['value_s']:
                        if  n_ in self.resnet_pair['short_cut'].values():
                            layer_num -= 1
                NUM_LAYERS.append(layer_num)
                #     NUM_ARRAYS.append(attributes[0]['value_i'][k])

        self.interface['NUM_LAYERS'] = NUM_LAYERS
        # self.interface['NUM_ARRAYS'] = NUM_ARRAYS

    def layer_parser(self):

        layer = self.program_dict['layers']
        SIZE_IMAGE = []
        NUM_COPY = []
        NUM_PARAL = []
        CONCAT = []
        NUM_CHANNELS = []
        SIZE_KERNEL = []
        NUM_KERNELS = []
        STRIDE = []
        PADDING_EN = []
        POOLING_MODE = []
        FC_MODE = []
        for i in layer:
            if self.resnet_pair != None and i['name'] in self.resnet_pair['short_cut'].values():
                continue
            for k in i['attributes']:
                if k['key'] == 'mode':
                    mode = k['value_s'][0]
                    break
            if mode == 'conv2d':
                size_image = []
                input_ = i['inputs'][0]['shape']
                size_image.append(input_[1])
                size_image.append(input_[2])
                SIZE_IMAGE.append(size_image)
                NUM_CHANNELS.append(input_[0])
                pad = i['attributes'][0]['value_i']
                stride = i['attributes'][1]['value_i']
                STRIDE.append(stride)
                PADDING_EN.append(pad[0])
                ispool = False
                for j in i['attributes']:
                    if j['key'] == 'pooling':
                        ispool = True
                        pool = j['value_s'][0]
                        if pool == 'max':
                            POOLING_MODE.append('MaxPool')
                        elif pool == 'avg':
                            POOLING_MODE.append('AveragePool')
                if not ispool:
                    POOLING_MODE.append('None')
                weights = i['weights'][0]['shape']
                size_kernel = []
                size_kernel.append(weights[2])
                size_kernel.append(weights[3])
                NUM_COPY.append(1)
                NUM_PARAL.append(1)
                SIZE_KERNEL.append(size_kernel)
                NUM_KERNELS.append(weights[0])
                if self.resnet_pair != None:
                    if i['name'] in self.resnet_pair['short_cut'].keys():
                        for j in layer:
                            if j['name'] == self.resnet_pair['short_cut'][i['name']] :
                                sc_sizekernel = [j['weights'][0]['shape'][2], j['weights'][0]['shape'][3]]
                                sc_numkernel = j['weights'][0]['shape'][0]
                                sc_numchannel = j['inputs'][0]['shape'][0]
                                stride = j['attributes'][1]['value_i']
                                input_ = j['inputs'][0]['shape']
                                sc_size_image = []
                                sc_size_image.append(input_[1])
                                sc_size_image.append(input_[2])
                                break
                        CONCAT.append([1, 1, [sc_numkernel, sc_numchannel,sc_size_image[0],sc_size_image[1]],sc_sizekernel,stride,0])
                    elif i['name'] in self.resnet_pair['jump']:
                        sc_numkernel = weights[0]
                        sc_numchannel = input_[0]
                        CONCAT.append([1, 1, [sc_numkernel, sc_numchannel,size_image[0],size_image[1]],[1,1],[1,1],0])
                    else:
                        CONCAT.append([0])
                else:
                    CONCAT.append([0])
                FC_MODE.append(0)
            elif mode == 'matmul':
                inout = i['weights'][0]['shape']
                SIZE_IMAGE.append([inout[1]])
                NUM_COPY.append(1)
                NUM_PARAL.append(1)
                CONCAT.append([0])
                NUM_CHANNELS.append(0)
                SIZE_KERNEL.append([0])
                NUM_KERNELS.append(inout[0])
                STRIDE.append([0])
                PADDING_EN.append(0)
                POOLING_MODE.append('None')
                FC_MODE.append(1)

        self.interface['NUM_COPY'] = NUM_COPY
        self.interface['NUM_PARAL'] = NUM_PARAL
        self.interface['CONCAT'] = CONCAT
        self.interface['SIZE_IMAGE'] = SIZE_IMAGE
        self.interface['NUM_CHANNELS'] = NUM_CHANNELS
        self.interface['SIZE_KERNEL'] = SIZE_KERNEL
        self.interface['NUM_KERNELS'] = NUM_KERNELS
        self.interface['STRIDE'] = STRIDE
        self.interface['PADDING_EN'] = PADDING_EN
        self.interface['POOLING_MODE'] = POOLING_MODE
        self.interface['FC_MODE'] = FC_MODE

    def outputfile(self):
        self.make_json(self.interface)

    def make_json(self,program_dict):
        json_str = json.dumps(program_dict, cls = self.MyEncoder, indent=4)
        with open(self.SystemCfile, 'w') as json_file:
            json_file.write(json_str)

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
