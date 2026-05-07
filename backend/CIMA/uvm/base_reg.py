
def _ns_get(key, namespace, clses, dft=None):
    try:
        return namespace[key]
    except KeyError:
        pass
    for cls in clses:
        try:
            return getattr(cls, key)
        except AttributeError:
            pass
    return dft


class RegType(type):

    def __new__(cls, clsname, bases, namespace):
        fields = _ns_get('FIELDS', namespace, bases)
        nbytes = _ns_get('NBYTES', namespace, bases)
        if fields is not None and nbytes is not None:
            nbits = 0
            names = {}
            total = nbytes * 8
            for name, bits in fields:
                if bits == 0:
                    bits = total - nbits
                assert 0 < bits <= total, f'invalid field {name} bits = {bits}'
                assert nbits + bits <= total, f'field {name} overflowed'
                if name != '_':
                    assert name not in names, f'field {name} duplicated'
                    names[name] = (nbits, bits)
                nbits += bits
            assert 'bits' not in names, 'field \"bits\" is invalid'
            namespace.update(__slots__=('bits', *names), _FIELDS=names)
        return type.__new__(cls, clsname, bases, namespace)


class BaseReg(metaclass=RegType):

    FIELDS = None
    NBYTES = None

    def __init__(self, bits=0, **kwargs):
        self.bits = bits
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name in self._FIELDS.keys():
            pos, bits = self._FIELDS[name]
            return (self.bits >> pos) & ((1 << bits) - 1)

    def __setattr__(self, name, value):
        if name in ('bits',):
            return super().__setattr__(name, value)

        if name in self._FIELDS.keys():
            pos, bits = self._FIELDS[name]
            mask = ((1 << bits) - 1) << pos
            self.bits = (self.bits & ~mask) | (value << pos)

    def to_bytes(self, big_endian=False):
        return self.bits.to_bytes(self.NBYTES,
                                  'big' if big_endian else 'little')

    def from_bytes(self, value, big_endian=False):
        assert len(value) == self.NBYTES, \
                f'bytes {value} length != {self.NBYTES}'
        self.bits = int.from_bytes(value, 'big' if big_endian else 'little')

    def bits_length_of(self, name):
        return self._FIELDS[name][1]

    def to_verilog_hex(self):
        hex_ = hex(self.bits)
        if self.NBYTES == 4:
            head_str = "32'h"
            len_ = 10
        elif self.NBYTES == 2:
            head_str = "16'h"
            len_ = 6
        elif self.NBYTES == 8:
            head_str = "64'h"
            len_ = 18
        else:
            raise ValueError(f'Message translated to English.')

        str_list = list(hex_)
        while len(str_list) < len_:
            str_list.insert(2, '0')
        hex_ = ''.join(str_list)
        return hex_.replace("0x", head_str)

    def get_field_dict(self):
        field_dict = {}
        for k in list(self._FIELDS.keys()):
            k_ = k + f"-[{self._FIELDS[k][0]}:{self._FIELDS[k][0] + self._FIELDS[k][1] - 1}]"
            field_dict[k_] = self.__getattr__(k)
        return field_dict

class Reg32(BaseReg):
    NBYTES = 4

class Reg64(BaseReg):
    NBYTES = 8

class Reg16(BaseReg):
    NBYTES = 2
