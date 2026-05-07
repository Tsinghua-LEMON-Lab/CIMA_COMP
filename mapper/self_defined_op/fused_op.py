from irtool.ops.conv import *
from irtool.ops.matmul import *
from irtool.ops.math import *
from irtool.ops.trans import *
from irtool.ops.pool import *
from irtool.ops.activate import *
from irtool.core.type_util import to_cls_obj
from irtool.ops.split import *

class fused_conv2d(Conv2dOp):

    op_id = 'fused_conv2d'
    relu = None
    # pool = None
    silu = None
    # split = None
    mul = None
    add = None
    with_bn = False

    def __init__(self, *, relu = None, silu = None, mul=None, add=None, with_bn=False, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('silu', silu)
        self.set_attr('mul', mul)
        self.set_attr('add', add)
        self.set_attr('with_bn', with_bn)
        #     self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)

        if mul != None:
            self.mul = to_cls_obj(mul, cls=MulOp)

        if add != None:
            self.add = to_cls_obj(silu, cls=AddOp)

        if with_bn:
            self.with_bn = with_bn

class fused_fc(MatMulOp):

    op_id = 'fused_fc'
    relu = None
    silu = None

    def __init__(self, *, relu = None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('silu', silu)

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)

class fused_add(AddOp):

    op_id = 'fused_add'
    relu = None
    pool = None
    split = None
    silu = None

    def __init__(self, *, relu = None, pool = None, split =None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('pool', pool)
        self.set_attr('split', split)
        self.set_attr('silu', silu)

        if pool != None:
            self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if split != None:
            self.split = to_cls_obj(split, cls=SplitOp)

        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)

class fused_concat(ConcatOp):

    op_id = 'fused_concat'
    relu = None
    pool = None
    split = None
    silu = None

    def __init__(self, *, relu = None, pool = None, split = None, silu = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('relu', relu)
        self.set_attr('pool', pool)
        self.set_attr('split', split)
        self.set_attr('silu', silu)

        if pool != None:
            self.pool = pool

        if relu != None:
            self.relu = to_cls_obj(relu, cls=ReluOp)

        if split != None:
            self.split = to_cls_obj(split, cls=SplitOp)

        if silu != None:
            self.silu = to_cls_obj(silu, cls=SiluOp)
