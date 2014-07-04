# -*- coding: utf-8 -*-

import random

from pythonsollya import *

from core.ml_operations import *
from core.ml_formats import *

if __name__ == "__main__": 
    cte = [SollyaObject(random.randrange(-1000, 1000) / 1000.0) for i in xrange(100)]
    operation_list = {
        Addition: lambda *ops: ops[0] + ops[1], 
        Subtraction: lambda *ops: ops[0] - ops[1], 
        Multiplication: lambda *ops: ops[0] * ops[1], 
        NearestInteger: lambda *ops: nearestint(ops[0])
    }
    specifier_operation_list = {
        FusedMultiplyAdd: {
            FusedMultiplyAdd.Standard: lambda *ops: ops[0] * ops[1] + ops[2], 
            FusedMultiplyAdd.Subtract: lambda *ops: ops[0] * ops[1] - ops[2],  
            FusedMultiplyAdd.Negate: lambda *ops: -ops[0] * ops[1] - ops[2], 
            FusedMultiplyAdd.SubtractNegate: lambda: -ops[0] * ops[1] + ops[2], 
        }
    }
    operation_arity = {
        Addition: 2,
        Subtraction: 2,
        Multiplication: 2,
        NearestInteger: 1,
        FusedMultiplyAdd: 3,
    }

    def get_random_value()
        return SollyaObject(random.randrange(-1000, 1000) / 1000.0)

    def gen_random_tree(size):
        if size == 1:
            cst_value = get_random_value() 
            return Constant(SollyaObject

    random_depth = random.randrange(5, 15)
    op_num = 0



