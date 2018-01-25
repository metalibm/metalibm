# -*- coding: utf-8 -*-
# optimization pass to expand multi-precision node to simple precision
# implementation

import sollya

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO

from metalibm_core.core.ml_operations import (
    Addition, Constant, Multiplication
)
from metalibm_core.opt.ml_blocks import (
    Add222, Add122, Add212, Add211, Mul212, Mul211, Mul222
)

from metalibm_core.utility.log_report import Log


def dirty_multi_node_expand(node, precision, mem_map=None):
    """ Dirty expand node into Hi and Lo part, storing
        already processed temporary values in mem_map """
    mem_map = mem_map or {}
    if node in mem_map:
        return mem_map[node]
    elif isinstance(node, Constant):
        value = node.get_value()
        value_hi = sollya.round(value, precision.sollya_object, sollya.RN)
        value_lo = sollya.round(value - value_hi, precision.sollya_object, sollya.RN)
        ch = Constant(value_hi,
                      tag=node.get_tag() + "hi",
                      precision=precision)
        cl = Constant(value_lo,
                      tag=node.get_tag() + "lo",
                      precision=precision
                      ) if value_lo != 0 else None
        if cl is None:
            Log.report(Log.Info, "simplified constant")
        result = ch, cl
        mem_map[node] = result
        return result
    else:
        # Case of Addition or Multiplication nodes:
        # 1. retrieve inputs
        # 2. dirty convert inputs recursively
        # 3. forward to the right metamacro
        assert isinstance(node, Addition) or isinstance(node, Multiplication)
        lhs = node.get_input(0)
        rhs = node.get_input(1)
        op1h, op1l = dirty_multi_node_expand(lhs, precision, mem_map)
        op2h, op2l = dirty_multi_node_expand(rhs, precision, mem_map)
        if isinstance(node, Addition):
            result = Add222(op1h, op1l, op2h, op2l) \
                    if op1l is not None and op2l is not None \
                    else Add212(op1h, op2h, op2l) \
                    if op1l is None and op2l is not None \
                    else Add212(op2h, op1h, op1l) \
                    if op2l is None and op1l is not None \
                    else Add211(op1h, op2h)
            mem_map[node] = result
            return result

        elif isinstance(node, Multiplication):
            result = Mul222(op1h, op1l, op2h, op2l) \
                    if op1l is not None and op2l is not None \
                    else Mul212(op1h, op2h, op2l) \
                    if op1l is None and op2l is not None \
                    else Mul212(op2h, op1h, op1l) \
                    if op2l is None and op1l is not None \
                    else Mul211(op1h, op2h)
            mem_map[node] = result
            return result


class Pass_ExpandMultiPrecision(OptreeOptimization):
    """ Generic Multi-Precision expansion pass """
    pass_tag = "expand_multi_precision"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "multi-precision expansion pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}


    def expand_node(self, optree):
        return optree

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.expand_node(optree)


Log.report(LOG_PASS_INFO, "Registering expand_multi_precision pass")
# register pass
Pass.register(Pass_ExpandMultiPrecision)
