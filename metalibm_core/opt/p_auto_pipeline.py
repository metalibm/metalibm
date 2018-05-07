# -*- coding: utf-8 -*-
###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import math

from functools import total_ordering

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_hdl_operations import (
    Signal, SubSignalSelection, Concatenation, ZeroExt
)
from metalibm_core.core.advanced_operations import (
    FixedPointPosition
)
from metalibm_core.utility.log_report import Log

CP_LOG2_CST = math.log(2.0)


def cp_log2(v):
    return math.log(v) / CP_LOG2_CST

@total_ordering
class CriticalPath(object):
    def __init__(self, node, value, previous=None):
        self.node = node
        self.value = value
        self.previous = previous

    def __add__(self, cp1):
        return CriticalPath(self.node, self.value + cp1.value, previous=cp1)

    def __lt__(self, cp1):
        return self.value < cp1.value

    def __eq__(self, cp1):
        return self.value == cp1.value

class TimingModel:
    ADD_LEVEL = 2.0
    AND_LEVEL = 1.0
    OR_LEVEL = 1.0
    NOT_LEVEL = 0.5
    XOR_LEVEL = 1.5
    COMP_4TO2 = 3.5
    MUX_LEVEL = 1.5

def tree_cp_eval_value(n, level_value, r=1):
    """ evaluate the timing of a reduction tree from n to r-bit
        whose level timing is level_value """
    return level_value * (cp_log2(n) - cp_log2(r))

def add_cp_eval_optree(optree):
    n = optree.get_precision().get_bit_size()
    return CriticalPath(optree, add_cp_eval_value(n))
def add_cp_eval_value(n):
    return (TimingModel.ADD_LEVEL * cp_log2(n))

def mul_cp_eval_optree(optree):
    n = optree.get_input(0).get_precision().get_bit_size()
    m = optree.get_input(1).get_precision().get_bit_size()
    o = optree.get_precision().get_bit_size()
    return CriticalPath(optree, mul_cp_eval_value(n, m, o))
def mul_cp_eval_value(n, m, o):
    # one level to evaluate partial product
    cp = TimingModel.AND_LEVEL
    # compression tree
    cp += cp_log2(min(m, n)) * TimingModel.COMP_4TO2
    # final addition
    cp += add_cp_eval_value(o)
    return cp

def cmp_cp_eval_optree(optree):
    """ Evaluate timing of a Comparison node """
    if optree.specifier in [Comparison.Equal, Comparison.NotEqual]:
        # Equality is a bitwise XOR followed by a AND reduction tree
        cp = TimingModel.XOR_LEVEL
        cp += tree_cp_eval_value(
            optree.get_precision().get_bit_size(),
            TimingModel.AND_LEVEL
        )
        return CriticalPath(optree, cp)
    else:
        # evaluate as if comp is a subtraction
        return add_cp_eval_optree(optree)

def shift_cp_eval_value(output_size, data_size, amount_size):
    return TimingModel.MUX_LEVEL * min(amount_size, cp_log2(max(data_size, output_size)))

def shift_cp_eval_optree(optree):
    """ Evaluate critical path for binary shift """
    data = optree.get_input(0)
    amount = optree.get_input(1)
    o = data.get_precision().get_bit_size()
    n = data.get_precision().get_bit_size()
    r = data.get_precision().get_bit_size()
    cp = shift_cp_eval_value(o, n, r)
    return CriticalPath(optree, cp)

def complex_timing(local_eval_fct, arity=2):
    def cp_evaluator(evaluator, optree):
        op_cp = max(
            evaluator.evaluate_critical_path(
                optree.get_input(index)
            ) for index in range(arity)
        )
        return local_eval_fct(optree) + op_cp
    return cp_evaluator

def level_timing(level_timing, arity=2):
    def cp_evaluator(evaluator, optree):
        input_cp = [evaluator.evaluate_critical_path(optree.get_input(index)) for index in range(arity)]
        return CriticalPath(optree, level_timing) + max(input_cp)
    return cp_evaluator

def static_timing(timing_value, arity=1):
    def cp_evaluator(evaluator, optree):
        return CriticalPath(timing_value, optree)
    return cp_evaluator

def lzc_cp_eval_optree(optree):
    """ Evaluate CountLeadingZeros critical path """
    op = optree.get_input(0)
    n = op.get_precision().get_bit_size()
    # FIXME: make more precision
    return CriticalPath(
        optree,
        tree_cp_eval_value(n, max(TimingModel.AND_LEVEL, TimingModel.OR_LEVEL))
    )


OPERATION_CLASS_TIMING_MODEL = {
    Addition: complex_timing(add_cp_eval_optree),
    Subtraction: complex_timing(add_cp_eval_optree),
    Comparison: complex_timing(cmp_cp_eval_optree),
    Multiplication: complex_timing(mul_cp_eval_optree),

    CountLeadingZeros: complex_timing(lzc_cp_eval_optree, arity=1),

    BitLogicRightShift: complex_timing(shift_cp_eval_optree),
    BitLogicLeftShift: complex_timing(shift_cp_eval_optree),

    LogicalOr: level_timing(TimingModel.OR_LEVEL),
    LogicalAnd: level_timing(TimingModel.AND_LEVEL),
    LogicalNot: level_timing(TimingModel.NOT_LEVEL, arity=1),

    BitLogicXor: level_timing(TimingModel.XOR_LEVEL),
    BitLogicAnd: level_timing(TimingModel.AND_LEVEL),
    BitLogicOr: level_timing(TimingModel.OR_LEVEL),
    BitLogicNegate: level_timing(TimingModel.NOT_LEVEL, arity=1),

    # transparent operators (no delay)
    ExponentExtraction: level_timing(0.0, arity=1),
    SubSignalSelection: level_timing(0.0, arity=1),
    VectorElementSelection: level_timing(0.0, arity=1),
    Concatenation: level_timing(0.0),
    Conversion: level_timing(0.0, arity=1),

    ZeroExt: level_timing(0.0, arity=1),
}



class Pass_CriticalPathEval(OptreeOptimization):
    """ Evaluate the critical path length (timing) for each node of
        an implementation """
    pass_tag = "critical_path_eval"

    def __init__(self, target):
        OptreeOptimization.__init__(self, "check_target_support", target)
        self.memoization_map = {}

    def memoize_result(self, optree, value):
        """ Stores <value> in association with optree
            and returns it """
        self.memoization_map[optree] = value
        return value

    ## Test if @p optree is supported by self.target
    #  @param optree operation tree to be tested
    #  @param memoization_map memoization map of parallel executions
    #  @param debug enable debug messages
    #  @return boolean support
    def evaluate_critical_path(self, optree):
        """  evalute the critical path of optree towards any input """
        if optree in self.memoization_map:
            return self.memoization_map[optree]
        elif isinstance(optree, Statement):
            for op in optree.get_inputs():
                self.evaluate_critical_path(op)
            return self.memoize_result(optree, None)
        elif isinstance(optree, Variable) or isinstance(optree, Signal):
            return CriticalPath(optree, 0.0)
        elif isinstance(optree, Constant):
            return self.memoize_result(optree, CriticalPath(optree, 0))
        elif isinstance(optree, TypeCast):
            op = optree.get_input(0)
            return self.memoize_result(
                optree,
                CriticalPath(optree, 0.0) + self.evaluate_critical_path(op)
            )
        elif isinstance(optree, SpecificOperation):
            if optree.specifier is SpecificOperation.CopySign:
                op = optree.get_input(0)
                return self.memoize_result(
                    optree,
                    CriticalPath(optree, 0.0) + self.evaluate_critical_path(op)
                )
            else:
                Log.report(Log.Error, "unknown specifier in evaluate_critical_path", optree)
        elif isinstance(optree, ReferenceAssign):
            assign_value = optree.get_input(1)
            assign_var = optree.get_input(0)
            assign_cp = self.evaluate_critical_path(assign_value)
            self.memoize_result(optree, assign_cp)
            return self.memoize_result(assign_var, assign_cp)
        elif isinstance(optree, FixedPointPosition):
            return CriticalPath(None, 0.0)
        elif isinstance(optree, Select):
            cond = optree.get_input(0)
            if_value = optree.get_input(1)
            else_value = optree.get_input(2)
            cond_cp = self.evaluate_critical_path(cond)
            if_value_cp = self.evaluate_critical_path(if_value)
            else_value_cp = self.evaluate_critical_path(else_value)
            select_cp = CriticalPath(optree, TimingModel.MUX_LEVEL) + max(cond_cp, if_value_cp, else_value_cp)
            return self.memoize_result(optree, select_cp)
        elif optree.__class__ in OPERATION_CLASS_TIMING_MODEL:
            return self.memoize_result(
                optree,
                OPERATION_CLASS_TIMING_MODEL[optree.__class__](self, optree)
            )
        else:
            Log.report(Log.Error, "unkwown node in evaluate_critical_path: {}", optree)

    def execute(self, optree):
        self.evaluate_critical_path(optree)
        ordered_list = [op for op in self.memoization_map.keys() if not self.memoization_map[op] is None]
        ordered_list = sorted(ordered_list, key=lambda v: self.memoization_map[v].value, reverse=True)

        longest_path = self.memoization_map[ordered_list[0]]
        current = longest_path
        while current != None:
            print(current.node.get_tag(), current.value)
            current = current.previous




Log.report(LOG_PASS_INFO, "Registering critical_path_eval pass")
# register pass
Pass.register(Pass_CriticalPathEval)
