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
import collections

from functools import total_ordering

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import (
    Addition, Subtraction, Comparison, Multiplication, Negation,

    MantissaExtraction, CountLeadingZeros,
    BitLogicRightShift, BitLogicLeftShift,

    LogicalOr, LogicalAnd, LogicalNot,
    BitLogicXor, BitLogicAnd, BitLogicOr, BitLogicNegate,
    Select, Test,

    ExponentExtraction,
    VectorElementSelection, Conversion,

    Min, Max,

    Statement, Variable,
    TypeCast, Constant,
    SpecificOperation, ReferenceAssign,
)
from metalibm_core.core.ml_hdl_operations import (
    Signal, SubSignalSelection, Concatenation, ZeroExt
)
from metalibm_core.core.advanced_operations import (
    FixedPointPosition
)
from metalibm_core.core.ml_formats import ML_AbstractFormat
from metalibm_core.utility.log_report import Log

CP_LOG2_CST = math.log(2.0)


def cp_log2(v):
    """ Dummy implementation of log2(x) """
    return math.log(v) / CP_LOG2_CST

def cp_add_stage_aware(cp0, cp1):
    """ Addition of two CriticalPath object in stage-aware mode """
    assert cp0.stage >= cp1.stage
    stage = cp0.stage
    if cp1.stage < cp0.stage:
        comb_latency = cp0.comb_latency
    elif cp1.stage == cp0.stage:
        comb_latency = cp0.comb_latency + cp1.comb_latency
    else:
        raise Exception()
    return CriticalPath(cp0.node, comb_latency, previous=cp1, combinatorial=cp0.combinatorial)


def cp_add_combinatorial(cp0, cp1):
    """ Addition of two CriticalPath objects in combinatorial mode """
    comb_latency = cp0.comb_latency + cp1.comb_latency
    return CriticalPath(cp0.node, comb_latency, previous=cp1, combinatorial=cp0.combinatorial)

def cp_lt_stage_aware(cp0, cp1):
    """ Less-than comparison of two CriticalPath objects in stage-aware mode """
    if cp0.stage is None:
        # CriticalPath with None stage is always shorter than any other
        # critical path
        return True
    elif cp1.stage is None:
        return False
    return (cp0.stage < cp1.stage) or (cp0.stage == cp1.stage and cp0.comb_latency < cp1.comb_latency)

def cp_eq_stage_aware(cp0, cp1):
    """ Equal comparison of two CriticalPath objects in stage-aware mode """
    return cp0.stage == cp1.stage and cp0.comb_latency == cp1.comb_latency

def cp_lt_combinatorial(cp0, cp1):
    """ Less-than comparison of two CriticalPath objects in combinatorial mode """
    return cp0.comb_latency < cp1.comb_latency

def cp_eq_combinatorial(cp0, cp1):
    """ Equal comparison of two CriticalPath objects in combinatorial mode """
    return cp0.comb_latency == cp1.comb_latency

@total_ordering
class CriticalPath(object):
    """ object to store critical path:
        This class distinguishes two modes:
        - combinatorial is a mode where node's stage is not taken into account,
        the whole meta entity is considered as combinatorial (not pipelined) and
        the critical path is constructed from input to output.
        - stage-aware is a mode where a node's pipeline stage (init_stage) is
        taken into account. The meta entity is considered stage-by-stage to
        compute a critical path for each stage
        @param node leaf node
        @param value path timing value
        @param previous previous node in the chain
        @param combinatorial select between stage-aware / combinatorial path
    """

    def __init__(self, node, comb_latency, previous=None, combinatorial=False):
        self.node = node
        # combinatorial latency
        self.comb_latency = comb_latency
        self.previous = previous
        self.combinatorial = combinatorial

    def __repr__(self):
        return "[{}, {}]".format(self.stage, self.comb_latency)
    def __str__(self):
        return "[{}, {}]".format(self.stage, self.comb_latency)

    @property
    def stage(self):
        """ return pipeline stage associated with the head of @p self
            critical path object """
        if self.node is None:
            if self.previous is None:
                return None
            return self.previous.stage
        return self.node.attributes.init_stage

    def __add__(self, cp1):
        """ non commutative addition of 2 critical paths
            left-hand-side must be at a latter (or equal) stage compared to cp1
            """
        if self.combinatorial:
            return cp_add_combinatorial(self, cp1)
        else:
            return cp_add_stage_aware(self, cp1)

    def __lt__(self, cp1):
        """ Comparison less-than of two critical path objects: self and cp1 """
        if self.combinatorial:
            return cp_lt_combinatorial(self, cp1)
        else:
            return cp_lt_stage_aware(self, cp1)

    def __eq__(self, cp1):
        """ Comparison equal of two critical path objects: self and cp1 """
        if self.combinatorial:
            return cp_eq_combinatorial(self, cp1)
        else:
            return cp_eq_stage_aware(self, cp1)

class TimingModel:
    """ Basic timing model for various RTL "gates" """
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

def add_cp_eval_optree(optree, combinatorial):
    """ Evaluation of the critical path object associated with
        @p optree Addition node in @p combinatorial mode """
    n = optree.get_precision().get_bit_size()
    return CriticalPath(optree, add_cp_eval_value(n), combinatorial=combinatorial)
def add_cp_eval_value(n):
    """ Evaluation of the numerical value of critical-path latency
        for an n-bit addition """
    return (TimingModel.ADD_LEVEL * cp_log2(n))

def mul_cp_eval_optree(optree, combinatorial):
    """ Evaluation of the critical path object associated with
        @p optree Multiplication node in @p combinatorial mode """
    n = optree.get_input(0).get_precision().get_bit_size()
    m = optree.get_input(1).get_precision().get_bit_size()
    o = optree.get_precision().get_bit_size()
    return CriticalPath(optree, mul_cp_eval_value(n, m, o), combinatorial=combinatorial)
def mul_cp_eval_value(n, m, o):
    """ Evaluation of the numerical value of critical-path latency
        for an n-bit x m-bit -> o-bit multiplication  """
    # one level to evaluate partial product
    cp = TimingModel.AND_LEVEL
    # compression tree
    cp += cp_log2(min(m, n)) * TimingModel.COMP_4TO2
    # final addition
    cp += add_cp_eval_value(o)
    return cp

def cmp_cp_eval_optree(optree, combinatorial):
    """ Evaluate timing of a Comparison node """
    if optree.specifier in [Comparison.Equal, Comparison.NotEqual]:
        # Equality is a bitwise XOR followed by a AND reduction tree
        cp = TimingModel.XOR_LEVEL
        cp += tree_cp_eval_value(
            optree.get_precision().get_bit_size(),
            TimingModel.AND_LEVEL
        )
        return CriticalPath(optree, cp, combinatorial=combinatorial)
    else:
        # evaluate as if comp is a subtraction
        return add_cp_eval_optree(optree, combinatorial)

def shift_cp_eval_value(output_size, data_size, amount_size):
    """ Evaluation of the numerical value of critical-path latency
        for an data_size-bit [<<|>>] amount_size-bit -> output_size-bit shift
    """
    num_levels = min(amount_size, cp_log2(max(data_size, output_size)))
    return TimingModel.MUX_LEVEL * num_levels

def shift_cp_eval_optree(optree, combinatorial):
    """ Evaluate critical path for binary shift """
    data = optree.get_input(0)
    amount = optree.get_input(1)
    o = data.get_precision().get_bit_size()
    n = data.get_precision().get_bit_size()
    r = data.get_precision().get_bit_size()
    cp = shift_cp_eval_value(o, n, r)
    return CriticalPath(optree, cp, combinatorial=combinatorial)

def mant_extraction_cp_eval_optree(optree, combinatorial):
    # we assume mantissa extraction is an OR on the exponent bit
    # to determine the implicit bit value
    n = optree.get_input(0).get_precision().get_base_format().get_exponent_size()
    cp = tree_cp_eval_value(
        n,
        TimingModel.OR_LEVEL
    )
    return CriticalPath(optree, cp, combinatorial=combinatorial)


def complex_timing(local_eval_fct, arity=None):
    def cp_evaluator(evaluator, optree, combinatorial=False):
        op_cp = max(
            evaluator.evaluate_critical_path(
                optree.get_input(index),
                combinatorial=combinatorial
            ) for index in range(optree.arity if arity is None else arity)
        )
        return local_eval_fct(optree, combinatorial) + op_cp
    return cp_evaluator

def level_timing(level_timing, arity=2):
    """ build a CriticalPath evaluator which adds @p level_timing
        as local added latency to the maximum of inputs critical path
        latencies """
    def cp_evaluator(evaluator, optree, combinatorial=False):
        """ local critical-path evaluation for level_timing constructor """
        input_cp = [
            evaluator.evaluate_critical_path(
                optree.get_input(index),
                combinatorial=combinatorial
            ) for index in range(arity)]
        return CriticalPath(optree, level_timing,
                            combinatorial=combinatorial) + max(input_cp)
    return cp_evaluator

def static_timing(timing_value, arity=1):
    """ build a critical_path evaluator which always returns a constant latency
        equals to @p timing_value """
    def cp_evaluator(evaluator, optree, combinatorial=False):
        """ local critical-path evaluation for static_timing constructor """
        return CriticalPath(timing_value, optree, combinatorial)
    return cp_evaluator

def lzc_cp_eval_optree(optree, combinatorial):
    """ Evaluate CountLeadingZeros critical path """
    op = optree.get_input(0)
    size = op.get_precision().get_bit_size()
    # FIXME: make more precision
    return CriticalPath(
        optree,
        tree_cp_eval_value(size, max(TimingModel.AND_LEVEL, TimingModel.OR_LEVEL)),
        combinatorial=combinatorial
    )

def test_cp_eval_optree(optree, combinatorial):
    """ Evaluate critical path for Test operation class """
    if optree.specifier is Test.IsZero:
        # floating-point is zero comparison is assumed to be equivalent
        # to a AND reduction on exponent + mantissa size
        operand = optree.get_input(0)
        base_precision = operand.get_precision().get_base_format()
        return CriticalPath(
            optree,
            tree_cp_eval_value(base_precision.get_exponent_size() + base_precision.get_mantissa_size(), TimingModel.AND_LEVEL),
            combinatorial=combinatorial
        )
    elif optree.specifier in [Test.IsNaN, Test.IsInfty, Test.IsPositiveInfty, Test.IsNegativeInfty]:
        # floating-point is nan/(+/-)infty test is assumed to be equivalent
        # to a AND reduction on exponent and an OR reduction on mantissa
        # plus neglected terms
        operand = optree.get_input(0)
        base_precision = operand.get_precision().get_base_format()
        return CriticalPath(
            optree,
            TimingModel.AND_LEVEL + max(
                tree_cp_eval_value(base_precision.get_exponent_size(), TimingModel.AND_LEVEL),
                tree_cp_eval_value(base_precision.get_mantissa_size(), TimingModel.OR_LEVEL)
            ),
            combinatorial=combinatorial
        )

    else:
        Log.report(Log.Error, "unknown Test specifier in test_cp_eval_optree: {}", optree)


OPERATION_CLASS_TIMING_MODEL = {
    Addition: complex_timing(add_cp_eval_optree),
    Subtraction: complex_timing(add_cp_eval_optree),
    Comparison: complex_timing(cmp_cp_eval_optree),
    Multiplication: complex_timing(mul_cp_eval_optree),
    # FIXME: assuming Negation is an Incrementer whose timing is equivalent
    # to an addition
    Negation: complex_timing(add_cp_eval_optree, arity=1),

    MantissaExtraction: complex_timing(mant_extraction_cp_eval_optree, arity=1),

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

    Test: complex_timing(test_cp_eval_optree),

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

    def memoize_result(self, optree, value, combinatorial=None):
        """ Stores <value> in association with optree
            and returns it """
        if combinatorial is None:
            combinatorial = value.combinatorial
        self.memoization_map[(optree, combinatorial)] = value
        return value

    ## Test if @p optree is supported by self.target
    #  @param optree operation tree to be tested
    #  @param memoization_map memoization map of parallel executions
    #  @param debug enable debug messages
    #  @return boolean support
    def evaluate_critical_path(self, optree, combinatorial=False):
        """  evalute the critical path of optree towards any input
            in mode @p combinatorial """
        if optree in self.memoization_map:
            return self.memoization_map[(optree, combinatorial)]
        elif isinstance(optree, Statement):
            for op in optree.get_inputs():
                self.evaluate_critical_path(op, combinatorial=combinatorial)
            return self.memoize_result(optree, None, combinatorial=combinatorial)
        elif isinstance(optree, Variable) or isinstance(optree, Signal):
            return CriticalPath(optree, 0.0, combinatorial=combinatorial)
        elif isinstance(optree, Constant):
            return self.memoize_result(optree, CriticalPath(optree, 0, combinatorial=combinatorial))
        elif isinstance(optree, TypeCast):
            op = optree.get_input(0)
            return self.memoize_result(
                optree,
                CriticalPath(optree, 0.0, combinatorial=combinatorial) + self.evaluate_critical_path(op, combinatorial=combinatorial)
            )
        elif isinstance(optree, SpecificOperation):
            if optree.specifier is SpecificOperation.CopySign:
                op = optree.get_input(0)
                return self.memoize_result(
                    optree,
                    CriticalPath(optree, 0.0, combinatorial=combinatorial) + self.evaluate_critical_path(op, combinatorial=combinatorial)
                )
            else:
                Log.report(Log.Error, "unknown specifier in evaluate_critical_path", optree)
        elif isinstance(optree, ReferenceAssign):
            assign_value = optree.get_input(1)
            assign_var = optree.get_input(0)
            assign_cp = self.evaluate_critical_path(assign_value, combinatorial=combinatorial)
            self.memoize_result(optree, assign_cp)
            return self.memoize_result(assign_var, assign_cp)
        elif isinstance(optree, FixedPointPosition):
            return CriticalPath(None, 0.0)
        elif isinstance(optree, Min) or isinstance(optree, Max):
            lhs = optree.get_input(0)
            rhs = optree.get_input(1)
            lhs_cp = self.evaluate_critical_path(lhs, combinatorial=combinatorial)
            rhs_cp = self.evaluate_critical_path(rhs, combinatorial=combinatorial)
            # evalute opree as an Addition, to get Comparison (Subtraction) timing
            comp_cp_value = add_cp_eval_value(optree.get_precision().get_bit_size())
            minmax_cp = CriticalPath(optree, TimingModel.MUX_LEVEL + comp_cp_value, combinatorial=combinatorial) + max(lhs_cp, rhs_cp)
            return self.memoize_result(optree, minmax_cp)
        elif isinstance(optree, Select):
            cond = optree.get_input(0)
            if_value = optree.get_input(1)
            else_value = optree.get_input(2)
            cond_cp = self.evaluate_critical_path(cond, combinatorial=combinatorial)
            if_value_cp = self.evaluate_critical_path(if_value, combinatorial=combinatorial)
            else_value_cp = self.evaluate_critical_path(else_value, combinatorial=combinatorial)
            select_cp = CriticalPath(optree, TimingModel.MUX_LEVEL, combinatorial=combinatorial) + max(cond_cp, if_value_cp, else_value_cp)
            return self.memoize_result(optree, select_cp)
        elif optree.__class__ in OPERATION_CLASS_TIMING_MODEL:
            return self.memoize_result(
                optree,
                OPERATION_CLASS_TIMING_MODEL[optree.__class__](self, optree, combinatorial=combinatorial)
            )
        else:
            Log.report(Log.Error, "unkwown node in evaluate_critical_path: {}", optree)


    def execute(self, optree):
        """ Execute Pass_CriticalPathEval on node @p optree """
        self.evaluate_critical_path_by_stage(optree)

        self.evaluate_critical_path_combinatorial(optree)

    def evaluate_critical_path_by_stage(self, optree):
        """ Evaluate critical-path starting with node @p optree
            in stage-by-stage mode """
        self.evaluate_critical_path(optree, combinatorial=False)

        valid_entry_list = [op for (op, combinatorial) in self.memoization_map.keys() if not combinatorial and not self.memoization_map[(op, combinatorial)] is None]

        entry_by_stage = collections.defaultdict(list)
        for op in valid_entry_list:
            cp_value = self.memoization_map[(op, False)]
            entry_by_stage[cp_value.stage].append(op)
        ordered_list_by_stage = {}
        for stage_key in entry_by_stage:
            ordered_list = sorted(entry_by_stage[stage_key], key=lambda v: self.memoization_map[(v, False)].comb_latency, reverse=True)
            ordered_list_by_stage[stage_key] = ordered_list

            if not len(ordered_list):
                print("ordered list is empty for stage {}".format(stage_key))
                continue
            print("longest path for stage {}:".format(stage_key))
            longest_path_exit = self.memoization_map[(ordered_list[0], False)]
            longest_path = linearize_critical_path(longest_path_exit, min_stage=stage_key)
            display_path_info(longest_path)
            print("\n")

    def evaluate_critical_path_combinatorial(self, optree):
        """ Evaluate critical-path starting with node @p optree
            in combinatorial mode """
        combinatorial_flag = True
        self.evaluate_critical_path(optree, combinatorial=combinatorial_flag)

        valid_entry_list = [op for (op, combinatorial) in self.memoization_map.keys() if combinatorial and not self.memoization_map[(op, combinatorial)] is None]

        ordered_list = sorted(valid_entry_list, key=lambda v: self.memoization_map[(v, combinatorial_flag)].comb_latency, reverse=True)

        longest_path_exit = self.memoization_map[(ordered_list[0], combinatorial_flag)]
        longest_path = linearize_critical_path(longest_path_exit, min_stage=None)
        print("longest combinatorial path:")
        display_path_info(longest_path)
        print("\n")


def linearize_critical_path(cp, min_stage=None):
    """ Recursively build the list of nodes starting at @p cp.node """
    current = cp
    path = []
    # reversing list to start longest path from the root
    while not current is None and (min_stage is None or current.stage >= min_stage):
        path.append(current)
        current = current.previous
    return path


def display_path_info(path):
    """ Display information on the path which is the ordered
        list of nodes of the longest critical path """
    def generate_op_name(optree):
        """ generate a pretty operation description with class name
            and basic information size """
        op_class_name = optree.name
        op_size = optree.get_precision().get_bit_size()
        def get_op_size(index):
            """ Retrieve optree's index-th operand output bit size """
            operand_format = optree.get_input(index).get_precision()
            if isinstance(operand_format, ML_AbstractFormat):
                return None
            return operand_format.get_bit_size()
        if isinstance(optree, Addition):
            op_desc = "{:4} <- {} + {}".format(op_size, get_op_size(0), get_op_size(1))
        elif isinstance(optree, Multiplication):
            op_desc = "{:4} <- {} * {}".format(op_size, get_op_size(0), get_op_size(1))
        elif isinstance(optree, Subtraction):
            op_desc = "{:4} <- {} - {}".format(op_size, get_op_size(0), get_op_size(1))
        else:
            op_desc = "{:4}".format(op_size)
        return "{:>30}{:2}{:20}".format(op_class_name, optree.attributes.init_stage, "[" + op_desc + "]")

    for critical_path in path[::-1]:
        def name_cleaner(cp):
            """ generate cleaned name (converting None) for
                CriticalPath object cp """
            tag = cp.node.get_tag()
            if tag is None:
                return "<undef>"
            return tag
        print("{:30} {}> = {:.2f}".format(
            name_cleaner(critical_path),
            generate_op_name(critical_path.node),
            critical_path.comb_latency))




Log.report(LOG_PASS_INFO, "Registering critical_path_eval pass")
# register pass
Pass.register(Pass_CriticalPathEval)
