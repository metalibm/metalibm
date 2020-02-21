# -*- coding: utf-8 -*-
###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# Description: optimization pass to legalize vector of virtual boolean element
###############################################################################

import sollya
# must not import inf or sup from sollya, as overloaded functions
# are imported from core.meta_interval
from sollya import SollyaObject

from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO

from metalibm_core.opt.node_transformation import Pass_NodeTransformation
from metalibm_core.opt.opt_utils import (
    forward_attributes,
)


from metalibm_core.core.ml_formats import ml_infty
from metalibm_core.core.ml_operations import (
    is_leaf_node,
    Min, Max, Constant, Test, Comparison, ConditionBlock,
    LogicOperation, LogicalAnd, LogicalOr, LogicalNot,
    Statement,
)
from metalibm_core.core.legalizer import is_constant
from metalibm_core.core.meta_interval import MetaInterval, MetaIntervalList, inf, sup

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi

LOG_VERBOSE_NUMERICAL_SIMPLIFICATION = Log.LogLevel("NumericalSimplificationVerbose")

from metalibm_core.opt.opt_utils import is_false, is_true

def is_simplifiable_to_cst(node):
    """ node can be simplified to a constant """
    node_interval = node.get_interval()
    if node_interval is None or isinstance(node, Constant):
        return False
    elif isinstance(node_interval, SollyaObject) and node_interval.is_range():
        return sollya.inf(node_interval) == sollya.sup(node_interval)
    elif isinstance(node_interval, (MetaInterval, MetaIntervalList)):
        return not node_interval.is_empty and (node_interval.inf == node_interval.sup)
    else:
        return False


def generate_uniform_cst(value, precision):
    if not precision.is_vector_format():
        return Constant(value, precision=precision)
    else:
        return Constant([value] * precision.get_vector_size(), precision=precision)

def simplify_logical_op(node):
    """ Simplify LogicOperation node """
    if isinstance(node, LogicalAnd):
        lhs = node.get_input(0)
        rhs = node.get_input(1)
        if is_false(lhs) or is_false(rhs):
            # FIXME: manage vector constant
            cst = generate_uniform_cst(False, node.get_precision())
            return cst
        elif is_true(lhs) and is_true(rhs):
            # FIXME: manage vector constant
            cst = generate_uniform_cst(True, node.get_precision())
            return cst
        elif is_true(lhs):
            return rhs
        elif is_true(rhs):
            return lhs
    elif isinstance(node, LogicalOr):
        lhs = node.get_input(0)
        rhs = node.get_input(1)
        if is_false(lhs) and is_false(rhs):
            # FIXME: manage vector constant
            cst = generate_uniform_cst(False, node.get_precision())
            return cst
        elif is_true(lhs) or is_true(rhs):
            # FIXME: manage vector constant
            cst = generate_uniform_cst(True, node.get_precision())
            return cst
        elif is_constant(lhs) and is_constant(rhs):
            if node.get_precision().is_vector_format():
                return Constant(
                    [(sub_lhs or sub_rhs) for sub_lhs, sub_rhs in zip(lhs.get_value(), rhs.get_value())],
                    precision=node.get_precision())
            else:
                return Constant(lhs.get_value() or rhs.get_value(), precision=node.get_precision())
    elif isinstance(node, LogicalNot):
        op = node.get_input(0)
        if is_constant(op):
            # only support simplification of LogicalNot(Constant)
            if not op.get_precision().is_vector_format():
                return Constant(not op.get_value(), precision=node.get_precision())
            else:
                return Constant(
                    [not elt_value for elt_value in op.get_value()]
                    , precision=node.get_precision())
    return None


def is_simplifiable_min(node, lhs, rhs):
    if not isinstance(node, Min):
        return False
    lhs_interval = lhs.get_interval()
    rhs_interval = rhs.get_interval()
    if lhs_interval is None or rhs_interval is None:
        return False
    elif sup(lhs_interval) < inf(rhs_interval):
        return lhs
    elif sup(rhs_interval) < inf(lhs_interval):
        return rhs
    else:
        return False

def is_simplifiable_max(node, lhs, rhs):
    if not isinstance(node, Max):
        return False
    assert isinstance(node, Max)
    lhs_interval = lhs.get_interval()
    rhs_interval = rhs.get_interval()
    if lhs_interval is None or rhs_interval is None:
        return False
    elif sup(lhs_interval) < inf(rhs_interval):
        return rhs
    elif sup(rhs_interval) < inf(lhs_interval):
        return lhs
    else:
        return False

def simplify_condition_block(node):
    assert isinstance(node, ConditionBlock)
    cond = node.get_input(0)
    if isinstance(cond, Constant):
        if cond.get_value():
            return Statement(
                node.get_pre_statement(),
                node.get_input(1)
            )
        elif len(node.inputs) >= 3:
            return Statement(
                node.get_pre_statement(),
                node.get_input(2)
            )

    return None


class BooleanValue:
    class AlwaysTrue: pass
    class AlwaysFalse: pass
    class Indecisive: pass


def is_simplifiable_cmp(node, lhs, rhs):
    """ test wheter the Comparison node(lhs, rhs) can be simplified """
    lhs_interval = lhs.get_interval()
    rhs_interval = rhs.get_interval()
    if lhs_interval is None or rhs_interval is None:
        return BooleanValue.Indecisive
    elif node.specifier is Comparison.Equal:
        return BooleanValue.Indecisive
    elif node.specifier is Comparison.NotEqual:
        return BooleanValue.Indecisive
    elif node.specifier is Comparison.Less:
        if sup(lhs_interval) < inf(rhs_interval):
            return BooleanValue.AlwaysTrue
        elif inf(lhs_interval) >= sup(rhs_interval):
            return BooleanValue.AlwaysFalse
        else:
            return BooleanValue.Indecisive
    elif node.specifier is Comparison.LessOrEqual:
        if sup(lhs_interval) <= inf(rhs_interval):
            return BooleanValue.AlwaysTrue
        elif inf(lhs_interval) > sup(rhs_interval):
            return BooleanValue.AlwaysFalse
        else:
            return BooleanValue.Indecisive
    elif node.specifier is Comparison.Greater:
        if inf(lhs_interval) > sup(rhs_interval):
            return BooleanValue.AlwaysTrue
        elif sup(lhs_interval) <= inf(rhs_interval):
            return BooleanValue.AlwaysFalse
        else:
            return BooleanValue.Indecisive
    elif node.specifier is Comparison.GreaterOrEqual:
        if inf(lhs_interval) >= sup(rhs_interval):
            return BooleanValue.AlwaysTrue
        elif sup(lhs_interval) < inf(rhs_interval):
            return BooleanValue.AlwaysFalse
        else:
            return BooleanValue.Indecisive
    return BooleanValue.Indecisive


def is_simplifiable_test(node, simp_node_inputs):
    """ test it the Test node(simp_node_inputs) is simplifiable """
    if node.specifier is Test.IsZero:
        op = simp_node_inputs[0] or node.get_input(0)
        if not op.get_interval() is None:
            if not 0 in op.get_interval():
                # test is always False
                return BooleanValue.AlwaysFalse
            elif inf(op.get_interval()) == sup(op.get_interval()) == 0:
                # test is always True
                return BooleanValue.AlwaysTrue
        # nothing to say
        return BooleanValue.Indecisive
    elif node.specifier is Test.IsSubnormal:
        op = simp_node_inputs[0] or node.get_input(0)
        if not op.get_interval() is None and not op.get_precision() is None:
            op_precision = op.get_precision()
            if op.get_precision().is_vector_format():
                elt_precision = op_precision.get_scalar_format()
            else:
                elt_precision = op_precision
            if inf(op.get_interval()) >= elt_precision.get_min_normal_value():
                # test always False
                return BooleanValue.AlwaysFalse
            elif sup(op.get_interval()) <= elt_precision.get_max_subnormal_value():
                # test always True
                return BooleanValue.AlwaysTrue
        # nothing to say
        return BooleanValue.Indecisive
    elif node.specifier is Test.IsInfty or node.specifier is Test.IsInfOrNaN:
        op = simp_node_inputs[0] or node.get_input(0)
        if not op.get_interval() is None:
            if sup(op.get_interval()) < ml_infty:
                # test always False
                return BooleanValue.AlwaysFalse
            elif inf(op.get_interval()) == ml_infty:
                # test always True
                return BooleanValue.AlwaysTrue
    else:
        # nothing to say
        return BooleanValue.Indecisive


def is_simplifiable_logical_op(node, node_inputs):
    """ test if the LogicOperation node(node_inputs) is simplifiable """
    if isinstance(node, (LogicalAnd, LogicalOr)):
        lhs = node_inputs[0]
        rhs = node_inputs[1]
        return is_constant(lhs) or is_constant(rhs)
    elif isinstance(node, LogicalNot):
        return is_constant(node_inputs[0])
    else:
        return False

class NumericalSimplifier:
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def is_simplifiable(self, node):
        if is_simplifiable_to_cst(node):
            return True
        elif isinstance(node, Min) and is_simplifiable_min(node, node.get_input(0), node.get_input(1)):
            return True
        elif isinstance(node, Max) and is_simplifiable_max(node, node.get_input(0), node.get_input(1)):
            return True
        elif isinstance(node, Comparison) and not is_simplifiable_cmp(node, node.get_input(0), node.get_input(1)) is BooleanValue.Indecisive:
            return True
        elif isinstance(node, Test) and not is_simplifiable_test(node, node.inputs) is BooleanValue.Indecisive:
            return True
        elif isinstance(node, LogicOperation) and is_simplifiable_logical_op(node, node.inputs):
            return True
        elif isinstance(node, ConditionBlock):
            result = simplify_condition_block(node)
            pass
        else:
            return False


    def simplify(self, node):
        def get_node_input(index):
            # look for input into simpifield list
            # and return directly node input if simplified input is None
            return node.get_input(index)

        result = None
        if node in self.memoization_map:
            return self.memoization_map[node]
        else:
            if not is_leaf_node(node):
                for index, op in enumerate(node.inputs):
                    new_op = self.simplify(op)
                    # replacing modified inputs
                    if not new_op is None:
                        node.set_input(index, new_op)
            if is_simplifiable_to_cst(node):
                new_node = Constant(
                    sollya.inf(node.get_interval()),
                    precision=node.get_precision()
                )
                forward_attributes(node, new_node)
                result = new_node
            elif isinstance(node, Min):
                simplified_min = is_simplifiable_min(node, get_node_input(0), get_node_input(1))
                if simplified_min:
                    result = simplified_min
            elif isinstance(node, Max):
                simplified_max = is_simplifiable_max(node, get_node_input(0), get_node_input(1))
                if simplified_max:
                    result = simplified_max
            elif isinstance(node, Comparison):
                cmp_value = is_simplifiable_cmp(node, get_node_input(0), get_node_input(1))
                if cmp_value is BooleanValue.AlwaysTrue:
                    result = generate_uniform_cst(True, node.get_precision())
                elif cmp_value is BooleanValue.AlwaysFalse:
                    result = generate_uniform_cst(False, node.get_precision())
            elif isinstance(node, Test):
                test_value = is_simplifiable_test(node, node.inputs)
                if test_value is BooleanValue.AlwaysTrue:
                    result = generate_uniform_cst(True, node.get_precision())
                elif test_value is BooleanValue.AlwaysFalse:
                    result = generate_uniform_cst(False, node.get_precision())
            elif isinstance(node, ConditionBlock):
                result = simplify_condition_block(node)
            elif isinstance(node, LogicOperation):
                result = simplify_logical_op(node)
            if not result is None:
                Log.report(LOG_VERBOSE_NUMERICAL_SIMPLIFICATION, "{} has been simplified to {}", node, result)
            self.memoization_map[node] = result
            return result





class Pass_NumericalSimplification(FunctionPass):
    """ Pass to expand oepration on vector masks """
    pass_tag = "numerical_simplification"

    def __init__(self, target):
        FunctionPass.__init__(
            self, "virtual vector bool legalization pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.simplifier = NumericalSimplifier(target)

    def execute_on_optree(self, node, fct=None, fct_group=None, memoization_map=None):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.simplifier.simplify(node)



Log.report(LOG_PASS_INFO, "Registering numerical_simplification pass")
# register pass
Pass.register(Pass_NumericalSimplification)
