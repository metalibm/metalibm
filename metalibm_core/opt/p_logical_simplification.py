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


from metalibm_core.core.ml_operations import (
    is_leaf_node,
    LogicalAnd, LogicalOr, LogicalNot,
)
from metalibm_core.core.legalizer import is_constant

from metalibm_core.utility.log_report import Log

LOG_LOGICAL_SIMPLIFICATION = Log.LogLevel("LogicalSimplification")

from metalibm_core.opt.opt_utils import is_false, is_true





class LogicalSimplification:
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def is_simplifiable(self, node):
        if isinstance(node, (LogicalAnd, LogicalOr)):
            # And(Not(lhs), Not(rhs)) => Not(Or(lhs, rhs)) : 3 ops -> 2 ops
            # Or(Not(lhs), Not(rhs)) => Not(And(lhs, rhs)) : 3 ops -> 2 ops
            lhs = node.get_input(0)
            rhs = node.get_input(1)
            return isinstance(lhs, LogicalNot) and isinstance(rhs, LogicalNot)
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
            if self.is_simplifiable(node):
                if isinstance(node, (LogicalAnd, LogicalOr)):
                    # And(Not(lhs), Not(rhs)) => Not(Or(lhs, rhs)) : 3 ops -> 2 ops
                    # Or(Not(lhs), Not(rhs)) => Not(And(lhs, rhs)) : 3 ops -> 2 ops
                    lhs = node.get_input(0)
                    rhs = node.get_input(1)
                    assert isinstance(lhs, LogicalNot)
                    assert isinstance(rhs, LogicalNot)
                    lhs = self.simplify(lhs.get_input(0))
                    rhs = self.simplify(rhs.get_input(0))
                    ctor = LogicalAnd if isinstance(node, LogicalOr) else LogicalOr
                    result = LogicalNot(
                        ctor(lhs, rhs, precision=node.get_precision()),
                        precision=node.get_precision()
                    )
                    forward_attributes(result, node)
            elif not is_leaf_node(node):
                for index, op in enumerate(node.inputs):
                    new_op = self.simplify(op)
                    if new_op != op:
                        node.set_input(index, new_op)
            if not result is None:
                Log.report(LOG_LOGICAL_SIMPLIFICATION, "{} has been simplified to {}", node, result)
            else:
                # no simplification
                result = node
            self.memoization_map[node] = result
            return result





class Pass_LogicalSimplification(FunctionPass):
    """ Pass to simplify logical operation """
    pass_tag = "logical_simplification"

    def __init__(self, target):
        FunctionPass.__init__(
            self, "logical operation simplification pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.simplifier = LogicalSimplification(target)

    def execute_on_optree(self, node, fct=None, fct_group=None, memoization_map=None):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.simplifier.simplify(node)



Log.report(LOG_PASS_INFO, "Registering logical_simplification pass")
# register pass
Pass.register(Pass_LogicalSimplification)
