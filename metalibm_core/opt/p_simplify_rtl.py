
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
# Description: optimization pass to expand multi-precision node to simple
#              precision implementation
###############################################################################

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import (
    VectorElementSelection
)
from metalibm_core.core.ml_hdl_operations import (
    BitSelection, SubSignalSelection
)
from metalibm_core.core.legalizer import evaluate_cst_graph

from metalibm_core.opt.node_transformation import Pass_NodeTransformation
from metalibm_core.opt.opt_utils import forward_attributes

from metalibm_core.utility.log_report import Log

LOG_VERBOSE_SIMPLIFY_RTL = Log.LogLevel("SimplifyRTLVerbose")

def is_simplifiable(node):
    return (isinstance(node, VectorElementSelection) and isinstance(node.get_input(0), SubSignalSelection)) or \
        (isinstance(node, SubSignalSelection) and isinstance(node.get_input(0), SubSignalSelection))

class BasicSimplifier(object):
    """ Basic expansion engine """
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def simplify_node(self, node):
        """ return the simplified version of @p node when possible
            else the node itself """
        result = node

        if node in self.memoization_map:
            return self.memoization_map[node]
        elif isinstance(node, SubSignalSelection):
            op_input = self.simplify_node(node.get_input(0))
            lo_index = evaluate_cst_graph(node.get_input(1))
            hi_index = evaluate_cst_graph(node.get_input(2))
            if isinstance(op_input, SubSignalSelection):
                parent_input = self.simplify_node(op_input.get_input(0))
                parent_lo_index = evaluate_cst_graph(op_input.get_input(1))
                new_node = SubSignalSelection(parent_input, parent_lo_index + lo_index, parent_lo_index + hi_index)
                forward_attributes(node, new_node)
                result = new_node

        elif isinstance(node, VectorElementSelection):
            op_input = self.simplify_node(node.get_input(0))
            op_index = evaluate_cst_graph(node.get_input(1))
            if isinstance(op_input, SubSignalSelection):
                parent_input = op_input.get_input(0)
                base_index = evaluate_cst_graph(op_input.get_input(1))
                new_node = BitSelection(parent_input, base_index + op_index)
                forward_attributes(node, new_node)
                result = new_node
        else:
            result = node

        Log.report(LOG_VERBOSE_SIMPLIFY_RTL, "Simplifying {} to {}", node, result)

        self.memoization_map[node] = result
        return result

    def is_simplifiable(self, node):
        return is_simplifiable(node)


class Pass_SimplifyRTL(Pass_NodeTransformation):
    """ RTL Simplification pass """
    pass_tag = "simplify_rtl"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "rtl simplification pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.simplifier = BasicSimplifier(target)

    def can_be_transformed(self, node, *args):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        return self.simplifier.is_simplifiable(node)

    def transform_node(self, node, transformed_inputs, *args):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.simplifier.simplify_node(node)

    def reconstruct_from_transformed(self, node, transformed_node):
        """return a node at the root of a transformation chain,
            compatible with untransformed nodes """
        return transformed_node

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)


Log.report(LOG_PASS_INFO, "Registering simplify_rtl pass")
# register pass
Pass.register(Pass_SimplifyRTL)
