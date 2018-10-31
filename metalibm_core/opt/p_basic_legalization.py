
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

import sollya

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.opt.node_transformation import Pass_NodeTransformation
from metalibm_core.opt.opt_utils import forward_attributes

from metalibm_core.core.ml_formats import (
    ML_FP_MultiElementFormat,
    ML_Binary32, ML_Binary64,
    ML_SingleSingle,
    ML_DoubleDouble, ML_TripleDouble
)

from metalibm_core.core.ml_operations import (
    ExponentExtraction,
    ExponentInsertion,
    Test,
    is_leaf_node,
)
from metalibm_core.core.legalizer import (
    generate_test_expansion, generate_exp_extraction,
    generate_exp_insertion
)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi


# List of tests which can be expanded by BasicExpander
EXPANDABLE_TEST_LIST = [
    Test.IsInfOrNaN, Test.IsInfty, Test.IsNaN, Test.IsZero, Test.IsSubnormal,
    Test.IsQuietNaN, Test.IsSignalingNaN
]

def is_expandable_test(node):
    return isinstance(node, Test) and \
            node.specifier in EXPANDABLE_TEST_LIST

class BasicExpander(object):
    """ Basic expansion engine """
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def expand_node(self, node):
        """ return the expansion of @p node when possible
            else None """
        def wrap_expansion(node, transformed_node):
            return node if transformed_node is None else transformed_node

        if node in self.memoization_map:
            return self.memoization_map[node]
        elif isinstance(node, ExponentExtraction):
            op = wrap_expansion(node.get_input(0), self.expand_node(node.get_input(0)))
            result = generate_exp_extraction(op)
            forward_attributes(node, result)
        elif isinstance(node, ExponentInsertion):
            op = wrap_expansion(node.get_input(0), self.expand_node(node.get_input(0)))
            result = generate_exp_insertion(op, node.precision)
            forward_attributes(node, result)
        elif is_expandable_test(node):
            op = wrap_expansion(node.get_input(0), self.expand_node(node.get_input(0)))
            result = generate_test_expansion(node.specifier, op)
            forward_attributes(node, result)
        else:
            result = None

        self.memoization_map[node] = result
        return result


    def is_expandable(self, node):
        """ return True if @p node is expandable else False """
        return isinstance(node, ExponentExtraction) or \
               isinstance(node, ExponentInsertion) or \
               is_expandable_test(node)


class Pass_BasicLegalization(Pass_NodeTransformation):
    """ Basic legalization pass """
    pass_tag = "basic_legalization"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "basic legalization pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.expander = BasicExpander(target)

    def can_be_transformed(self, node, *args):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        return self.expander.is_expandable(node)

    def transform_node(self, node, transformed_inputs, *args):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.expander.expand_node(node)

    def reconstruct_from_transformed(self, node, transformed_node):
        """return a node at the root of a transformation chain,
            compatible with untransformed nodes """
        return transformed_node

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)


Log.report(LOG_PASS_INFO, "Registering basic_legalization pass")
# register pass
Pass.register(Pass_BasicLegalization)
