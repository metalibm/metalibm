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

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.opt.node_transformation import Pass_NodeTransformation
from metalibm_core.opt.opt_utils import (
    forward_attributes, logical_and_reduce, logical_or_reduce
)

from metalibm_core.core.ml_formats import (
    ML_Bool, ML_Integer,
    VECTOR_TYPE_MAP,
)

from metalibm_core.core.ml_operations import (
    Equal, NotEqual, Constant,
    VectorElementSelection,
    Comparison, BooleanOperation,
    LogicalNot, LogicalOr, LogicalAnd,
    Test,
)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi

LOG_VERBOSE_VECTOR_MASK_TEST = Log.LogLevel("VectorMaskTestVerbose")

# List of tests which can be expanded by VectorMaskTestExpander
EXPANDABLE_TEST_LIST = [
    Test.IsMaskAllZero, Test.IsMaskNotAllZero,
    Test.IsMaskNotAnyZero, Test.IsMaskAnyZero
]

def is_vector_mask_test(node):
    return isinstance(node, Test) and node.specifier in EXPANDABLE_TEST_LIST

class VectorMaskTestExpander(object):
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
        elif not is_vector_mask_test(node):
            result = None
        elif isinstance(node, Test):
            vector_input = node.get_input(0)
            scalar_format = vector_input.get_precision().get_scalar_format()

            def index_node(index_value):
                return Constant(index_value, precision=ML_Integer)

            def check_true(element):
                if element.get_precision() is ML_Bool:
                    return element
                else:
                    return NotEqual(element, Constant(0, precision=element.get_precision()))

            def check_false(element):
                if element.get_precision() is ML_Bool:
                    return LogicalNot(element, precision=ML_Bool)
                else:
                    return Equal(element, Constant(0, precision=element.get_precision()))

            if node.specifier is Test.IsMaskAllZero:
                result = logical_and_reduce(
                    [
                        check_false(
                            VectorElementSelection(vector_input, index_node(index), precision=scalar_format),
                        ) for index in range(vector_input.get_precision().get_vector_size())
                    ])
            elif node.specifier is Test.IsMaskAnyZero:
                result = logical_or_reduce(
                    [
                        check_false(
                            VectorElementSelection(vector_input, index_node(index), precision=scalar_format),
                        ) for index in range(vector_input.get_precision().get_vector_size())
                    ])
            elif node.specifier is Test.IsMaskNotAllZero:
                result = logical_or_reduce(
                    [
                        check_true(
                            VectorElementSelection(vector_input, index_node(index), precision=scalar_format),
                        ) for index in range(vector_input.get_precision().get_vector_size())
                    ])
            elif node.specifier is Test.IsMaskNotAnyZero:
                result = logical_and_reduce(
                    [
                        check_true(
                            VectorElementSelection(vector_input, index_node(index), precision=scalar_format),
                        ) for index in range(vector_input.get_precision().get_vector_size())
                    ])
            else:
                raise NotImplementedError
            Log.report(LOG_VERBOSE_VECTOR_MASK_TEST, "expand vector mask test from {} to {}", node, result)
        else:
            result = None

        self.memoization_map[node] = result
        return result


    def is_expandable(self, node):
        """ return True if @p node is expandable else False """
        return is_vector_mask_test(node)


class Pass_VectorMaskTestLegalization(Pass_NodeTransformation):
    """ Pass to expand oepration on vector masks """
    pass_tag = "vector_mask_test_legalization"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "virtual vector bool legalization pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.expander = VectorMaskTestExpander(target)

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


Log.report(LOG_PASS_INFO, "Registering vector_mask_test_legalization pass")
# register pass
Pass.register(Pass_VectorMaskTestLegalization)
