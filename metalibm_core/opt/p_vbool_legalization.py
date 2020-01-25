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
from metalibm_core.opt.opt_utils import forward_attributes

from metalibm_core.core.ml_formats import (
    ML_Bool,
    VECTOR_TYPE_MAP,
)

from metalibm_core.core.ml_operations import (
    Comparison, BooleanOperation,
    LogicalNot, LogicalOr, LogicalAnd,
    Test,
)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi

LOG_VERBOSE_VBOOL_LEGALIZATION = Log.LogLevel("VirtualBoolLegalizeVerbose")

# List of tests which can be expanded by VirtualBoolVectorLegalizer
EXPANDABLE_TEST_LIST = [
    Test.IsInfOrNaN, Test.IsInfty, Test.IsNaN, Test.IsZero, Test.IsSubnormal,
    Test.IsQuietNaN, Test.IsSignalingNaN
]

def is_virtual_bool_node(node):
    return isinstance(node, BooleanOperation) and \
        node.get_precision().is_vector_format() and \
        node.get_precision().get_scalar_format() is ML_Bool

class VirtualBoolVectorLegalizer(object):
    """ Basic expansion engine """
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def legalize_node(self, node):
        """ return the expansion of @p node when possible
            else None """
        def wrap_expansion(node, transformed_node):
            return node if transformed_node is None else transformed_node

        if node in self.memoization_map:
            return self.memoization_map[node]
        elif not is_virtual_bool_node(node):
            result = None
        elif isinstance(node, Comparison) or isinstance(node, LogicalAnd) or isinstance(node, LogicalOr):
            scalar_bit_size = max(
                node.get_input(0).get_precision().get_scalar_format().get_bit_size(),
                node.get_input(1).get_precision().get_scalar_format().get_bit_size()
            )
            legalized_format = VECTOR_TYPE_MAP[ML_Bool][scalar_bit_size][node.get_precision().get_vector_size()]
            Log.report(LOG_VERBOSE_VBOOL_LEGALIZATION, "legalizing format of {} from {} to {}", node, node.get_precision(), legalized_format)
            node.set_attributes(precision=legalized_format)
            result = node
        elif isinstance(node, LogicalNot) or (isinstance(node, Test) and node.specifier in EXPANDABLE_TEST_LIST):
            scalar_bit_size = node.get_input(0).get_precision().get_scalar_format().get_bit_size()
            legalized_format = VECTOR_TYPE_MAP[ML_Bool][scalar_bit_size][node.get_precision().get_vector_size()]
            Log.report(LOG_VERBOSE_VBOOL_LEGALIZATION, "legalizing format of {} from {} to {}", node, node.get_precision(), legalized_format)
            node.set_attributes(precision=legalized_format)
            result = node
        else:
            result = None

        self.memoization_map[node] = result
        return result


    def is_legalizable(self, node):
        """ return True if @p node is expandable else False """
        return isinstance(node, Comparison) or \
               is_virtual_bool_node(node)


class Pass_VirtualVectorBoolLegalization(Pass_NodeTransformation):
    """ Pass to legalize vector of virtual bools """
    pass_tag = "virtual_vector_bool_legalization"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "virtual vector bool legalization pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.expander = VirtualBoolVectorLegalizer(target)

    def can_be_transformed(self, node, *args):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        return self.expander.is_legalizable(node)

    def transform_node(self, node, transformed_inputs, *args):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.expander.legalize_node(node)

    def reconstruct_from_transformed(self, node, transformed_node):
        """return a node at the root of a transformation chain,
            compatible with untransformed nodes """
        return transformed_node

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)


Log.report(LOG_PASS_INFO, "Registering virtual_vector_bool_legalization pass")
# register pass
Pass.register(Pass_VirtualVectorBoolLegalization)
