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

from metalibm_core.core.ml_formats import (
    ML_FP_MultiElementFormat,
    ML_Binary32, ML_Binary64,
    ML_SingleSingle,
    ML_DoubleDouble, ML_TripleDouble
)

from metalibm_core.core.ml_operations import (
    Addition, Constant, Multiplication, Variable, 
    BuildFromComponent
)
from metalibm_core.opt.ml_blocks import (
    Add222, Add122, Add221, Add212, 
    Add121, Add112, 
    Add211, Mul212, Mul211, Mul222
)

from metalibm_core.utility.log_report import Log


class MultiPrecisionExpander:
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}

    def expand_cst(self, cst_node):
        """ Expand a Constant node in multi-precision format into a list
            of Constants node in scalar format and returns the list """
        cst_multiformat = cst_node.precision
        cst_value = cst_node.get_value()
        cst_list = []
        for elt_format in cst_multiformat.field_format_list:
            cst_sub_value = elt_format.round_sollya_object(cst_value)
            cst_list.append(Constant(cst_sub_value, precision=elt_format))
            # updating cst_value
            cst_value -= cst_sub_value
        return tuple(cst_list)

    def expand_var(self, var_node):
        """ Expand a variable in multi-precision format into
            a list of ComponentSelection nodes """
        var_multiformat = var_node.precision
        if len(var_multiformat.field_format_list) == 2:
            return (var_node.hi, var_node.lo)
        elif len(var_multiformat.field_format_list) == 3:
            return (var_node.hi, var_node.me, var_node.lo)
        else:
            return tuple([
                ComponentSelection(
                    var_node,
                    precision=elt_format,
                    specifier=ComponentSelection.Field(index)
                ) for index, elt_format in enumerate(var_multiformat.field_format_list)
            ])

    def tag_expansion(self, node, expansion):
        """ set tags to element of list @p expansion
            which were dervied from @p node """
        suffix_list = {
            1: ["_hi"],
            2: ["_hi", "_lo"],
            3: ["_hi", "_me", "_lo"]
        }
        expansion_len = len(expansion)
        node_tag = node.get_tag()
        if expansion_len in suffix_list:
            for elt, suffix in zip(expansion, suffix_list[expansion_len]):
                elt.set_tag(node_tag + suffix)
        else:
            for index, elt in enumerate(expansion):
                elt.set_tag("{}_s{}".format(node_tag, expansion_len - 1 - index))

    def expand_binary_op(self, node, expander_map):
        """ Generic expansion method for 2-operand node """
        lhs = node.get_input(0)
        rhs = node.get_input(1)

        lhs_list = self.expand_node(lhs)
        rhs_list = self.expand_node(rhs)

        lhs_precision = lhs.precision
        rhs_precision = rhs.precision
        result_precision = node.precision

        def get_elementary_precision(multi_precision):
            """ return the elementary precision corresponding
                to multi_precision """
            if isinstance(multi_precision, ML_FP_MultiElementFormat):
                return multi_precision.field_format_list[0]
            else:
                return mutli_precision

        elt_precision = get_elementary_precision(result_precision)

        try:
            expander = expander_map[(result_precision, (lhs_precision, rhs_precision))]
        except KeyError:
            Log.report(
                Log.Error,
                "unable to find expander for ({}, {}, {})",
                result_precision, lhs_precision, rhs_precision)
        new_op = expander(*(lhs_list + rhs_list), precision=elt_precision)
        # setting dedicated name to expanded node
        self.tag_expansion(node, new_op)
        return new_op


    def expand_add(self, add_node):
        """ Expand Addition """
        ADD_EXPANSION_MAP = {
            # double precision based formats
            (ML_DoubleDouble, (ML_Binary64, ML_Binary64)): Add211,
            (ML_DoubleDouble, (ML_DoubleDouble, ML_Binary64)): Add221,
            (ML_DoubleDouble, (ML_Binary64, ML_DoubleDouble)): Add212,
            (ML_DoubleDouble, (ML_DoubleDouble, ML_DoubleDouble)): Add222,
            (ML_Binary64, (ML_DoubleDouble, ML_Binary64)): Add121,
            (ML_Binary64, (ML_Binary64, ML_DoubleDouble)): Add112,
            # single precision based formats
            (ML_SingleSingle, (ML_Binary32, ML_Binary32)): Add211,
            (ML_SingleSingle, (ML_SingleSingle, ML_Binary32)): Add221,
            (ML_SingleSingle, (ML_Binary32, ML_SingleSingle)): Add212,
            (ML_SingleSingle, (ML_SingleSingle, ML_SingleSingle)): Add222,
            (ML_Binary32, (ML_SingleSingle, ML_Binary32)): Add121,
            (ML_Binary32, (ML_Binary32, ML_SingleSingle)): Add112,
        }
        return self.expand_binary_op(add_node, ADD_EXPANSION_MAP)


    def is_expandable(self, node):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        return isinstance(node.precision, ML_FP_MultiElementFormat) and \
            (isinstance(node, Addition) or \
             isinstance(node, Multiplication) or \
             isinstance(node, Subtraction) or \
             isinstance(node, FusedMultiplyAdd))


    def expand_conversion(self, node):
        # optimizing Conversion
        op_input = self.expand_node(node.get_input())
        # checking if you are looking at the Conversion of a
        # BuildFromComponent node which can be simplified directly
        # to a component
        # TODO: does not manage different component format in
        #       field_format_list
        if isinstance(op_input, BuildFromComponent) and isinstance(op_input.precision, ML_FP_MultiElementFormat) and op_input.precision.field_format_list[0] == node.precision:
            return op_input.get_input(0)
        return node

    def expand_sub(self, node):
        raise NotImplementedError
    def expand_mul(self, node):
        raise NotImplementedError
    def expand_fma(self, node):
        raise NotImplementedError

    def expand_node(self, node):
        """ If node @p node is a multi-precision node, expands to a list
            of scalar element, ordered from most to least significant """

        if node in self.memoization_map:
            return self.memoization_map[node]
        else:
            if not isinstance(node.precision, ML_FP_MultiElementFormat):
                return (node,)
            if isinstance(node, Variable):
                return self.expand_var(node)
            elif isinstance(node, Constant):
                return self.expand_cst(node)
            elif isinstance(node, Addition):
                return self.expand_add(node)
            elif isinstance(node, Multiplication):
                return self.expand_mul(node)
            elif isinstance(node, Subtraction):
                return self.expand_sub(node)
            elif isinstance(node, FusedMultiplyAdd):
                return self.expand_fma(node)
            elif isinstance(node, Conversion):
                return self.expand_conversion(node)
            else:
                # no modification
                return None

            self.memoization_map[node] = elt_list


def dirty_multi_node_expand(node, precision, mem_map=None, fma=True):
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
        op1h, op1l = dirty_multi_node_expand(lhs, precision, mem_map, fma)
        op2h, op2l = dirty_multi_node_expand(rhs, precision, mem_map, fma)
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
            result = Mul222(op1h, op1l, op2h, op2l, fma=fma) \
                    if op1l is not None and op2l is not None \
                    else Mul212(op1h, op2h, op2l, fma=fma) \
                    if op1l is None and op2l is not None \
                    else Mul212(op2h, op1h, op1l, fma=fma) \
                    if op2l is None and op1l is not None \
                    else Mul211(op1h, op2h, fma=fma)
            mem_map[node] = result
            return result


class Pass_ExpandMultiPrecision(Pass_NodeTransformation):
    """ Generic Multi-Precision expansion pass """
    pass_tag = "expand_multi_precision"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "multi-precision expansion pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.expander = MultiPrecisionExpander(target)

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
        if isinstance(node, Constant):
            return node
        else:
            return BuildFromComponent(*tuple(transformed_node), precision=node.precision)

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)


Log.report(LOG_PASS_INFO, "Registering expand_multi_precision pass")
# register pass
Pass.register(Pass_ExpandMultiPrecision)
