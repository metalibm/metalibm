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
    Addition, Subtraction, Multiplication,
    FusedMultiplyAdd,
    Conversion, Negation,
    Constant, Variable, SpecificOperation,
    BuildFromComponent, ComponentSelection,
    is_leaf_node,
)
from metalibm_core.opt.ml_blocks import (
    Add222, Add122, Add221, Add212,
    Add121, Add112, Add122,
    Add211,
    Mul212, Mul221, Mul211, Mul222,
    Mul122, Mul121, Mul112,
    MP_FMA2111, MP_FMA2112, MP_FMA2122, MP_FMA2212, MP_FMA2121, MP_FMA2211,
    MP_FMA2222,

    MP_Add323, MP_Add332, MP_Add333,
    MP_Mul322, MP_Mul332, MP_Mul323,

    subnormalize_multi,
    Normalize_33,
)

from metalibm_core.utility.log_report import Log

# high verbosity log-level for expand_multi_precision pass module
LOG_LEVEL_EXPAND_VERBOSE = Log.LogLevel("ExpandVerbose")

def is_subnormalize_op(node):
    """ test if @p node is a Subnormalize operation """
    return isinstance(node, SpecificOperation) and node.specifier is SpecificOperation.Subnormalize


def get_elementary_precision(multi_precision):
    """ return the elementary precision corresponding
        to multi_precision """
    multi_precision = multi_precision.get_match_format()
    if isinstance(multi_precision, ML_FP_MultiElementFormat):
        return multi_precision.field_format_list[0]
    else:
        return multi_precision


def is_multi_precision_format(precision):
    """ check if precision is a multi-element FP format """
    if precision is None:
        return False
    return isinstance(precision.get_match_format(), ML_FP_MultiElementFormat)
def multi_element_output(node):
    """ return True if node's output format is a multi-precision type """
    return is_multi_precision_format(node.precision)
def multi_element_inputs(node):
    """ return True if any of node's input has a multi-precision type """
    return not is_leaf_node(node) and any(is_multi_precision_format(op_input.precision) for op_input in node.get_inputs())

def has_component_selection_input(node):
    """ Check if any of node's input is a ComponentSelection node """
    return not is_leaf_node(node) and any(isinstance(op, ComponentSelection) for op in node.get_inputs())

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
        var_multiformat = var_node.precision.get_match_format()
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
        tag_prefix = "" if node_tag is None else node_tag
        if expansion_len in suffix_list:
            for elt, suffix in zip(expansion, suffix_list[expansion_len]):
                elt.set_tag(tag_prefix + suffix)
        else:
            for index, elt in enumerate(expansion):
                elt.set_tag("{}_s{}".format(node_tag, expansion_len - 1 - index))

    def expand_op(self, node, expander_map, arity=2):
        """ Generic expansion method for 2-operand node """
        operands = [node.get_input(i) for i in range(arity)]

        def wrap_expand(op):
            """ expand node and returns it if no modification occurs """
            expanded_node = self.expand_node(op)
            return (op,) if expanded_node is None else expanded_node

        operands_expansion = [list(wrap_expand(op)) for op in operands]
        operands_format = [op.precision.get_match_format() for op in operands]

        result_precision = node.precision.get_match_format()

        elt_precision = get_elementary_precision(result_precision)

        try:
            expansion_key = (result_precision, tuple(operands_format))
            expander = expander_map[expansion_key]
        except KeyError:
            Log.report(
                Log.Error,
                "unable to find multi-precision expander for {}, key is {}",
                node, str(expansion_key))
        new_op = expander(*(sum(operands_expansion, [])), precision=elt_precision)
        # setting dedicated name to expanded node
        self.tag_expansion(node, new_op)
        # forward other attributes
        for elt in new_op:
           elt.set_debug(node.get_debug())
           elt.set_handle(node.get_handle())
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
            (ML_Binary64, (ML_DoubleDouble, ML_DoubleDouble)): Add122,
            (ML_TripleDouble, (ML_TripleDouble, ML_TripleDouble)): MP_Add333,
            (ML_TripleDouble, (ML_TripleDouble, ML_DoubleDouble)): MP_Add332,
            (ML_TripleDouble, (ML_DoubleDouble, ML_TripleDouble)): MP_Add323,
            # single precision based formats
            (ML_SingleSingle, (ML_Binary32, ML_Binary32)): Add211,
            (ML_SingleSingle, (ML_SingleSingle, ML_Binary32)): Add221,
            (ML_SingleSingle, (ML_Binary32, ML_SingleSingle)): Add212,
            (ML_SingleSingle, (ML_SingleSingle, ML_SingleSingle)): Add222,
            (ML_Binary32, (ML_SingleSingle, ML_Binary32)): Add121,
            (ML_Binary32, (ML_Binary32, ML_SingleSingle)): Add112,
            (ML_Binary32, (ML_SingleSingle, ML_SingleSingle)): Add122,
        }
        return self.expand_op(add_node, ADD_EXPANSION_MAP, arity=2)

    def expand_mul(self, mul_node):
        """ Expand Multiplication """
        MUL_EXPANSION_MAP = {
            # double precision based formats
            (ML_DoubleDouble, (ML_DoubleDouble, ML_DoubleDouble)): Mul222,
            (ML_DoubleDouble, (ML_Binary64, ML_DoubleDouble)): Mul212,
            (ML_DoubleDouble, (ML_DoubleDouble, ML_Binary64)): Mul221,
            (ML_DoubleDouble, (ML_Binary64, ML_Binary64)): Mul211,
            (ML_Binary64, (ML_DoubleDouble, ML_DoubleDouble)): Mul122,
            (ML_Binary64, (ML_DoubleDouble, ML_Binary64)): Mul121,
            (ML_Binary64, (ML_Binary64, ML_DoubleDouble)): Mul112,
            (ML_TripleDouble, (ML_DoubleDouble, ML_DoubleDouble)): MP_Mul322,
            (ML_TripleDouble, (ML_TripleDouble, ML_DoubleDouble)): MP_Mul332,
            (ML_TripleDouble, (ML_DoubleDouble, ML_TripleDouble)): MP_Mul323,
            # single precision based formats
            (ML_SingleSingle, (ML_SingleSingle, ML_SingleSingle)): Mul222,
            (ML_SingleSingle, (ML_Binary32, ML_SingleSingle)): Mul212,
            (ML_SingleSingle, (ML_SingleSingle, ML_Binary32)): Mul221,
            (ML_SingleSingle, (ML_Binary32, ML_Binary32)): Mul211,
            (ML_Binary32, (ML_SingleSingle, ML_SingleSingle)): Mul122,
            (ML_Binary32, (ML_SingleSingle, ML_Binary32)): Mul121,
            (ML_Binary32, (ML_Binary32, ML_SingleSingle)): Mul112,
        }
        return self.expand_op(mul_node, MUL_EXPANSION_MAP, arity=2)

    def expand_fma(self, fma_node):
        """ Expand Fused-Multiply Add """
        FMA_EXPANSION_MAP = {
            # double precision based formats
            (ML_DoubleDouble, (ML_DoubleDouble, ML_DoubleDouble, ML_DoubleDouble)): MP_FMA2222,
            (ML_DoubleDouble, (ML_DoubleDouble, ML_Binary64, ML_DoubleDouble)): MP_FMA2212,
            (ML_DoubleDouble, (ML_Binary64, ML_DoubleDouble, ML_DoubleDouble)): MP_FMA2122,
            (ML_DoubleDouble, (ML_Binary64, ML_Binary64, ML_DoubleDouble)): MP_FMA2112,
            (ML_DoubleDouble, (ML_Binary64, ML_DoubleDouble, ML_Binary64)): MP_FMA2121,
            (ML_DoubleDouble, (ML_DoubleDouble, ML_Binary64, ML_Binary64)): MP_FMA2211,
            (ML_DoubleDouble, (ML_Binary64, ML_Binary64, ML_Binary64)): MP_FMA2111,
            # single precision based formats
            (ML_SingleSingle, (ML_SingleSingle, ML_SingleSingle, ML_SingleSingle)): MP_FMA2222,
            (ML_SingleSingle, (ML_SingleSingle, ML_Binary32, ML_SingleSingle)): MP_FMA2212,
            (ML_SingleSingle, (ML_Binary32, ML_SingleSingle, ML_SingleSingle)): MP_FMA2122,
            (ML_SingleSingle, (ML_Binary32, ML_Binary32, ML_SingleSingle)): MP_FMA2112,
            (ML_SingleSingle, (ML_Binary32, ML_SingleSingle, ML_Binary32)): MP_FMA2121,
            (ML_SingleSingle, (ML_SingleSingle, ML_Binary32, ML_Binary32)): MP_FMA2211,
            (ML_SingleSingle, (ML_Binary32, ML_Binary32, ML_Binary32)): MP_FMA2111,
        }
        return self.expand_op(fma_node, FMA_EXPANSION_MAP, arity=3)

    def expand_subnormalize(self, sub_node):
        """ Expand SpecificOperation.Subnormalize on multi-component node """
        operand = sub_node.get_input(0)
        factor = sub_node.get_input(1)
        exp_operand = self.expand_node(operand)
        elt_precision = get_elementary_precision(sub_node.precision)
        return subnormalize_multi(exp_operand, factor, precision=elt_precision)

    def expand_negation(self, neg_node):
        """ Expand Negation on multi-component node """
        op_input = neg_node.get_input(0)
        neg_operands = self.expand_node(op_input)
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding Negation {} into {}", neg_node, neg_operands)
        return [Negation(op, precision=op.precision) for op in neg_operands]

    def is_expandable(self, node):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        expandable = ((multi_element_output(node) or multi_element_inputs(node))) and \
            (isinstance(node, Addition) or \
             isinstance(node, Multiplication) or \
             isinstance(node, Subtraction) or \
             isinstance(node, Conversion) or \
             isinstance(node, FusedMultiplyAdd) or \
             isinstance(node, Negation) or \
             isinstance(node, BuildFromComponent) or \
             isinstance(node, ComponentSelection) or \
             is_subnormalize_op(node))
        if not expandable:
            Log.report(LOG_LEVEL_EXPAND_VERBOSE, "{} cannot be expanded", node)
        return expandable


    def expand_conversion(self, node):
        """ Expand Conversion node """
        # optimizing Conversion
        op_input = self.expand_node(node.get_input(0))
        if not op_input is None:
            if op_input[0].precision == node.precision:
                # if the conversion is from a multi-precision node to a result whose
                # precision matches the multi-precision high component, then directly
                # returns it
                # TODO/FIXME: does not take into account possible overlap between
                #             limbs
                Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding conversion {} into {}", node, op_input[0])
                return [op_input[0]]
            elif is_multi_precision_format(node.precision) and \
                 node.precision.limb_num >= len(op_input) and \
                 all(op.precision == limb_prec for op, limb_prec in zip(op_input, node.precision.field_format_list)):
                # if the conversion is from a multi-element format to a larger multi-element format
                # just pad the input with 0
                pad_size = node.precision.limb_num - len(op_input)
                return op_input + tuple(Constant(0, precision=node.precision.get_limb_precision(len(op_input) + i)) for i in range(pad_size))
            elif is_multi_precision_format(node.precision) and \
                node.precision.limb_num < len(op_input) and \
                all(op.precision == limb_prec for op, limb_prec in zip(op_input, node.precision.field_format_list)):
                # if the conversion if from a multi-element format to a smaller
                # multi-element format than insert a normalization and
                # return the appropriate limb
                assert node.precision.limb_num == 2 and len(op_input) == 3
                normalized_op = Normalize_33(*op_input, precision=op_input[0].precision)
                Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding conversion {} into {}, {}", node, normalized_op[0], normalized_op[1])
                return normalized_op[0], normalized_op[1]


        return None

    def expand_sub(self, node):
        lhs = node.get_input(0)
        rhs = node.get_input(1)
        tag = node.get_tag()
        precision = node.get_precision()
        # Subtraction x - y is transformed into x + (-y)
        # WARNING: if y is not expandable (e.g. scalar precision)
        #          this could stop expansion
        new_node = Addition(
            lhs,
            Negation(
                rhs,
                precision=rhs.precision
            ),
            precision=precision
        )
        forward_attributes(node, new_node)
        expanded_node =  self.expand_node(new_node)
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding Subtraction {} into {} with expanded form {}", node, new_node, ", ".join((op.get_str(display_precision=True, depth=None)) for op in expanded_node))
        return expanded_node

    def expand_build_from_component(self, node):
        op_list = ((self.expand_node(op), op) for op in node.get_inputs())
        result = tuple(op if expanded is None else expanded for (op, expanded) in op_list)
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding BuildFromComponent {} into {}", node, result)
        return result

    def expand_component_selection(self, node):
        # TODO: manage TD normalization properly
        if is_leaf_node(node.get_input(0)):
            # discard expansion of leaf nodes (Variable, ...)
            return None
        op_list = self.expand_node(node.get_input(0))
        OP_INDEX_MAP = {
            ComponentSelection.Hi: 0,
            ComponentSelection.Me: -2,
            ComponentSelection.Lo: -1
        }
        op_index = OP_INDEX_MAP[node.specifier]
        result = op_list[op_index]
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expanding ComponentSelection {} into {}", node, result)
        return (result,)

    def reconstruct_from_transformed(self, node, transformed_node):
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "reconstructed : {}", node)
        Log.report(
            LOG_LEVEL_EXPAND_VERBOSE,
            "from transformed: {}", "\n".join([str(n) for n in transformed_node]))
        if isinstance(node, Constant):
            result = node
        else:
            if len(transformed_node) == 1:
                result = transformed_node[0]
            else:
                result = BuildFromComponent(*tuple(transformed_node), precision=node.precision)
            forward_attributes(node, result)
            result.set_tag(node.get_tag())
        Log.report(LOG_LEVEL_EXPAND_VERBOSE, "  result is : {}", result)
        return result


    def expand_node(self, node):
        """ If node @p node is a multi-precision node, expands to a list
            of scalar element, ordered from most to least significant """

        if node in self.memoization_map:
            return self.memoization_map[node]
        else:
            if not (multi_element_output(node) or multi_element_inputs(node)):
                if not is_leaf_node(node):
                    # recursive processing of node's input
                    for index, op in enumerate(node.get_inputs()):
                        op_input = self.expand_node(op)
                        if not op_input is None:
                            reconstructed_input = self.reconstruct_from_transformed(op, op_input)
                            node.set_input(index, reconstructed_input)
                result = (node,)
            elif isinstance(node, Variable):
                result = self.expand_var(node)
            elif isinstance(node, Constant):
                result = self.expand_cst(node)
            elif isinstance(node, Addition):
                result = self.expand_add(node)
            elif isinstance(node, Multiplication):
                result = self.expand_mul(node)
            elif isinstance(node, Subtraction):
                result = self.expand_sub(node)
            elif isinstance(node, FusedMultiplyAdd):
                result = self.expand_fma(node)
            elif isinstance(node, Conversion):
                result = self.expand_conversion(node)
            elif isinstance(node, Negation):
                result = self.expand_negation(node)
            elif isinstance(node, BuildFromComponent):
                result = self.expand_build_from_component(node)
            elif isinstance(node, ComponentSelection):
                result = self.expand_component_selection(node)
            elif is_subnormalize_op(node):
                result = self.expand_subnormalize(node)
            else:
                if is_leaf_node(node):
                    pass
                else:
                    # recursive processing of node's input
                    for index, op in enumerate(node.get_inputs()):
                        op_input = self.expand_node(op)
                        if not op_input is None:
                            reconstructed_input = self.reconstruct_from_transformed(op, op_input)
                            node.set_input(index, reconstructed_input)
                # no modification
                result = None

            if result is None:
                Log.report(LOG_LEVEL_EXPAND_VERBOSE, "expansion is None for {}", node)
            self.memoization_map[node] = result
            return result


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
    """ Expand node working with Multi-Precision formats into
        expanded operation sub-graph working only with single-word formats """
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
        return self.expander.reconstruct_from_transformed(node, transformed_node)

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)


Log.report(LOG_PASS_INFO, "Registering expand_multi_precision pass")
# register pass
Pass.register(Pass_ExpandMultiPrecision)
