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
""" Optimization pass which finely tune datapath widths in
    RTL entities """

import sollya

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import (
    Pass, LOG_PASS_INFO,
    FunctionPass, METALIBM_PASS_REGISTER,
    LinearizedGraphOptimization,
)

from metalibm_core.core.ml_operations import (
    Select, Conversion, Comparison, Min, Max,
    BitLogicRightShift, BitLogicAnd, Constant,
    BitLogicOr, BitLogicLeftShift, VectorElementSelection,
    TypeCast,
    is_leaf_node,
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_operations import (
    SubSignalSelection, Concatenation
)
from metalibm_core.core.ml_hdl_format import is_fixed_point
from metalibm_core.core.ml_formats import ML_Custom_FixedPoint_Format, ML_Int32

from metalibm_core.opt.p_size_datapath import (
    solve_format_Comparison, FormatSolver
)

from metalibm_core.core.legalizer import (
    subsignalsection_legalizer, fixed_point_position_legalizer,
    minmax_legalizer_wrapper
)

from metalibm_core.opt.opt_utils import forward_attributes

###############################################################################
# PASS DESCRIPTION:
# The pass implemented in this file processes an optree and  legalize every
# supported node
# the output format
###############################################################################

LOG_LEVEL_LEGALIZE = Log.LogLevel("LegalizeVerbose")

def legalize_Select(optree):
    """ legalize Select operation node by converting if and else inputs to
        Select output format if the bit sizes do not match """
    cond = optree.get_input(0)
    op0 = optree.get_input(1)
    op1 = optree.get_input(2)
    precision = optree.get_precision()
    if precision is None:
        Log.report(Log.Error, "None precision for Select:\n{}", optree)
    if op0.get_precision().get_bit_size() != precision.get_bit_size():
        optree.set_input(
            1,
            Conversion(
                op0,
                precision = precision
            )
        )
    if op1.get_precision().get_bit_size() != precision.get_bit_size():
        optree.set_input(
            2,
            Conversion(
                op1,
                precision = optree.get_precision()
            )
        )
    return optree

def legalize_Comparison(optree):
    assert isinstance(optree, Comparison)
    new_format = solve_format_Comparison(optree)
    return optree


def legalize_single_operation(optree, format_solver=None):
    if isinstance(optree, SubSignalSelection):
        new_optree = subsignalsection_legalizer(optree)
        return True, new_optree
    elif isinstance(optree, FixedPointPosition):
        new_optree = fixed_point_position_legalizer(optree)
        return True, new_optree
    elif isinstance(optree, Select):
        new_optree = legalize_Select(optree)
        return True, new_optree
    elif isinstance(optree, Comparison):
        new_optree = legalize_Comparison(optree)
        return True, new_optree
    elif isinstance(optree, Min):
        new_optree = minmax_legalizer_wrapper(Comparison.Less)(optree)
        format_solver(new_optree)
        return True, new_optree
    elif isinstance(optree, Max):
        new_optree = minmax_legalizer_wrapper(Comparison.Greater)(optree)
        format_solver(new_optree)
        return True, new_optree
    return False, optree

def sw_legalize_single_operation(optree, format_solver=None):
    if isinstance(optree, SubSignalSelection):
        pre_optree = subsignalsection_legalizer(optree)
        if not pre_optree is optree:
            _, new_optree = sw_legalize_single_operation(pre_optree)
        else:
            new_optree = sw_legalize_subselection(pre_optree)
        return True, new_optree
    elif isinstance(optree, FixedPointPosition):
        new_optree = fixed_point_position_legalizer(optree)
        return True, new_optree
    elif isinstance(optree, Select):
        new_optree = legalize_Select(optree)
        return True, new_optree
    elif isinstance(optree, Comparison):
        new_optree = legalize_Comparison(optree)
        return True, new_optree
    elif isinstance(optree, Min):
        new_optree = minmax_legalizer_wrapper(Comparison.Less)(optree)
        format_solver(new_optree)
        return True, new_optree
    elif isinstance(optree, Max):
        new_optree = minmax_legalizer_wrapper(Comparison.Greater)(optree)
        format_solver(new_optree)
        return True, new_optree
    elif isinstance(optree, Concatenation):
        new_optree = sw_legalize_concatenation(optree)
        return True, new_optree
    elif isinstance(optree, VectorElementSelection):
        if not optree.get_input(0).get_precision().is_vector_format():
            new_optree = sw_legalize_vector_element_selection(optree)
            return True, new_optree
    return False, optree



## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
class Pass_RTLLegalize(LinearizedGraphOptimization):
    """ Legalization of RTL operations """
    pass_tag = "rtl_legalize"

    def __init__(self, target, tag="rtl legalize"):
        """ pass initialization """
        LinearizedGraphOptimization.__init__(self, tag, target)
        self.format_solver = FormatSolver()

    def legalize_operation(self, optree):
        """ """
        # looking into memoization map
        if optree in self.memoization_map:
            return self.memoization_map[optree]

        # has the npde been modified ?
        arg_changed = False

        if is_leaf_node(optree):
            pass
        else:
            for index, op_input in enumerate(optree.get_inputs()):
                is_modified, new_node = self.memoization_map[op_input]
                if is_modified:
                    optree.set_input(index, new_node)
                    arg_changed = True

        local_changed, new_optree = self.legalize_single_operation(optree, self.format_solver)
        if local_changed:
            forward_attributes(optree, new_optree)
            Log.report(LOG_LEVEL_LEGALIZE, "legalized {} to {}", optree, new_optree)

        self.memoization_map[optree] = local_changed, new_optree
        return (local_changed or arg_changed), new_optree

    def legalize_single_operation(self, node, format_solver):
        return legalize_single_operation(node, format_solver)

    def apply_on_node(self, node):
        return self.legalize_operation(node)

    def is_leaf_node(self, node):
        return is_leaf_node(node)

    def extract_result(self, input_node):
        _, result_node = self.memoization_map[input_node]
        return result_node


def generate_bitfield_extraction(target_format, input_node, lo_index, hi_index):
    shift = lo_index
    mask_size = hi_index - lo_index + 1

    input_format = input_node.get_precision().get_base_format()
    if is_fixed_point(input_format) and is_fixed_point(target_format):
        frac_size = target_format.get_frac_size()
        int_size = input_format.get_bit_size() - frac_size
        cast_format = ML_Custom_FixedPoint_Format(
            int_size, frac_size, signed=False)
    else:
        cast_format = None

    # 1st step: shifting the input node the right amount
    shifted_node = input_node if shift == 0 else BitLogicRightShift(input_node, Constant(shift, precision=ML_Int32), precision=input_format)
    raw_format = ML_Custom_FixedPoint_Format(input_format.get_bit_size(), 0, signed=False)
    # 2nd step: masking the input node
    # TODO/FIXME: check thast mask does not overflow or wrap-around
    masked_node = BitLogicAnd(
        TypeCast(shifted_node, precision=raw_format),
        Constant((2**mask_size - 1), precision=raw_format),
        precision=raw_format
    )

    if not cast_format is None:
        casted_node = TypeCast(masked_node, precision=cast_format)
    else:
        casted_node = masked_node

    converted_node = Conversion(casted_node, precision=target_format)
    return converted_node

    # raw_format = ML_Custom_FixedPoint_Format(target_format.get_bit_size(), 0, signed=False)

    # casted_node = TypeCast(
    #     input_node if conv_format is None else Conversion(
    #         input_node,
    #         precision=conv_format
    #     ),
    #     precision=raw_format
    # )
    # shifted_node = casted_node if shift == 0 else BitLogicRightShift(casted_node, shift, precision=raw_format)
    #     
    # return TypeCast(
    #     BitLogicAnd(
    #         shifted_node,
    #         Constant((2**mask_size - 1), precision=raw_format),
    #         precision=raw_format
    #     ),
    #     precision=target_format
    # )

def sw_legalize_subselection(node):
    """ legalize a RTL SubSignalSelection into
        a sub-graph compatible with software implementation """
    assert isinstance(node, SubSignalSelection)
    op = node.get_input(0)
    lo = node.get_input(1).get_value()
    hi = node.get_input(2).get_value()
    return generate_bitfield_extraction(node.get_precision(), op, lo, hi)

def sw_legalize_concatenation(node):
    """ Legalize a RTL Concatenation node into a sub-graph
        of operation compatible with software implementation """
    assert len(node.inputs) == 2
    lhs = node.get_input(0)
    rhs = node.get_input(1)
    return BitLogicOr(
        BitLogicLeftShift(
            Conversion(lhs, precision=node.get_precision()),
            rhs.get_precision().get_bit_size(), precision=node.get_precision()),
        Conversion(rhs, precision=node.get_precision()),
        precision=node.get_precision()
    )

def sw_legalize_vector_element_selection(node):
    """ Legalize a RTL VectorElementSelection node, if it correspond to a bit selection
        into a sub-graph of operation compatible with software implementation """
    assert len(node.inputs) == 2
    op = node.get_input(0)
    assert not op.get_precision().is_vector_format() 
    assert isinstance(node.get_input(1), Constant)
    index = node.get_input(1).get_value()
    return generate_bitfield_extraction(node.get_precision(), op, index, index)


@METALIBM_PASS_REGISTER
class Pass_SoftLegalize(Pass_RTLLegalize, FunctionPass):
    pass_tag = "soft_legalize"
    def __init__(self, target):
        Pass_RTLLegalize.__init__(self, target, tag="legalize SW node")

    def legalize_single_operation(self, node, format_solver):
        return sw_legalize_single_operation(node, format_solver)

    def execute_on_optree(self, optree, fct, fct_group, memoization_map):
        _, new_node = self.execute(optree)
        return new_node

Log.report(LOG_PASS_INFO, "Registering size_datapath pass")
# register pass
Pass.register(Pass_RTLLegalize)
