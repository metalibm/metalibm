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
# created:          Nov 17th, 2016
# last-modified:    Mar  7th, 2018
#
# author(s):    Nicolas Brunie (nbrunie@kalray.eu)
# description:  Implement a basic VHDL backend for hardware description
#               generation
###############################################################################

import operator

from ..utility.log_report import Log
from .generator_utility import *
from .complex_generator import *
from .code_element import *
from ..core.ml_formats import *
from ..core.ml_hdl_format import *
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *
from ..core.ml_hdl_operations import *
from ..core.legalizer import (
    min_legalizer, max_legalizer, fixed_point_position_legalizer,
    legalize_fixed_point_subselection, evaluate_cst_graph
)
from ..core.hdl_legalizer import (
    mantissa_extraction_modifier
)
from ..core.advanced_operations import FixedPointPosition
from metalibm_core.core.target import TargetRegister

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT

from ..opt.p_size_datapath import solve_format_Constant

from metalibm_hw_blocks.rtl_blocks import (
    zext, zext_to_size, sext, rzext,
    fp_is_nan, fp_is_pos_inf, fp_is_neg_inf, fp_is_infornan,
    fp_is_subnormal, fp_is_zero, fp_is_inf
)

# import metalibm_hw_blocks.lzc as ml_rtl_lzc

from .abstract_backend import AbstractBackend

from metalibm_core.opt.rtl_fixed_point_utils import (
    largest_format, test_format_equality
)

from metalibm_core.opt.opt_utils import evaluate_range, forward_attributes


def exclude_std_logic(optree):
    """ Backend matching predicate to exclude optree whose
        precision is a derivate from ML_StdLogicVectorFormat """
    return not isinstance(optree.get_precision(), ML_StdLogicVectorFormat)


def include_std_logic(optree):
    """ Backend matching predicate to include optree whose
        precision is a derivate from ML_StdLogicVectorFormat """
    return isinstance(optree.get_precision(), ML_StdLogicVectorFormat)

# return a lambda function associating to an output node
#  its transformation to a predicate node
#  @param predicate_function lambda optree -> predicate(optree)
#  @param kw dictionnary of attributes to attach to the resulting node
#  @return lambda function optree -> predicate(optree's input) annotated
#          with @p kw attributes


def test_modifier(predicate_function, **kw):
    return lambda optree, **kw: predicate_function(optree.get_input(0)).modify_attributes(**kw)

# Copy the value of the init_stage attribute field
#  from @p src node to @p dst node


def copy_init_stage(src, dst):
    init_stage = src.attributes.get_dyn_attribute("init_stage")
    dst.attributes.init_stage = init_stage


def shift_by_cst_legalizer(optree):
    """ Transform a shift by a constant into a Concatenation of a subsignal
        selection and a constant """
    op = optree.get_input(0)
    shift_amount_node = optree.get_input(1)
    assert isinstance(shift_amount_node, Constant)
    shift_amount = shift_amount_node.get_value()
    full_format = ML_StdLogicVectorFormat(optree.get_precision().get_bit_size())
    padding_format = ML_StdLogicVectorFormat(shift_amount)
    if shift_amount == 0:
        return op
    else:
        if isinstance(optree, BitLogicRightShift):
            lo_index = shift_amount
            hi_index = op.get_precision().get_bit_size() - 1
            raw_result = Concatenation(
                Constant(0, precision=padding_format),
                SubSignalSelection(
                    op,
                    lo_index,
                    hi_index
                ),
                precision=full_format
            )
        elif isinstance(optree, BitArithmeticRightShift):
            lo_index = shift_amount
            hi_index = op.get_precision().get_bit_size() - 1
            sign_digit = BitSelection(op, hi_index)
            raw_result = Concatenation(
                Replication(
                    sign_digit, Constant(shift_amount, precision=ML_Integer),
                    precision=padding_format
                ),
                SubSignalSelection(
                    op,
                    lo_index,
                    hi_index
                ),
                precision=full_format
            )
        elif isinstance(optree, BitLogicLeftShift):
            lo_index = 0
            hi_index = op.get_precision().get_bit_size() - 1 - shift_amount
            raw_result = Concatenation(
                SubSignalSelection(
                    op,
                    lo_index,
                    hi_index
                ),
                Constant(0, precision=padding_format),
                precision=full_format
            )
        else:
            raise NotImplementedError
        return TypeCast(
            raw_result,
            precision=optree.get_precision(),
        )

def zext_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")
    ext_input = optree.get_input(0)
    ext_size = optree.ext_size
    assert ext_size >= 0
    if ext_size == 0:
        Log.report(
            Log.Warning, "zext_modifer called with ext_size=0 on {}",
            optree
            )
        return ext_input
    else:
        precision = ML_StdLogicVectorFormat(ext_size)
        ext_precision = ML_StdLogicVectorFormat(
            ext_size + ext_input.get_precision().get_bit_size())
        result = Concatenation(
            Constant(0, precision=precision),
            ext_input, precision=ext_precision,
            tag=optree.get_tag("dummy") + "_zext",
            init_stage=init_stage
        )
        copy_init_stage(optree, result)
        forward_attributes(optree, result)
        return result

# Operation code generator modifier for Sign Extension


def sext_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    ext_size = optree.ext_size
    ext_input = optree.get_input(0)
    if ext_size == 0:
        Log.report(
            Log.Warning, "sext_modifer called with ext_size=0 on {}",
            optree
            )
        return ext_input
    else:
        ext_precision = ML_StdLogicVectorFormat(
            ext_size + ext_input.get_precision().get_bit_size())
        op_size = ext_input.get_precision().get_bit_size()
        sign_digit = VectorElementSelection(ext_input, Constant(
            op_size - 1, precision=ML_Integer), precision=ML_StdLogic, init_stage=init_stage)
        precision = ML_StdLogicVectorFormat(ext_size)
        result = Concatenation(
            Replication(
                sign_digit, Constant(ext_size, precision=ML_Integer),
                precision=precision, init_stage=init_stage
            ),
            ext_input, precision=ext_precision,
            tag=optree.get_tag("dummy") + "_sext",
            init_stage=init_stage
        )
        return result


def negation_modifer(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    neg_input = optree.get_input(0)
    precision = optree.get_precision()
    result = Addition(
        SignCast(
            BitLogicNegate(neg_input, precision=precision, init_stage=init_stage),
            specifier=SignCast.Unsigned,
            precision=get_unsigned_precision(precision)
        ),
        Constant(1, precision=ML_StdLogic),
        precision=precision,
        tag=optree.get_tag(),
        init_stage=init_stage
    )
    forward_attributes(optree, result)
    return result

def fixed_point_negation_modifier(optree):
    """ Legalize a Negation node on fixed-point inputs """
    op_format = optree.get_precision()
    op_input = optree.get_input(0)
    casted_format = ML_StdLogicVectorFormat(op_format.get_bit_size())
    casted_optree = TypeCast(
        Conversion(
            op_input,
            precision = op_format
        ),
        precision = casted_format,
        tag = "neg_casted_optree"
    )
    casted_negated = Negation(casted_optree, precision = casted_format)
    result = TypeCast(
        casted_negated,
        precision = op_format,
        tag = optree.get_tag() or "neg_casted_negated"
    )
    forward_attributes(optree, result)
    return result


def truncate_generator(optree):
    truncate_input = optree.get_input(0)
    result_size = optree.get_precision().get_bit_size()
    return TemplateOperator("%%s(%d downto 0)" % (result_size - 1), arity=1)


def conversion_generator(optree):
    output_size = optree.get_precision().get_bit_size()
    return TemplateOperator("conv_std_logic_vector(%s, {output_size})".format(output_size=output_size), arity=1)


# dynamic dyn_operator helper for Shift operations
#  @param dyn_operator string name of the operation
def shift_generator(dyn_operator, optree):
    width = optree.get_precision().get_bit_size()
    return TemplateOperator("conv_std_logic_vector({}(unsigned(%s), unsigned(%s)), {})".format(dyn_operator, width), arity=2, force_folding=True, force_input_variable=True)


# @p optree 0-th input has ML_Bool precision and must be converted
#  to optree's precision
def conversion_from_bool_generator(optree):
    op_input = optree.get_input(0)
    op_precision = optree.get_precision()
    return Select(op_input, Constant(1, precision=op_precision), Constant(0, precision=op_precision), precision=op_precision)


def copy_sign_generator(optree):
    sign_input = optree.get_input(0)
    sign_index = sign_input.get_precision().get_bit_size() - 1
    return TemplateOperator("%%s(%d)" % (sign_index), arity=1)


def sub_signal_generator(optree):
    op = optree.get_input(0)
    inf_index = evaluate_cst_graph(optree.get_inf_index())
    sup_index = evaluate_cst_graph(optree.get_sup_index())
    assert isinstance(inf_index, int)
    assert isinstance(sup_index, int)
    if isinstance(op, Constant):
        assert sup_index >= inf_index
        value = op.get_value()
        code_value = (value >> inf_index) & (2**(sup_index-inf_index+1) - 1)
        return TemplateOperator(
            ("\"{:0%db}\"" % (sup_index - inf_index + 1)).format(code_value),
            arity=0, force_folding=True
        )

    else:
        range_direction = "to" if (inf_index > sup_index) else "downto"
        return TemplateOperator(
            "%s({sup_index} {direction} {inf_index})".format(
                inf_index=inf_index, direction=range_direction,
                sup_index=sup_index
            ), arity=1, force_folding=True,
            no_parenthesis=True
        )

def integer2fixed_conversion_modifier(optree):
    cst_graph = optree.get_input(0)
    cst_val = evaluate_cst_graph(cst_graph)
    result = Constant(cst_val, precision = optree.get_precision())
    forward_attributes(optree, result)
    return result

def bool2fixed_conversion_modifier(optree):
    bool_value = optree.get_input(0)
    op_prec = optree.get_precision()
    result = Select(
        bool_value,
        Constant(1, precision = op_prec),
        Constant(0, precision = op_prec),
        precision = op_prec
    )
    forward_attributes(optree, result)
    return result

# @p optree is a conversion node which should be modified
def fixed_conversion_modifier(optree):
    arg = optree.get_input(0)
    conv_precision = optree.get_precision()
    arg_precision = arg.get_precision()
    assert is_fixed_point(conv_precision)
    assert is_fixed_point(arg_precision)
    if not ((conv_precision.get_signed() and arg_precision.get_signed()) or not arg_precision.get_signed()):
        Log.report(
            Log.Warning,
            "incompatible input -> output precision signedess in fixed_conversion_modifier:\n"
            "  input is {}, output is {}",
            arg_precision, conv_precision
        )
    result = arg
    raw_arg_precision = ML_StdLogicVectorFormat(
        arg_precision.get_bit_size()
    )
    result_raw = TypeCast(
        result,
        precision=raw_arg_precision,
        tag = "fixed_conv_result_raw"
    )
    signed = arg_precision.get_signed()
    int_ext_size = conv_precision.get_integer_size() - arg_precision.get_integer_size()
    if int_ext_size > 0:
        result_raw = (sext if signed else zext)(result_raw, int_ext_size)
    elif int_ext_size < 0:
        # sub-signal extraction
        result_raw = SubSignalSelection(
            result_raw, 0,
            result_raw.get_precision().get_bit_size() - 1 + int_ext_size
        )
    frac_ext_size = conv_precision.get_frac_size() - arg_precision.get_frac_size()
    if frac_ext_size > 0:
        result_raw = rzext(result_raw, frac_ext_size)
    elif frac_ext_size < 0:
        # sub-signal extraction
        result_raw = SubSignalSelection(
            result_raw, -frac_ext_size,
            result_raw.get_precision().get_bit_size() - 1
        )
    #forward_attributes(optree, result_raw)
    result = TypeCast(
        result_raw,
        precision=conv_precision,
        tag="final_fixed_conv_result"
    )
    forward_attributes(optree, result)
    return result

def legalizing_fixed_shift_amount(shift_amount):
    precision = shift_amount.get_precision()
    if is_fixed_point(precision):
        sa_range = evaluate_range(shift_amount)
        if inf(sa_range) < 0:
            Log.report(
                Log.Error,
                "shift amount {} whose range has been evaluated to {} may take negative values",
                shift_amount,
                sa_range
            )
        # TODO support negative frac size via static amount shifting
        assert(precision.get_frac_size() == 0)
        casted_format = ML_StdLogicVectorFormat(precision.get_bit_size())
        casted_sa = TypeCast(
            shift_amount,
            precision = casted_format,
            tag = shift_amount.get_tag() or "casted_sa"
        )
        forward_attributes(shift_amount, casted_sa)
        return casted_sa
    else:
        return shift_amount


def fixed_shift_modifier(shift_class):
    """ legalize a shift node on fixed-point operation tree

        Args:
            optree (ML_Operation): operation node input
        Returns:
            legalize node
    """
    def fixed_shift_modifier_routine(optree):
        shift_input = optree.get_input(0)
        out_precision = optree.get_precision()
        shift_amount = legalizing_fixed_shift_amount(optree.get_input(1))
        converted_input = Conversion(
            shift_input,
            precision = optree.get_precision()
        )
        # check precision equality
        casted_format = ML_StdLogicVectorFormat(out_precision.get_bit_size())
        casted_input = TypeCast(
            converted_input,
            precision = casted_format,
            tag = "shift_casted_input"
        )
        # inserting shift
        casted_shift = shift_class(
            casted_input,
            shift_amount,
            precision = casted_format
        )
        # casting back
        fixed_result = TypeCast(
            casted_shift,
            precision = out_precision,
            tag = optree.get_tag() or "shift_fixed_result"
        )
        forward_attributes(optree, fixed_result)
        return fixed_result
    return fixed_shift_modifier_routine


# If @p optree's precision does not match @p new_format
#  insert a conversion


def convert_if_needed(optree, new_format):
    if not test_format_equality(optree.get_precision(), new_format):
        return Conversion(
            optree,
            precision=new_format
        )
    else:
        return optree

# @p optree is a comparison node between fixed-point value
#  to be legalized


def fixed_comparison_modifier(optree):
    lhs, rhs = optree.get_input(0), optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()
    assert is_fixed_point(lhs_precision)
    assert is_fixed_point(rhs_precision)
    # before comparing the value we must scale them to an integer format
    # compatible with std_logic_vector support format
    unified_format = largest_format(lhs_precision, rhs_precision)
    lhs = convert_if_needed(lhs, unified_format)
    rhs = convert_if_needed(rhs, unified_format)
    lhs = TypeCast(
        lhs,
        precision=lhs.get_precision().get_support_format(),
        tag = "comp_lhs_casted"
    )
    rhs = TypeCast(
        rhs,
        precision=rhs.get_precision().get_support_format(),
        tag = "comp_rhs_casted"
    )
    lhs = SignCast(
        lhs,
        specifier=SignCast.Signed if lhs_precision.get_signed() else
        SignCast.Unsigned,
        precision=get_numeric_precision(lhs.get_precision(), lhs_precision.get_signed()),
        tag = "comp_lhs_signcasted"
    )
    rhs = SignCast(
        rhs,
        specifier=SignCast.Signed if rhs_precision.get_signed() else
        SignCast.Unsigned,
        precision=get_numeric_precision(rhs.get_precision(), rhs_precision.get_signed()),
        tag = "comp_rhs_signcasted"
    )
    # we must keep every initial properties of the Comparison node
    # except the operand nodes
    result = optree.copy(
        copy_map={
            optree.get_input(0): lhs,
            optree.get_input(1): rhs
        }
    )
    forward_attributes(optree, result)
    return result

# adapt a fixed-optree @p raw_result assumimg fixed format
#  with @p integer_size and @p frac_size
# to match format of @optree


def adapt_fixed_optree(raw_optree, integer_frac_size, optree):
    # extracting params
    integer_size, frac_size = integer_frac_size
    optree_prec = optree.get_precision()
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    # MSB extension/reduction (left)
    msb_delta = optree_prec.get_integer_size() - integer_size
    if msb_delta >= 0:
        result_lext = (sext if optree_prec.get_signed()
                       else zext)(raw_optree, msb_delta)
    else:
        result_lext = SubSignalSelection(
            raw_optree, 0, frac_size + integer_size - 1 + msb_delta)
    # LSB extension/reduction (right)
    lsb_delta = optree_prec.get_frac_size() - frac_size
    if lsb_delta >= 0:
        result_rext = rzext(result_lext, lsb_delta)
    else:
        result_rext = SubSignalSelection(
            result_lext,
            -lsb_delta,
            result_lext.get_precision().get_bit_size() - 1
        )
    # final format casting
    result = TypeCast(
        result_rext,
        init_stage=init_stage,
        precision=optree_prec,
        tag = optree.get_tag() or "fixed_adapt_result"
    )
    forward_attributes(optree, result)
    return result

# fixed point operation generation block


def fixed_point_op_modifier(optree, op_ctor=Addition):
    # TODO: fixed node attribute transmission
    init_stage = optree.attributes.get_dyn_attribute("init_stage")
    # left hand side and right hand side operand extraction
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_prec = lhs.get_precision().get_base_format()
    rhs_prec = rhs.get_precision().get_base_format()
    optree_prec = optree.get_precision().get_base_format()

    if lhs.get_precision() is ML_Integer:
        lhs_value = evaluate_cst_graph(lhs)
        new_node = Constant(lhs_value)
        lhs_prec = solve_format_Constant(new_node)
        new_node.set_precision(lhs_prec)
        lhs = new_node

    if rhs.get_precision() is ML_Integer:
        rhs_value = evaluate_cst_graph(rhs)
        new_node = Constant(rhs_value)
        rhs_prec = solve_format_Constant(new_node)
        new_node.set_precision(rhs_prec)
        rhs = new_node

    # TODO: This assume integer_size is at least 0 (no negative
    # (frac end before fixed point) accepted
    result_frac_size = max(lhs_prec.get_frac_size(), rhs_prec.get_frac_size())
    result_integer_size = max(
        lhs_prec.get_integer_size(),
        rhs_prec.get_integer_size()
    ) + 1

    lhs_casted = TypeCast(
        lhs,
        precision=lhs_prec.get_support_format(),
        init_stage=init_stage,
        tag = "op_lhs_casted"
    )
    rhs_casted = TypeCast(
        rhs,
        precision=rhs_prec.get_support_format(),
        init_stage=init_stage,
        tag = "op_rhs_casted"
    )

    # vhdl does not support signed <op> unsigned unless result size is increased
    # those we always SignCast the same way both operands
    signed_op = lhs_prec.get_signed() or rhs_prec.get_signed()

    lhs_ext = (sext if lhs_prec.get_signed() else zext)(
        rzext(lhs_casted, result_frac_size - lhs_prec.get_frac_size()),
        result_integer_size - lhs_prec.get_integer_size()
    )
    lhs_ext = SignCast(
        lhs_ext, precision=get_numeric_precision(lhs_ext.get_precision(), signed_op),
        specifier=SignCast.Signed if signed_op else SignCast.Unsigned
    )

    rhs_ext = (sext if rhs_prec.get_signed() else zext)(
        rzext(rhs_casted, result_frac_size - rhs_prec.get_frac_size()),
        result_integer_size - rhs_prec.get_integer_size()
    )
    rhs_ext = SignCast(
        rhs_ext, precision=get_numeric_precision(rhs_ext.get_precision(), signed_op),
        specifier=SignCast.Signed if signed_op else SignCast.Unsigned
    )
    raw_result = op_ctor(
        lhs_ext,
        rhs_ext,
        precision=ML_StdLogicVectorFormat(
            result_frac_size + result_integer_size)
    )
    adapted_result = adapt_fixed_optree(
        raw_result, (result_integer_size, result_frac_size), optree
    )
    return adapted_result


def fixed_point_mul_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    # left hand side and right hand side operand extraction
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_prec = lhs.get_precision().get_base_format()
    rhs_prec = rhs.get_precision().get_base_format()

    # if lhs and rhs operand sign, an exta MSB digit must be added
    extra_sign_digit = 1 if lhs_prec.get_signed() ^ rhs_prec.get_signed() else 0

    optree_prec = optree.get_precision().get_base_format()
    # max, optree_prec.get_frac_size())
    result_frac_size = (lhs_prec.get_frac_size() + rhs_prec.get_frac_size())
    # max, optree_prec.get_integer_size())
    result_integer_size = (lhs_prec.get_integer_size() +
                           rhs_prec.get_integer_size() +
                           extra_sign_digit)
    #assert optree_prec.get_frac_size() >= result_frac_size
    #assert optree_prec.get_integer_size() >= result_integer_size
    lhs_casted = TypeCast(
        lhs,
        precision=ML_StdLogicVectorFormat(
            lhs_prec.get_bit_size()
        ), init_stage=init_stage,
        tag = "mul_lhs_casted",
    )
    lhs_casted = SignCast(
        lhs_casted,
        precision=get_numeric_precision(lhs_casted.get_precision(), lhs_prec.get_signed()),
        specifier=SignCast.Signed if lhs_prec.get_signed() else SignCast.Unsigned)
    rhs_casted = TypeCast(
        rhs,
        precision=ML_StdLogicVectorFormat(
            rhs_prec.get_bit_size()
        ), init_stage=init_stage,
        tag = "mul_rhs_casted",
    )
    rhs_casted = SignCast(
        rhs_casted,
        precision=get_numeric_precision(rhs_casted.get_precision(), rhs_prec.get_signed()),
        specifier=SignCast.Signed if rhs_prec.get_signed() else SignCast.Unsigned)

    mult_prec = ML_StdLogicVectorFormat(result_frac_size + result_integer_size)
    Log.report(Log. Verbose, "Multiplication {}: {} x {} = {} bits".format(
        optree.get_tag(),
        lhs_casted.get_precision().get_bit_size(),
        rhs_casted.get_precision().get_bit_size(),
        mult_prec.get_bit_size()
    ))
    raw_result = Multiplication(
        lhs_casted,
        rhs_casted,
        precision=mult_prec,
        tag=optree.get_tag(),
        init_stage=init_stage
    )
    # adapting raw result to output format
    return adapt_fixed_optree(
        raw_result, (result_integer_size, result_frac_size), optree
    )


def fixed_point_add_modifier(optree):
    return fixed_point_op_modifier(optree, op_ctor=Addition)


def fixed_point_sub_modifier(optree):
    return fixed_point_op_modifier(optree, op_ctor=Subtraction)


vhdl_comp_symbol = {
    Comparison.Equal: "=",
    Comparison.NotEqual: "/=",
    Comparison.Less: "<",
    Comparison.LessOrEqual: "<=",
    Comparison.GreaterOrEqual: ">=",
    Comparison.Greater: ">",
    Comparison.LessSigned: "<",
    Comparison.LessOrEqualSigned: "<=",
    Comparison.GreaterOrEqualSigned: ">=",
    Comparison.GreaterSigned: ">",
}


def get_vhdl_bool_cst(self, value):
    if value:
        return "true"
    else:
        return "false"


# TODO: factorize following statements into
#  a clean function with correct execution as
#  "vhdl backend" install
# Updating standard format name for VHDL Code
ML_Integer.name[VHDL_Code] = "integer"
ML_Bool.name[VHDL_Code] = "boolean"
ML_Bool.get_cst_map[VHDL_Code] = get_vhdl_bool_cst

ML_String.name[VHDL_Code] = "string"
ML_String.get_cst_map[VHDL_Code] = ML_String.get_cst_map[C_Code]


HDL_FILE.name[VHDL_Code] = "file"
HDL_FILE.get_cst_map[VHDL_Code] = ML_String.get_cst_map[C_Code]

HDL_LINE.name[VHDL_Code] = "line"
HDL_LINE.get_cst_map[VHDL_Code] = ML_String.get_cst_map[C_Code]

# class Match custom std logic vector format
MCSTDLOGICV = TCM(ML_StdLogicVectorFormat)

# class Match custom unsigned vector format
MCHDLUNSIGNEDV = TCM(HDL_UnsignedVectorFormat)

# class Match custom unsigned vector format
MCHDLSIGNEDV = TCM(HDL_SignedVectorFormat)

MCHDLNUMERICV = TCM(HDL_NumericVectorFormat)

# class match custom fixed point format
MCFixedPoint = TCM(ML_Base_FixedPoint_Format)

formal_generation_table = {
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("+", arity=2, force_folding=False),
            },
        },
    },
    Subtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("-", arity=2, force_folding=False),
            },
        },
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("*", arity=2, force_folding=False),
            },
        },
    },
}


def bit_selection_legalizer(optree):
    assert isinstance(optree, VectorElementSelection)
    op_input = optree.get_input(0)
    op_index = optree.get_input(1)
    input_precision = op_input.get_precision()
    if is_fixed_point(input_precision):
        cast_format = ML_StdLogicVectorFormat(input_precision.get_bit_size())
        result = VectorElementSelection(
            TypeCast(
                op_input,
                precision = cast_format,
                tag = "bit_sel_input_casted"
            ),
            op_index,
            precision = optree.get_precision()
        )
        forward_attributes(optree, result)
        return result
    else:
        return optree

def bit_to_vector_conversion(optree):
    """ legalize a  std_logic to std_logic_vector(0 downto 0) """
    op_input = optree.get_input(0)
    assert optree.get_precision().get_bit_size() == 1
    result = Signal("conv", precision=optree.get_precision(), var_type=Variable.Local)
    vector_assign = ReferenceAssign(BitSelection(result, 0), op_input)
    return PlaceHolder(result, vector_assign)


def conversion_legalizer(optree):
    """ Legalize conversion operation. Currently supports
        legalization of ML_StdLogicVectorFormat to ML_StdLogicVectorFormat
        conversion """
    conv_input = optree.get_input(0)
    if MCSTDLOGICV(optree.get_precision()) and MCSTDLOGICV(conv_input.get_precision()):
        conv_input_size = conv_input.get_precision().get_bit_size()
        out_size = optree.get_precision().get_bit_size()
        result = None

        if conv_input_size == out_size:
            result = conv_input
        elif conv_input_size > out_size:
            result = SubSignalSelection(
                conv_input,
                0,
                out_size - 1,
                precision = optree.get_precision()
            )
        else:
            result = zext_to_size(
                conv_input,
                optree.get_precision().get_bit_size()
            )
        forward_attributes(optree, result)
        return result
    else:
        raise NotImplementedError


def fixed_cast_legalizer(optree):
    """ Legalize a TypeCast operation from a fixed-point input to a
        fixed-point output """
    assert isinstance(optree, TypeCast)
    out_prec = optree.get_precision()
    cast_input = optree.get_input(0)
    input_prec = cast_input.get_precision()
    assert is_fixed_point(out_prec) and is_fixed_point(input_prec)
    if input_prec.get_bit_size() != out_prec.get_bit_size():
        Log.report(
            Log.Error,
            "fixed_cast_legalizer only support same size I/O, input is {}\n, output is {}",
            cast_input,
            optree,
        )
        raise NotImplementedError
    else:
        # Using an exchange format derived from ML_StdLogicVectorFormat
        exch_format = ML_StdLogicVectorFormat(input_prec.get_bit_size())
        result = TypeCast(
            TypeCast( 
                cast_input,
                precision = exch_format,
                tag = "fixed_cast_input"
            ),
            precision = out_prec,
            tag = optree.get_tag() or "fixed_cast_legalizer"
        )
        forward_attributes(optree, result)
        return result


def vel_arg_process(code_object, code_generator, arg_list):
    """ VectorElementSelection argument processing """
    vel_input, vel_index = arg_list
    if isinstance(vel_input, CodeExpression):
        # declare input
        prefix = "vel_input"
        result_varname = code_object.get_free_var_name(vel_input.precision, prefix = prefix)
        code_object << code_generator.generate_assignation(result_varname, vel_input.get())
        vel_input = CodeVariable(result_varname, vel_input.precision)
    return [vel_input, vel_index]

# optree_modifier will be modified once ML_LeadingZeroCounter has been instanciated
# (as its depends on VHDLBackend, this can not happen here)
def fallback_modifier(_):
    raise NotImplementedError
handle_LZC_legalizer = ComplexOperator(optree_modifier = fallback_modifier)

vhdl_code_generation_table = {
    FixedPointPosition: {
        None: {
            lambda optree: True: {
                type_custom_match(type_all_match, MCFixedPoint, type_all_match):
                    ComplexOperator(optree_modifier = fixed_point_position_legalizer),
            }
        },
    },
    CountLeadingZeros: {
        None: {
            lambda optree: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint):
                    handle_LZC_legalizer,
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV):
                    handle_LZC_legalizer,
            }
        },
    },
    Min: {
        None: {
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=min_legalizer),
            }
        },
    },
    Max: {
        None: {
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=max_legalizer),
            }
        },
    },
    Addition: {
        None: {
            exclude_std_logic:
            build_simplified_operator_generation_nomap([
                ML_Int16, ML_UInt16, ML_Int32, ML_UInt32,
                ML_Int64, ML_UInt64
            ], 2, SymbolOperator("+", arity=2, force_folding=True),
                cond=(lambda _: True)
            ),
            include_std_logic:
            {
                type_custom_match(MCSTDLOGICV, MCHDLNUMERICV, MCHDLNUMERICV):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, MCHDLNUMERICV, FSM(ML_StdLogic)):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_StdLogic), MCHDLNUMERICV):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, MCHDLNUMERICV, FSM(ML_StdLogic)):
                SymbolOperator("+", arity=2, force_folding=True),
            },
            # fallback
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_point_add_modifier),
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=fixed_point_add_modifier),
                type_custom_match(MCFixedPoint, FSM(ML_Integer), MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_point_add_modifier),
            }
        }
    },
    Subtraction: {
        None: {
            exclude_std_logic:
            build_simplified_operator_generation_nomap([v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64,
                                                        ML_UInt64, ML_Int128, ML_UInt128], 2, SymbolOperator("-", arity=2, force_folding=True), cond=(lambda _: True)),
            include_std_logic:
            {
                type_custom_match(MCSTDLOGICV, MCHDLNUMERICV, MCHDLNUMERICV):  SymbolOperator("-", arity=2, force_folding=True),
            },
            # fallback
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_point_sub_modifier),
                type_custom_match(MCFixedPoint, FSM(ML_Integer), MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_point_sub_modifier),
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=fixed_point_sub_modifier),
            }
        }
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCHDLNUMERICV, MCHDLNUMERICV): SymbolOperator("*", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_point_mul_modifier),
            },
        },
    },
    BitLogicNegate: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_StdLogic, ML_StdLogic): FunctionOperator("not", arity=1),
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("not", arity=1),
                type_custom_match(MCFixedPoint, MCFixedPoint): FunctionOperator("not", arity = 1),
            },
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): 
                    ComplexOperator(optree_modifier=negation_modifer),
                type_custom_match(MCFixedPoint, MCFixedPoint): 
                    ComplexOperator(optree_modifier = fixed_point_negation_modifier),
            },
        },
    },
    LogicalAnd: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_Bool, ML_Bool): SymbolOperator("and", arity=2, force_folding=False),
            },
        },
    },
    LogicalOr: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_Bool, ML_Bool): SymbolOperator("or", arity=2, force_folding=False),
            },
        },
    },
    LogicalNot: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_Bool): FunctionOperator("not", arity=1, force_folding=False),
            },
        },
    },
    Event: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_StdLogic): SymbolOperator("\'event", lspace="", inverse=True, arity=1, force_folding=False),
            },
        },
    },
    Comparison:
    dict(
        [(specifier,
          {
              lambda _: True: {
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary64), \
                                    FSM(ML_Binary64)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2,
                      force_folding=False
                  ),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary32), FSM(ML_Binary32)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary16), FSM(ML_Binary16)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), MCHDLNUMERICV, MCHDLNUMERICV):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_strict_match(ML_Bool, ML_StdLogic, ML_StdLogic):
                    SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  # fixed-point comparison
                  type_custom_match(FSM(ML_Bool), MCFixedPoint, MCFixedPoint):
                  ComplexOperator(
                      optree_modifier=fixed_comparison_modifier
                  )
              },
          }) for specifier in [Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]]
        +
        [(specifier,
          {
              lambda _: True: {
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary64), \
                                    FSM(ML_Binary64)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2,
                      force_folding=False
                  ),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary32), FSM(ML_Binary32)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary16), FSM(ML_Binary16)):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), MCSTDLOGICV, MCSTDLOGICV):
                      SymbolOperator(
                          vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), MCHDLNUMERICV, MCHDLNUMERICV):
                      SymbolOperator(
                          vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_strict_match(ML_Bool, ML_StdLogic, ML_StdLogic):
                    SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  # fixed-point comparison
                  type_custom_match(FSM(ML_Bool), MCFixedPoint, MCFixedPoint):
                  SymbolOperator(
                      vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  # string comparison
                  type_custom_match(FSM(ML_Bool), FSM(ML_String), FSM(ML_String)):
                      SymbolOperator(
                          vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                  type_custom_match(FSM(ML_Bool), TCM(ML_StringClass), TCM(ML_StringClass)):
                      SymbolOperator(
                          vhdl_comp_symbol[specifier], arity=2, force_folding=False),
              },
          }) for specifier in [Comparison.Equal, Comparison.NotEqual]
          ]
        +
        [(specifier,
            {
                lambda _: True: {
                    type_custom_match(FSM(ML_Bool), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)):
                    TemplateOperator("signed(%%s) %s signed(%%s)" %
                                     vhdl_comp_symbol[specifier], arity=2, force_folding=False),
                },
            }) for specifier in [Comparison.GreaterSigned, \
                                 Comparison.GreaterOrEqualSigned, \
                                 Comparison.LessSigned, \
                                 Comparison.LessOrEqualSigned]
         ]
    ),
    ExponentExtraction: {
        None: {
            lambda _: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary64)):
                SymbolOperator("(62 downto 52)", lspace="", inverse=True, \
                               arity=1),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary32)):
                SymbolOperator("(30 downto 23)", lspace="", inverse=True, \
                               arity=1),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(BFloat16_Base)):
                SymbolOperator("(14 downto 7)", lspace="", inverse=True, \
                               arity=1),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary16)):
                SymbolOperator("(14 downto 10)", lspace="", inverse=True, \
                               arity=1),
            },
        },
    },
    ZeroExt: {
        None: {
            lambda _: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)):
                    ComplexOperator(optree_modifier=zext_modifier),
                type_custom_match(MCSTDLOGICV, FSM(ML_StdLogic)):
                    TemplateOperatorFormat("(0 => {0}, others => '0')", arity=1, force_folding=True),
            },
        }
    },
    SignExt: {
        None: {
            lambda _: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): ComplexOperator(optree_modifier=sext_modifier),
            },
        }
    },
    Concatenation: {
        None: {
            lambda _: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), TCM(ML_StdLogicVectorFormat)): SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic)): SymbolOperator("&", arity=2, force_folding=True),
                # only valid for 2-bit vectors
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), FSM(ML_StdLogic)): SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, MCSTDLOGICV, MCSTDLOGICV):
                    SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, FSM(ML_StdLogic), MCSTDLOGICV):
                    SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, MCSTDLOGICV, FSM(ML_StdLogic)):
                    SymbolOperator("&", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, FSM(ML_StdLogic), MCFixedPoint):
                    SymbolOperator("&", arity=2, force_folding=True),

                type_custom_match(FSM(ML_String), FSM(ML_String), FSM(ML_String)):
                    SymbolOperator("&", arity=2, force_folding=False, force_input_variable=False),
            },
        },
    },
    VectorElementSelection: {
        None: {
            # make sure index accessor is a Constant (or fallback to C implementation)
            lambda optree: True:  {
                type_custom_match(
                    FSM(ML_StdLogic),
                    TCM(ML_StdLogicVectorFormat),
                    type_all_match
                ): 
                    TemplateOperator("%s(%s)", arity=2, process_arg_list = vel_arg_process),
                type_custom_match(FSM(ML_StdLogic), MCFixedPoint, type_all_match): 
                    ComplexOperator(optree_modifier = bit_selection_legalizer),
            },
        },
    },
    Replication: {
        None: {
            lambda optree: True:  {
                type_custom_match(FSM(ML_StdLogic), FSM(ML_StdLogic)): 
                    TransparentOperator(), # IdentityOperator(),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), FSM(ML_Integer)): TemplateOperatorFormat("(0 to {1} - 1  => {0:s})", arity=2),
            },
        },
    },
    Conversion: {
        None: {
            lambda optree: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Integer)):
                    DynamicOperator(conversion_generator),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic)):
                    ComplexOperator(optree_modifier=bit_to_vector_conversion),
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV):
                    ComplexOperator(optree_modifier = conversion_legalizer),
                type_strict_match(ML_StdLogic, ML_Bool):
                    ComplexOperator(
                        optree_modifier=conversion_from_bool_generator),
                type_custom_match(FSM(ML_String), TCM(ML_StdLogicVectorFormat)):
                    FunctionOperator("to_hstring", arity=1, force_folding=False),
                type_custom_match(FSM(ML_String), FSM(ML_StdLogic)):
                    FunctionOperator("std_logic'image", arity=1, force_folding=False),
                # fixed-point conversion support
                type_custom_match(MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier=fixed_conversion_modifier),
                type_custom_match(MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=integer2fixed_conversion_modifier),
                type_custom_match(MCFixedPoint, FSM(ML_Bool)):
                    ComplexOperator(optree_modifier=bool2fixed_conversion_modifier),
            }
        },
    },
    MantissaExtraction: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)):
                    ComplexOperator(optree_modifier=mantissa_extraction_modifier),
                type_custom_match(MCSTDLOGICV, FSM(BFloat16_Base)):
                    ComplexOperator(optree_modifier=mantissa_extraction_modifier),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)):
                    ComplexOperator(optree_modifier=mantissa_extraction_modifier),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)):
                    ComplexOperator(optree_modifier=mantissa_extraction_modifier),
            },
        },
    },
    BitLogicXor: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("xor", arity=2),
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("xor", arity=2),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("and", arity=2),
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("and", arity=2),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("or", arity=2),
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("or", arity=2),
            },
        },
    },
    Truncate: {
        None: {
            lambda optree: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): DynamicOperator(truncate_generator),
            },
        },
    },
    SignCast: {
        SignCast.Signed: {
            lambda optree: True: {
                type_custom_match(MCHDLSIGNEDV, MCSTDLOGICV):
                    FunctionOperator("signed",
                        arity=1,
                        force_folding=False, force_input_variable = True
                    ),
                type_custom_match(MCFixedPoint, MCFixedPoint):
                    TransparentOperator(no_parenthesis=True),
            },
        },
        SignCast.Unsigned: {
            lambda optree: True: {
                type_custom_match(MCHDLUNSIGNEDV, MCSTDLOGICV):
                    FunctionOperator(
                        "unsigned", arity=1, force_folding=False,
                        force_input_variable = True
                    ),
                type_custom_match(MCFixedPoint, MCFixedPoint):
                    TransparentOperator(no_parenthesis=True),
            },
        },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Binary16), TCM(ML_StdLogicVectorFormat)):
					TransparentOperator(output_precision=ML_Binary16, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary16), FSM(ML_Binary16)):
					TransparentOperator(output_precision=ML_Binary16, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)):
					TransparentOperator(no_parenthesis=True),

                type_custom_match(FSM(BFloat16_Base), TCM(ML_StdLogicVectorFormat)):
					TransparentOperator(output_precision=BFloat16_Base, no_parenthesis=True),
                type_custom_match(FSM(BFloat16_Base), FSM(BFloat16_Base)):
					TransparentOperator(output_precision=BFloat16_Base, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(BFloat16_Base)):
					TransparentOperator(no_parenthesis=True),

                type_custom_match(FSM(ML_Binary32), TCM(ML_StdLogicVectorFormat)):
					TransparentOperator(output_precision=ML_Binary32, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary32), FSM(ML_Binary32)):
					TransparentOperator(output_precision=ML_Binary32, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)):
					TransparentOperator(no_parenthesis=True),

                type_custom_match(FSM(ML_Binary64), TCM(ML_StdLogicVectorFormat)):
					TransparentOperator(output_precision=ML_Binary64, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary64), FSM(ML_Binary64)):
					TransparentOperator(output_precision=ML_Binary64, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)):
					TransparentOperator(no_parenthesis=True),

                type_custom_match(MCSTDLOGICV, MCFixedPoint):
                    TransparentOperator(no_parenthesis=True),
                type_custom_match(MCFixedPoint, MCSTDLOGICV):
                    TransparentOperator(no_parenthesis=True),
                type_custom_match(MCFixedPoint, FSM(ML_StdLogic)):
                    TransparentOperator(no_parenthesis=True),
                type_custom_match(FSM(ML_StdLogic), MCFixedPoint):
                    TransparentOperator(no_parenthesis=True),

                type_custom_match(MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier = fixed_cast_legalizer),
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV):
                    TransparentOperator(no_parenthesis=True)

            },
        },
    },
    BitArithmeticRightShift: {
        None: {
            lambda optree: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=shift_by_cst_legalizer),
            },
        },
    },
    BitLogicRightShift: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):
                    DynamicOperator(lambda optree: shift_generator("shr", optree)),
                # FIXME/TODO: add is cst check on shift amount (ML_Integer format is not enough)
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=shift_by_cst_legalizer),
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=shift_by_cst_legalizer),

                type_custom_match(MCFixedPoint, MCFixedPoint, MCSTDLOGICV):
                    ComplexOperator(
                        optree_modifier = fixed_shift_modifier(BitLogicRightShift)
                    ),
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(
                        optree_modifier=fixed_shift_modifier(BitLogicRightShift)
                    ),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(
                        optree_modifier = fixed_shift_modifier(BitLogicRightShift)),
            },
        },
    },
    BitLogicLeftShift: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):
                    DynamicOperator(lambda optree: shift_generator("shl", optree)),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCSTDLOGICV):
                    ComplexOperator(optree_modifier = fixed_shift_modifier(BitLogicLeftShift)),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
                    ComplexOperator(optree_modifier = fixed_shift_modifier(BitLogicLeftShift)),

                # shifts by Integer constant
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=shift_by_cst_legalizer),
                type_custom_match(MCFixedPoint, MCFixedPoint, FSM(ML_Integer)):
                    ComplexOperator(optree_modifier=shift_by_cst_legalizer),
            },
        },
    },
    #CountLeadingZeros: {
    #    None: {
    #        lambda optree: True: {
    #            type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("count_leading_zeros", arity=1),
    #        },
    #    },
    #},
    #CopySign: {
    #    SpecificOperation.Copy: {
    #        lambda optree: True: {
    #            type_custom_match(FSM(ML_StdLogic), ML_Binary16): TemplateOperator("%s(15)", arity=1),
    #            type_custom_match(FSM(ML_StdLogic), ML_Binary32): TemplateOperator("%s(31)", arity=1),
    #            type_custom_match(FSM(ML_StdLogic), ML_Binary64): TemplateOperator("%s(63)", arity=1),
    #            type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV): DynamicOperator(copy_sign_generator),
    #            type_custom_match(FSM(ML_StdLogic), MCFixedPoint): DynamicOperator(copy_sign_generator),
    #        },
    #    },
    #},
    SpecificOperation: {
        SpecificOperation.CopySign: {
            lambda optree: True: {
                type_custom_match(FSM(ML_StdLogic), FSM(BFloat16_Base)):
                    TemplateOperator("%s(15)", arity=1),
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary16)):
                    TemplateOperator("%s(15)", arity=1),
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary32)):
                    TemplateOperator("%s(31)", arity=1),
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary64)):
                    TemplateOperator("%s(63)", arity=1),
                type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV):
                    DynamicOperator(copy_sign_generator),
                type_custom_match(FSM(ML_StdLogic), MCFixedPoint):
                    DynamicOperator(copy_sign_generator),
            },
        },
    },
    SubSignalSelection: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, type_strict_match(ML_Integer), type_strict_match(ML_Integer)):
                    DynamicOperator(sub_signal_generator),
                type_custom_match(MCFixedPoint, MCFixedPoint, type_strict_match(ML_Integer), type_strict_match(ML_Integer)): 
                    ComplexOperator(optree_modifier=legalize_fixed_point_subselection),
                type_custom_match(MCSTDLOGICV, MCFixedPoint, type_strict_match(ML_Integer), type_strict_match(ML_Integer)): 
                    ComplexOperator(optree_modifier=legalize_fixed_point_subselection),
            },
        },
    },
    # Floating-point predicates
    Test: {
        Test.IsNaN: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_nan)),
            }
        },
        Test.IsPositiveInfty: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_pos_inf)),
            }
        },
        Test.IsNegativeInfty: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_neg_inf)),
            }
        },
        Test.IsInfty: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_inf)),
            }
        },
        Test.IsInfOrNaN: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_infornan)),
            }
        },
        Test.IsZero: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_zero)),
            }
        },
        Test.IsSubnormal: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Bool), TCM(ML_Std_FP_Format)): ComplexOperator(optree_modifier=test_modifier(fp_is_subnormal)),
            }
        },
    },
    Report: {
        None: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Void), FSM(ML_String)):
                TemplateOperatorFormat(
                    "report {0}",
                    arity=1,
                    force_folding=False,
                    force_input_variable=False,
                    void_function=True
                ),
            }
        },
    },
    FunctionCall: {
        None: {
            lambda optree: True: {
                type_function_match: FunctionObjectOperator(),
            },
        },
    },
}


class FormalBackend(AbstractBackend):
    """ description of a formal Backend """
    target_name = "formal_backend"
    TargetRegister.register_new_target(target_name, lambda _: FormalBackend)

    code_generation_table = {
        VHDL_Code: formal_generation_table,
        C_Code: formal_generation_table,
        Gappa_Code: {}
    }

    def __init__(self):
        AbstractBackend.__init__(self)
        Log.report(LOG_BACKEND_INIT, "initializing an instance of Formal target")


class VHDLBackend(FormalBackend):
    """ description of VHDL's Backend """
    target_name = "vhdl_backend"
    TargetRegister.register_new_target(target_name, lambda _: VHDLBackend)

    code_generation_table = {
        VHDL_Code: vhdl_code_generation_table,
        Gappa_Code: {}
    }

    def __init__(self):
        AbstractBackend.__init__(self)
        Log.report(LOG_BACKEND_INIT, "initializing an instance of VHDL target")
