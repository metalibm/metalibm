# -*- coding: utf-8 -*-

###############################################################################
# This file is part of New Metalibm tool
# Copyrights  Nicolas Brunie (2016)
# All rights reserved
# created:          Nov 17th, 2016
# last-modified:    Nov 17th, 2016
#
# author(s):    Nicolas Brunie (nibrunie@gmail.com)
# description:  Implement a basic VHDL backend for hardware description
#               generation
###############################################################################

from ..utility.log_report import *
from .generator_utility import *
from .complex_generator import *
from .code_element import *
from ..core.ml_formats import *
from ..core.ml_hdl_format import *
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *
from ..core.ml_hdl_operations import *
from metalibm_core.core.target import TargetRegister

from metalibm_hw_blocks.rtl_blocks import *

from .abstract_backend import AbstractBackend

from metalibm_core.opt.rtl_fixed_point_utils import (
    largest_format, test_format_equality
)


def exclude_std_logic(optree):
    return not isinstance(optree.get_precision(), ML_StdLogicVectorFormat)


def include_std_logic(optree):
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


def zext_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")
    ext_input = optree.get_input(0)
    ext_size = optree.ext_size
    assert ext_size >= 0
    if ext_size == 0:
        Log.report(Log.Warning, "zext_modifer called with ext_size=0 on {}".format(
            optree.get_str()))
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
        return result

# Operation code generator modifier for Sign Extension


def sext_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    ext_size = optree.ext_size
    ext_input = optree.get_input(0)
    if ext_size == 0:
        Log.report(Log.Warning, "sext_modifer called with ext_size=0 on {}".format(
            optree.get_str()))
        return ext_input
    else:
        ext_precision = ML_StdLogicVectorFormat(
            ext_size + ext_input.get_precision().get_bit_size())
        op_size = ext_input.get_precision().get_bit_size()
        sign_digit = VectorElementSelection(ext_input, Constant(
            op_size - 1, precision=ML_Integer), precision=ML_StdLogic, init_stage=init_stage)
        precision = ML_StdLogicVectorFormat(ext_size)
        return Concatenation(
            Replication(
                sign_digit, Constant(ext_size, precision=ML_Integer),
                precision=precision, init_stage=init_stage
            ),
            ext_input, precision=ext_precision,
            tag=optree.get_tag("dummy") + "_sext",
            init_stage=init_stage
        )


def negation_modifer(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")

    neg_input = optree.get_input(0)
    precision = optree.get_precision()
    return Addition(
        BitLogicNegate(neg_input, precision=precision, init_stage=init_stage),
        Constant(1, precision=ML_StdLogic),
        precision=precision,
        tag=optree.get_tag(),
        init_stage=init_stage
    )


# Optree generation function for MantissaExtraction
def mantissa_extraction_modifier(optree):
    init_stage = optree.attributes.get_dyn_attribute("init_stage")
    op = optree.get_input(0)

    op_precision = op.get_precision().get_base_format()
    exp_prec = ML_StdLogicVectorFormat(op_precision.get_exponent_size())
    field_prec = ML_StdLogicVectorFormat(op_precision.get_field_size())

    exp_op = ExponentExtraction(op, precision=exp_prec, init_stage=init_stage)
    field_op = SubSignalSelection(
        TypeCast(
            op,
            precision=op.get_precision().get_support_format(),
            init_stage=init_stage
        ), 0, op_precision.get_field_size() - 1, precision=field_prec,
        init_stage=init_stage
    )

    implicit_digit = Select(
        Comparison(
            exp_op,
            Constant(
                op_precision.get_zero_exponent_value(),
                precision=exp_prec,
                init_stage=init_stage
            ),
            precision=ML_Bool,
            specifier=Comparison.Equal,
            init_stage=init_stage
        ),
        Constant(0, precision=ML_StdLogic),
        Constant(1, precision=ML_StdLogic),
        precision=ML_StdLogic,
        tag="implicit_digit",
        init_stage=init_stage
    )
    return Concatenation(
        implicit_digit,
        field_op,
        precision=ML_StdLogicVectorFormat(op_precision.get_mantissa_size()),
        tag=optree.get_tag(),
        debug=optree.get_debug(),
        init_stage=init_stage
    )


def truncate_generator(optree):
    truncate_input = optree.get_input(0)
    result_size = optree.get_precision().get_bit_size()
    return TemplateOperator("%%s(%d downto 0)" % (result_size - 1), arity=1)


def conversion_generator(optree):
    output_size = optree.get_precision().get_bit_size()
    return TemplateOperator("conv_std_logic_vector(%s, {output_size})".format(output_size=output_size), arity=1)


# dynamic operator helper for Shift operations
#  @param operator string name of the operation
def shift_generator(operator, optree):
    width = optree.get_precision().get_bit_size()
    return TemplateOperator("conv_std_logic_vector({}(unsigned(%s), unsigned(%s)), {})".format(operator, width), arity=2, force_folding=True)


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
    sign_input = optree.get_input(0)
    inf_index = optree.get_inf_index()
    sup_index = optree.get_sup_index()
    range_direction = "to" if (inf_index == sup_index) else "downto"
    return TemplateOperator(
        "%s({sup_index} {direction} {inf_index})".format(
            inf_index=inf_index, direction=range_direction,
            sup_index=sup_index
        ), arity=1, force_folding=True
    )


# @p optree is a conversion node which should be modified
def fixed_conversion_modifier(optree):
    arg = optree.get_input(0)
    conv_precision = optree.get_precision()
    arg_precision = arg.get_precision()
    assert is_fixed_point(conv_precision)
    assert is_fixed_point(arg_precision)
    assert (conv_precision.get_signed() and arg_precision.get_signed()) or not arg_precision.get_signed()
    result = arg
    raw_arg_precision = ML_StdLogicVectorFormat(
        arg_precision.get_bit_size()
    )
    result_raw = TypeCast(
        result,
        precision=raw_arg_precision
    )
    signed = conv_precision.get_signed()
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
    result = TypeCast(
        result_raw,
        precision=conv_precision
    )
    return result

def fixed_shift_modifier(optree):
    """ legalize a shift node on fixed-point operation tree

        Args:
            optree (ML_Operation): operation node input
        Returns:
            legalize node
    """
    shift_input = optree.get_input(0)
    out_precision = optree.get_precision()
    shift_amount = optree.get_input(1)
    converted_input = Conversion(
        shift_input,
        precision = optree.get_precision()
    )
    # check precision equality
    casted_format = ML_StdLogicVectorFormat(out_precision.get_bit_size())
    casted_input = TypeCast(
        converted_input,
        precision = casted_format
    )
    # inserting shift
    casted_shift = BitLogicRightShift(
        casted_input,
        shift_amount,
        precision = casted_format
    )
    # casting back
    fixed_result = TypeCast(
        casted_shift,
        precision = out_precision
    )
    return fixed_result


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
        precision=lhs_precision.get_support_format()
    )
    rhs = TypeCast(
        rhs,
        precision=rhs_precision.get_support_format()
    )
    lhs = SignCast(
        lhs,
        specifier=SignCast.Signed if lhs_precision.get_signed() else
        SignCast.Unsigned,
        precision=lhs.get_precision()
    )
    rhs = SignCast(
        rhs,
        specifier=SignCast.Signed if rhs_precision.get_signed() else
        SignCast.Unsigned,
        precision=rhs.get_precision()
    )
    # we must keep every initial properties of the Comparison node
    # except the operand nodes
    return optree.copy(
        copy_map={
            optree.get_input(0): lhs,
            optree.get_input(1): rhs
        }
    )

# adapt a fixed-optree @p raw_result assumimg fixed format
#  with @p integer_size and @p frac_size
# to match format of @optree


def adapt_fixed_optree(raw_optree, (integer_size, frac_size), optree):
    # extracting params
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
        tag=optree.get_tag(),
        init_stage=init_stage,
        precision=optree_prec,
    )
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
    # TODO: This assume integer_size is at least 0 (no negative
    # (frac end before fixed point) accepted
    result_frac_size = max(lhs_prec.get_frac_size(), rhs_prec.get_frac_size())
    result_integer_size = max(
        lhs_prec.get_integer_size(),
        rhs_prec.get_integer_size()
    ) + 1
    lhs_casted = TypeCast(
        lhs,
        precision=ML_StdLogicVectorFormat(lhs_prec.get_bit_size()),
        init_stage=init_stage
    )
    rhs_casted = TypeCast(
        rhs,
        precision=ML_StdLogicVectorFormat(rhs_prec.get_bit_size()),
        init_stage=init_stage
    )

    lhs_ext = (sext if lhs_prec.get_signed() else zext)(
        rzext(lhs_casted, result_frac_size - lhs_prec.get_frac_size()),
        result_integer_size - lhs_prec.get_integer_size()
    )
    lhs_ext = SignCast(
        lhs_ext, precision=lhs_ext.get_precision(),
        specifier=SignCast.Signed if lhs_prec.get_signed() else SignCast.Unsigned
    )

    rhs_ext = (sext if rhs_prec.get_signed() else zext)(
        rzext(rhs_casted, result_frac_size - rhs_prec.get_frac_size()),
        result_integer_size - rhs_prec.get_integer_size()
    )
    rhs_ext = SignCast(
        rhs_ext, precision=rhs_ext.get_precision(),
        specifier=SignCast.Signed if rhs_prec.get_signed() else SignCast.Unsigned
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
    optree_prec = optree.get_precision().get_base_format()
    # max, optree_prec.get_frac_size())
    result_frac_size = (lhs_prec.get_frac_size() + rhs_prec.get_frac_size())
    # max, optree_prec.get_integer_size())
    result_integer_size = (lhs_prec.get_integer_size() +
                           rhs_prec.get_integer_size())
    #assert optree_prec.get_frac_size() >= result_frac_size
    #assert optree_prec.get_integer_size() >= result_integer_size
    lhs_casted = TypeCast(lhs, precision=ML_StdLogicVectorFormat(
        lhs_prec.get_bit_size()), init_stage=init_stage)
    lhs_casted = SignCast(lhs_casted, precision=lhs_casted.get_precision(
    ), specifier=SignCast.Signed if lhs_prec.get_signed() else SignCast.Unsigned)
    rhs_casted = TypeCast(rhs, precision=ML_StdLogicVectorFormat(
        rhs_prec.get_bit_size()), init_stage=init_stage)
    rhs_casted = SignCast(rhs_casted, precision=rhs_casted.get_precision(
    ), specifier=SignCast.Signed if rhs_prec.get_signed() else SignCast.Unsigned)

    mult_prec = ML_StdLogicVectorFormat(result_frac_size + result_integer_size)
    print "Multiplication {}: {} x {} = {} bits".format(
        optree.get_tag(),
        lhs_casted.get_precision().get_bit_size(),
        rhs_casted.get_precision().get_bit_size(),
        mult_prec.get_bit_size()
    )
    raw_result = Multiplication(
        lhs_casted,
        rhs_casted,
        precision=mult_prec,
        tag=optree.get_tag(),
        init_stage=init_stage
    )
    # adapting raw result to output format
    return adapt_fixed_optree(raw_result, (result_integer_size, result_frac_size), optree)


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

# class Match custom std logic vector format
MCSTDLOGICV = TCM(ML_StdLogicVectorFormat)

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

vhdl_code_generation_table = {
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
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, FSM(ML_StdLogic)):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_StdLogic), MCSTDLOGICV):
                SymbolOperator("+", arity=2, force_folding=True),
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, FSM(ML_StdLogic)):
                SymbolOperator("+", arity=2, force_folding=True),
            },
            # fallback
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint):
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
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):  SymbolOperator("-", arity=2, force_folding=True),
            },
            # fallback
            lambda _: True: {
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint): ComplexOperator(optree_modifier=fixed_point_sub_modifier),
            }
        }
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV): SymbolOperator("*", arity=2, force_folding=True),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint): ComplexOperator(optree_modifier=fixed_point_mul_modifier),
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
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): ComplexOperator(optree_modifier=negation_modifer),
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
                  type_custom_match(FSM(ML_Bool), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)):
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
              #build_simplified_operator_generation([ML_Int32, ML_Int64, ML_UInt64, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator(">=", arity = 2), result_precision = ML_Int32),
          }) for specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]]
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
                               arity=1, force_folding=True),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary32)):
                SymbolOperator("(30 downto 23)", lspace="", inverse=True, \
                               arity=1, force_folding=True),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary16)):
                SymbolOperator("(14 downto 10)", lspace="", inverse=True, \
                               arity=1, force_folding=True),
            },
        },
    },
    ZeroExt: {
        None: {
            lambda _: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): ComplexOperator(optree_modifier=zext_modifier),
                type_custom_match(MCSTDLOGICV, FSM(ML_StdLogic)): TemplateOperatorFormat("(0 => {0}, others => '0')", arity=1, force_folding=True),
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

                type_custom_match(FSM(ML_String), FSM(ML_String), FSM(ML_String)):
                SymbolOperator("&", arity=2, force_folding=False),
            },
        },
    },
    VectorElementSelection: {
        None: {
            # make sure index accessor is a Constant (or fallback to C implementation)
            lambda optree: True:  {
                type_custom_match(FSM(ML_StdLogic), TCM(ML_StdLogicVectorFormat), type_all_match): TemplateOperator("%s(%s)", arity=2),
            },
        },
    },
    Replication: {
        None: {
            lambda optree: True:  {
                type_custom_match(FSM(ML_StdLogic), FSM(ML_StdLogic)): IdentityOperator(),
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), FSM(ML_Integer)): TemplateOperatorFormat("(0 to {1} - 1  => {0:s})", arity=2),
            },
        },
    },
    Conversion: {
        None: {
            lambda optree: True: {
                type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Integer)):
                DynamicOperator(conversion_generator),
                type_strict_match(ML_StdLogic, ML_Bool):
                ComplexOperator(
                    optree_modifier=conversion_from_bool_generator),
                type_custom_match(FSM(ML_String), TCM(ML_StdLogicVectorFormat)):
                FunctionOperator("to_hstring", arity=1, force_folding=False),
                # fixed-point conversion support
                type_custom_match(MCFixedPoint, MCFixedPoint):
                ComplexOperator(optree_modifier=fixed_conversion_modifier),
            }
        },
    },
    MantissaExtraction: {
        None: {
            lambda optree: True: {
                # TemplateOperator("%s(22 downto 0)", arity = 1),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)): ComplexOperator(optree_modifier=mantissa_extraction_modifier),
                # TemplateOperator("%s(22 downto 0)", arity = 1),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)): ComplexOperator(optree_modifier=mantissa_extraction_modifier),
                # TemplateOperator("%s(22 downto 0)", arity = 1),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)): ComplexOperator(optree_modifier=mantissa_extraction_modifier),
            },
        },
    },
    CopySign: {
        None: {
            lambda optree: True: {
                type_custom_match(FSM(ML_StdLogic), ML_Binary16): TemplateOperator("%s(15)", arity=1),
                type_custom_match(FSM(ML_StdLogic), ML_Binary32): TemplateOperator("%s(31)", arity=1),
                type_custom_match(FSM(ML_StdLogic), ML_Binary64): TemplateOperator("%s(63)", arity=1),
                type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV): DynamicOperator(copy_sign_generator),
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
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("signed", arity=1, force_folding=False),
            },
        },
        SignCast.Unsigned: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("unsigned", arity=1, force_folding=False),
            },
        },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Binary16), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision=ML_Binary16, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary16), FSM(ML_Binary16)): IdentityOperator(output_precision=ML_Binary16, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)): IdentityOperator(no_parenthesis=True),

                type_custom_match(FSM(ML_Binary32), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision=ML_Binary32, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary32), FSM(ML_Binary32)): IdentityOperator(output_precision=ML_Binary32, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)): IdentityOperator(no_parenthesis=True),

                type_custom_match(FSM(ML_Binary64), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision=ML_Binary64, no_parenthesis=True),
                type_custom_match(FSM(ML_Binary64), FSM(ML_Binary64)): IdentityOperator(output_precision=ML_Binary64, no_parenthesis=True),
                type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)): IdentityOperator(no_parenthesis=True),

                type_custom_match(MCSTDLOGICV, MCFixedPoint): IdentityOperator(no_parenthesis=True),
                type_custom_match(MCFixedPoint, MCSTDLOGICV): IdentityOperator(no_parenthesis=True),
            },
        },
    },
    BitLogicRightShift: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):
                    DynamicOperator(lambda optree: shift_generator("shr", optree)),
                type_custom_match(MCFixedPoint, MCFixedPoint, MCSTDLOGICV):
                    ComplexOperator(optree_modifier = fixed_shift_modifier),
            },
        },
    },
    BitLogicLeftShift: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):
                DynamicOperator(lambda optree: shift_generator("shl", optree)),
            },
        },
    },
    CountLeadingZeros: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("count_leading_zeros", arity=1),
            },
        },
    },
    SpecificOperation: {
        SpecificOperation.CopySign: {
            lambda optree: True: {
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary16)): TemplateOperator("%s(15)", arity=1),
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary32)): TemplateOperator("%s(31)", arity=1),
                type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary64)): TemplateOperator("%s(63)", arity=1),
                type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV): DynamicOperator(copy_sign_generator),
            },
        },
    },
    SubSignalSelection: {
        None: {
            lambda optree: True: {
                type_custom_match(MCSTDLOGICV, MCSTDLOGICV): DynamicOperator(sub_signal_generator),
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
                    void_function=True
                ),
            }
        },
    },
}


class FormalBackend(AbstractBackend):
    """ description of VHDL's Backend """
    target_name = "formal_backend"
    TargetRegister.register_new_target(target_name, lambda _: FormalBackend)

    code_generation_table = {
        VHDL_Code: formal_generation_table,
        C_Code: formal_generation_table,
        Gappa_Code: {}
    }

    def __init__(self):
        AbstractBackend.__init__(self)
        print "initializing Formal target"


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
        print "initializing VHDL target"
