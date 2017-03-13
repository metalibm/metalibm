# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2014)
# All rights reserved
# created:          Dec 24th, 2013
# last-modified:    Jun  6th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from sollya import S2

from ..utility.log_report import *

from ..core.ml_formats import *
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *

from .generator_utility import *
from .code_element import *
from .complex_generator import *
from .generator_helper import *
from .abstract_backend import *

def LibFunctionConstructor(require_header):
    def extend_kwords(kwords, ext_list):
        require_header_arg = [] if ((not "require_header" in kwords) or not kwords["require_header"]) else kwords["require_header"]
        require_header_arg += require_header
        kwords["require_header"] = require_header_arg
        return kwords

    return lambda *args, **kwords: FunctionOperator(*args, **extend_kwords(kwords, require_header))



## Generate a unary symbol operator for integer to integer conversion
#  @param optree input DAG for the operator generator
#  @return code generation operator for generating optree implementation
def dynamic_integer_conversion(optree):
  in_format = optree.get_input(0).get_precision()
  out_format = optree.get_precision()
  if in_format == out_format:
    return IdentityOperator(force_folding = False, no_parenthesis = True, output_precision = out_format)
  else:
    return SymbolOperator("(%s)" % out_format.get_name(language = C_Code), arity = 1, force_folding = False, output_precision = out_format)

def std_cond(optree):
    # standard condition for operator mapping validity
    return (not optree.get_silent()) and (optree.get_rounding_mode() == ML_GlobalRoundMode or optree.get_rounding_mode() == None)

def exclude_doubledouble(optree):
    return (optree.get_precision() != ML_DoubleDouble)
def include_doubledouble(optree):
    return not exclude_doubledouble(optree)

def exclude_for_mult(optree):
    return (optree.get_precision() != ML_DoubleDouble
     and (optree.get_precision() == optree.get_input(0).get_precision())
     and (optree.get_precision() == optree.get_input(1).get_precision()))
def include_for_mult(optree):
    return not exclude_for_mult(optree)

def fp_std_cond(optree):    
    return True
    #return (not isinstance(optree.get_precision(), ML_FP_Format)) or std_cond(optree)


def gen_raise_custom_gen_expr(self, code_generator, code_object, optree, arg_tuple, **kwords):
    exception_translation = {
        ML_FPE_Underflow: "FE_UNDERFLOW",
        ML_FPE_Overflow: "FE_OVERFLOW",
        ML_FPE_Invalid: "FE_INVALID",
        ML_FPE_Inexact: "FE_INEXACT",
        ML_FPE_DivideByZero: "FE_DIVBYZERO",
    }
    # generating list of arguments
    arg_result = [CodeExpression(exception_translation[arg.get_value()], None) for arg in arg_tuple]
    # assembling parent operator code
    return self.assemble_code(code_generator, code_object, optree, arg_result, **kwords)

def full_mul_modifier(optree):
    """ extend the precision of arguments of a multiplication to get the full result of the multiplication """
    op0 = optree.get_input(0)
    op1 = optree.get_input(1)
    optree_type = optree.get_precision()
    assert(is_std_integer_format(op0.get_precision()) and is_std_integer_format(op1.get_precision()) and is_std_integer_format(optree_type))
    op0_conv = Conversion(op0, precision = optree_type) if optree_type != op0.get_precision() else op0
    op1_conv = Conversion(op1, precision = optree_type) if optree_type != op1.get_precision() else op0
    return Multiplication(op0_conv, op1_conv, precision = optree_type)


## hash map of Comparison specifier -> C symbol relation
c_comp_symbol = {
  Comparison.Equal: "==", 
  Comparison.NotEqual: "!=",
  Comparison.Less: "<",
  Comparison.LessOrEqual: "<=",
  Comparison.GreaterOrEqual: ">=",
  Comparison.Greater: ">"
}

c_code_generation_table = {
    Select: {
        None: {
            lambda optree: True: 
              dict(
                sum(
                  [[(type_strict_match(ML_Int32, bool_precision, ML_Int32, ML_Int32), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_UInt32, bool_precision, ML_UInt32, ML_UInt32), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_Int64, bool_precision, ML_Int64, ML_Int64), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_UInt64, bool_precision, ML_UInt64, ML_UInt64), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_UInt32, bool_precision, ML_UInt32, ML_UInt32), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_Binary32, bool_precision, ML_Binary32, ML_Binary32), TemplateOperator("%s ? %s : %s", arity = 3)),
                   (type_strict_match(ML_Binary64, bool_precision, ML_Binary64, ML_Binary64), TemplateOperator("%s ? %s : %s", arity = 3))
                  ] for bool_precision in [ML_Bool, ML_Int32]],
                  []
                )
              ),
            
        },  
    },
    Abs: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): Libm_Function("fabsf", arity = 1),
                type_strict_match(ML_Binary64, ML_Binary64): Libm_Function("fabs", arity = 1),
                type_strict_match(ML_Int32, ML_Int32): Libm_Function("abs", arity = 1),
                type_strict_match(ML_Int64, ML_Int64): Libm_Function("lfabs", arity = 1),
            }
        }
    },
    TableLoad: {
        None: {
            lambda optree: True: {
                # multidimensional tables, with integer indices:  first two types should be equal, and all the other should be integers
                (lambda *type_tuple,**kwords:
                    len(type_tuple) >= 3 and type_strict_match(type_tuple[0])(type_tuple[1]) and type_std_integer_match(*type_tuple[2:])
                ) : True,
            },
        },
    },
    BitLogicAnd: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Int64, ML_UInt64], 2, SymbolOperator("&", arity = 2)),
    },
    BitLogicOr: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Int64, ML_UInt64], 2, SymbolOperator("|", arity = 2)),
    },
    BitLogicXor: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Int64, ML_UInt64], 2, SymbolOperator("^", arity = 2)),
    },
    BitLogicNegate: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Int64, ML_UInt64], 1, SymbolOperator("~", arity = 1)),
    },
    BitLogicLeftShift: {
        None: {
            lambda optree: True: {
                # shift any integer, as long as all types are integers, and the dest and first arg have the same type
                (lambda dst_type,op0_type,op1_type,**kwords:
                    type_strict_match(dst_type)(op0_type) and type_std_integer_match(op1_type)
                ) : SymbolOperator("<<", arity = 2),
            },
        },  
    },
    BitLogicRightShift: {
        None: {
            lambda optree: True: {
                # shift any integer, as long as all types are integers, and the dest and first arg have the same type
                (lambda dst_type,op0_type,op1_type,**kwords:
                    dst_type == op0_type and is_std_integer_format(op0_type) and is_std_integer_format(op1_type)
                ) : SymbolOperator(">>", arity = 2),
            },
        },  
    },
    LogicalOr: {
        None: build_simplified_operator_generation([ML_Bool, ML_Int32, ML_UInt32], 2, SymbolOperator("||", arity = 2)),
    },
    LogicalAnd: {
        None: build_simplified_operator_generation([ML_Bool, ML_Int32, ML_UInt32], 2, SymbolOperator("&&", arity = 2)),
    },
    LogicalNot: {
        None: build_simplified_operator_generation([ML_Bool, ML_Int32, ML_UInt32], 1, SymbolOperator("!", arity = 1)),
    },
    Negation: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_UInt64, ML_Int64, ML_Binary32, ML_Binary64], 1, SymbolOperator("-", arity = 1)),
    },
    Addition: {
        None: {
          exclude_doubledouble: build_simplified_operator_generation_nomap([ML_Binary32, ML_Binary64, v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("+", arity = 2), cond = fp_std_cond),
          include_doubledouble: { 
            type_strict_match(ML_DoubleDouble, ML_Binary64, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_add_dd_d2", arity = 2, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_Binary64, ML_DoubleDouble): ML_Multi_Prec_Lib_Function("ml_add_dd_d_dd", arity = 2, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_DoubleDouble, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_add_dd_d_dd", arity = 2, arg_map = {0: FO_Arg(1), 1: FO_Arg(0)}, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_DoubleDouble, ML_DoubleDouble): ML_Multi_Prec_Lib_Function("ml_add_dd_dd2", arity = 2, output_precision = ML_DoubleDouble),
          },
        },
    },
    Subtraction: {
        None: build_simplified_operator_generation([ML_Binary32, ML_Binary64, v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("-", arity = 2), cond = fp_std_cond),
    },
    Multiplication: {
        None: {
          exclude_for_mult: build_simplified_operator_generation_nomap([ML_Binary32, ML_Binary64, v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("*", arity = 2), cond = exclude_doubledouble),
          include_for_mult: {
            type_strict_match(ML_DoubleDouble, ML_Binary64, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_mult_dd_d2", arity = 2, output_precision = ML_DoubleDouble),
            type_std_integer_match: ComplexOperator(optree_modifier = full_mul_modifier, backup_operator = SymbolOperator("*", arity = 2)),
        }
      }
    },
    FusedMultiplyAdd: {
        FusedMultiplyAdd.Standard: {
            lambda optree: std_cond(optree): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): Libm_Function("fmaf", arity = 3),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): Libm_Function("fma", arity = 3),
                type_strict_match(ML_DoubleDouble, ML_Binary64, ML_Binary64, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_fma_dd_d3", arity = 3, speed_measure = 66.5),
            },
        },
        FusedMultiplyAdd.Negate: {
            lambda optree: std_cond(optree): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): SymbolOperator("-", arity = 1)(Libm_Function("fmaf", arity = 3, output_precision = ML_Binary32)),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): SymbolOperator("-", arity = 1)(Libm_Function("fma", arity = 3, output_precision = ML_Binary64)),
            },
        },
        FusedMultiplyAdd.SubtractNegate: {
            lambda optree: std_cond(optree): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): Libm_Function("fmaf", arity = 3, output_precision = ML_Binary32)(SymbolOperator("-", arity = 1, output_precision = ML_Binary32)(FO_Arg(0)), FO_Arg(1), FO_Arg(2)),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): Libm_Function("fma", arity = 3, output_precision = ML_Binary64)(SymbolOperator("-", arity = 1, output_precision = ML_Binary64)(FO_Arg(0)), FO_Arg(1), FO_Arg(2)),
            },
        },
        FusedMultiplyAdd.Subtract: {
            lambda optree: std_cond(optree): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): Libm_Function("fmaf", arity = 3, output_precision = ML_Binary32)(FO_Arg(0), FO_Arg(1), SymbolOperator("-", arity = 1, output_precision = ML_Binary32)(FO_Arg(2))),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): Libm_Function("fma", arity = 3, output_precision = ML_Binary64)(FO_Arg(0), FO_Arg(1), SymbolOperator("-", arity = 1, output_precision = ML_Binary64)(FO_Arg(2))),
            },
        },
    },
    Division: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("/", arity = 2)),
    },
    Modulo: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Int64], 2, SymbolOperator("%", arity = 2)),
    },
    Comparison: 
        dict (
          (specifier,
            { 
                lambda _: True: 
                dict(
                  (
                    type_strict_match_list([ML_Int32, ML_Bool], [op_type]),
                    SymbolOperator(c_comp_symbol[specifier], arity = 2)
                  ) for op_type in [ML_Int32, ML_Int64, ML_UInt64, ML_UInt32, ML_Binary32, ML_Binary64]
                ),
                #build_simplified_operator_generation([ML_Int32, ML_Int64, ML_UInt64, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator(">=", arity = 2), result_precision = ML_Int32),
            }) for specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]
    ),
    Test: {
        Test.IsIEEENormalPositive: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_normal_positive_fp32", arity = 1),
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_normal_positive_fp64", arity = 1),
            },
        },
        Test.IsInfOrNaN: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_nan_or_inff", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_nan_or_inf", arity = 1), 
            },
        },
        Test.IsNaN: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_nanf", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_nan", arity = 1), 
            },
        },
        Test.IsSignalingNaN: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_signaling_nanf", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_signaling_nan", arity = 1), 
            },
        },
        Test.IsQuietNaN: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_quiet_nanf", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_quiet_nan", arity = 1), 
            },
        },
        Test.IsSubnormal: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_subnormalf", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_subnormal", arity = 1), 
            },
        },
        Test.IsInfty: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_inff", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_inf", arity = 1), 
            },
        },
        Test.IsPositiveInfty: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_plus_inff", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_plus_inf", arity = 1), 
            },
        },
        Test.IsNegativeInfty: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_minus_inff", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_minus_inf", arity = 1), 
            },
        },
        Test.IsZero: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_zerof", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_zero", arity = 1), 
            },
        },
        Test.IsPositiveZero: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_positivezerof", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_positivezero", arity = 1), 
            },
        },
        Test.IsNegativeZero: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): ML_Utils_Function("ml_is_negativezerof", arity = 1), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): ML_Utils_Function("ml_is_negativezero", arity = 1), 
            },
        },
        Test.CompSign: {
            lambda optree: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32], [ML_Binary32]): ML_Utils_Function("ml_comp_signf", arity = 2), 
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64], [ML_Binary64]): ML_Utils_Function("ml_comp_sign", arity = 2), 
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Binary32): Libm_Function("nearbyintf", arity = 1),
                type_strict_match(ML_Binary32, ML_Binary32): Libm_Function("rintf", arity = 1),
                type_strict_match(ML_Int64, ML_Binary64): Libm_Function("nearbyint", arity = 1),
                type_strict_match(ML_Int32, ML_Binary64): Libm_Function("nearbyint", arity = 1),
                type_strict_match(ML_Binary64, ML_Binary64): Libm_Function("rint", arity = 1),
            },
        },
    },
    Ceil: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): Libm_Function("ceilf", arity = 1),
                type_strict_match(ML_Binary64, ML_Binary64): Libm_Function("ceil", arity = 1),
            },
        },
    },
    Trunc: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): Libm_Function("truncf", arity = 1),
                type_strict_match(ML_Binary64, ML_Binary64): Libm_Function("trunc", arity = 1),
            },
        },
    },
    Floor: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): Libm_Function("floorf", arity = 1),
                type_strict_match(ML_Binary64, ML_Binary64): Libm_Function("floor", arity = 1),
            },
        },
    },
    ExponentInsertion: {
        ExponentInsertion.Default: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Int32): ML_Utils_Function("ml_exp_insertion_fp32", arity = 1), 
                type_strict_match(ML_Binary64, ML_Int32): ML_Utils_Function("ml_exp_insertion_fp64", arity = 1),
            },
        },
        ExponentInsertion.NoOffset: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Int32): ML_Utils_Function("ml_exp_insertion_no_offset_fp32", arity = 1), 
                type_strict_match(ML_Binary64, ML_Int32): ML_Utils_Function("ml_exp_insertion_no_offset_fp64", arity = 1),
            },
        },
    },
    ExponentExtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Binary32): ML_Utils_Function("ml_exp_extraction_dirty_fp32", arity = 1), 
                type_strict_match(ML_Int32, ML_Binary64): ML_Utils_Function("ml_exp_extraction_dirty_fp64", arity = 1), 
            },
        },
    },
    MantissaExtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): ML_Utils_Function("ml_mantissa_extraction_fp32", arity = 1), 
                type_strict_match(ML_Binary64, ML_Binary64): ML_Utils_Function("ml_mantissa_extraction_fp64", arity = 1),
                type_strict_match(ML_Custom_FixedPoint_Format(0,52,False), ML_Binary64): ML_Utils_Function("ml_raw_mantissa_extraction_fp64", arity = 1),
                type_strict_match(ML_Custom_FixedPoint_Format(0,23,False), ML_Binary32): ML_Utils_Function("ml_raw_mantissa_extraction_fp32", arity = 1),
            },
        },
    },
    RawSignExpExtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Binary32): FunctionOperator("ml_raw_sign_exp_extraction_fp32", arity = 1), 
                type_strict_match(ML_Int32, ML_Binary64): FunctionOperator("ml_raw_sign_exp_extraction_fp64", arity = 1), 
            },
        },
    },
    CountLeadingZeros: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_UInt32, ML_UInt32): FunctionOperator("ml_count_leading_zeros_32b", arity = 1), 
                type_strict_match(ML_UInt64, ML_UInt64): FunctionOperator("ml_count_leading_zeros_64b", arity = 1), 
            },
        },
    },
    Conversion: {
        None: {
            lambda optree: True: {
                # implicit conversion from and to any integer,Binary64,Binary32 type
                (lambda dst_type, src_type, **kwords: (is_std_integer_format(dst_type) and is_std_integer_format(src_type)))
                : DynamicOperator(dynamic_function = dynamic_integer_conversion),
                (lambda dst_type,src_type,**kwords:
                    ((is_std_integer_format(dst_type) or is_std_integer_format(src_type)) and (dst_type == ML_Binary64 or dst_type == ML_Binary32 or src_type == ML_Binary64 or src_type == ML_Binary32)) or (dst_type in [ML_Binary32, ML_Binary64] and src_type in [ML_Binary64, ML_Binary32])
                ) :  IdentityOperator(),
            },
        },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary64, ML_Int64) : ML_Utils_Function("double_from_64b_encoding", arity = 1),
                type_strict_match(ML_Binary64, ML_UInt64): ML_Utils_Function("double_from_64b_encoding", arity = 1),
                type_strict_match(ML_Int64, ML_Binary64) : ML_Utils_Function("double_to_64b_encoding", arity = 1),
                type_strict_match(ML_UInt64, ML_Binary64): ML_Utils_Function("double_to_64b_encoding", arity = 1),
                type_strict_match(ML_Binary32, ML_Int32) : ML_Utils_Function("float_from_32b_encoding", arity = 1),
                type_strict_match(ML_Binary32, ML_UInt32): ML_Utils_Function("float_from_32b_encoding", arity = 1),
                type_strict_match(ML_Int32, ML_Binary32) : ML_Utils_Function("float_to_32b_encoding", arity = 1),
                type_strict_match(ML_UInt32, ML_Binary32): ML_Utils_Function("float_to_32b_encoding", arity = 1),
                type_strict_match(ML_UInt64, ML_Binary32): ML_Utils_Function("(uint64_t) float_to_32b_encoding", arity = 1),
                type_strict_match(ML_Binary32, ML_UInt64): ML_Utils_Function("float_from_32b_encoding", arity = 1),
            },
        },
    },
    ExceptionOperation: {
        ExceptionOperation.ClearException: {
            lambda optree: True: {
                type_strict_match(ML_Void): Fenv_Function("feclearexcept", arg_map = {0: "FE_ALL_EXCEPT"}, arity = 0),
            },
        },
        ExceptionOperation.RaiseException: {
            lambda optree: True: {
                type_strict_match(ML_Void): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
                type_strict_match(ML_Void,ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1, custom_generate_expr = gen_raise_custom_gen_expr),
                type_strict_match(ML_Void,ML_FPE_Type, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
            },
        },
        ExceptionOperation.RaiseReturn: {
            lambda optree: True: {
                type_strict_match(ML_Void): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
                type_strict_match(ML_Void, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1, custom_generate_expr = gen_raise_custom_gen_expr),
                type_strict_match(ML_Void, ML_FPE_Type, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
            },
        },
    },
    SpecificOperation: {
        SpecificOperation.Subnormalize: {
            lambda optree: True: {
                type_strict_match(ML_Binary64, ML_DoubleDouble, ML_Int32): FunctionOperator("ml_subnormalize_d_dd_i", arity = 2),
            },
        },
        SpecificOperation.CopySign: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32): ML_Utils_Function("ml_copy_signf", arity = 2),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64): ML_Utils_Function("ml_copy_sign", arity = 2),
            },
        },
    },
    Split: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_DoubleDouble, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_split_dd_d", arity = 1),
            },
        },
    },
    ComponentSelection: {
        ComponentSelection.Hi: {
            lambda optree: True: {
                type_strict_match(ML_Binary64, ML_DoubleDouble): TemplateOperator("%s.hi", arity = 1), 
            },
        },
        ComponentSelection.Lo: {
            lambda optree: True: {
                type_strict_match(ML_Binary64, ML_DoubleDouble): TemplateOperator("%s.lo", arity = 1), 
            },
        },
    },
    FunctionCall: {
        None: {
            lambda optree: True: {
                type_function_match: FunctionObjectOperator(), 
            },
        },
    },
    Dereference: {
      None: {
        lambda optree: True: {
          type_all_match:  SymbolOperator("*", arity = 1),
        },
      },
    },
}



gappa_code_generation_table = {
    Negation: {
        None: build_simplified_operator_generation([ML_Int64, ML_UInt64, ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 1, SymbolOperator("-", arity = 1), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Addition: {
        None: build_simplified_operator_generation([ML_Int64, ML_UInt64, ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("+", arity = 2), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Subtraction: {
        None: build_simplified_operator_generation([ML_Int64, ML_UInt64, ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("-", arity = 2), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Multiplication: {
        None: build_simplified_operator_generation([ML_Int64, ML_UInt64, ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("*", arity = 2, no_parenthesis = True), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Division: {
        None: {
            lambda optree: True: {
                lambda *args, **kwords: True: SymbolOperator("/", arity = 2),
            },
        },
    },
    FusedMultiplyAdd: {
        FusedMultiplyAdd.Standard: {
            lambda optree: not optree.get_commutated(): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2))),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2))),
                type_result_match(ML_Exact): SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2)),
            },
            lambda optree: optree.get_commutated(): {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("+", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1)))),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("+", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1)))),
                type_result_match(ML_Exact): SymbolOperator("+", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True,  force_folding = False)(FO_Arg(0), FO_Arg(1))),
            },
        },
        FusedMultiplyAdd.Negate: {
            lambda optree: not optree.get_commutated(): {
                type_relax_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("-", arity = 1, force_folding = False)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2)))),
                type_relax_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("-", arity = 1, force_folding = False)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2)))),
                type_relax_match(ML_Exact): SymbolOperator("-", arity = 1, force_folding = False)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2))),
            },
        },
        FusedMultiplyAdd.Subtract: {
            lambda optree: not optree.get_commutated(): {
                type_relax_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("-", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2))),
                type_relax_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("-", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2))),
                type_result_match(ML_Exact): SymbolOperator("-", arity = 2, force_folding = False)(SymbolOperator("*", no_parenthesis = True, arity = 2, force_folding = False)(FO_Arg(0), FO_Arg(1)), FO_Arg(2)),
            },
        },
        FusedMultiplyAdd.SubtractNegate: {
            lambda optree: not optree.get_commutated(): {
                type_relax_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("-", arity=1, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1))), FO_Arg(2))),
                type_relax_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("-", arity=1, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1))), FO_Arg(2))),
                type_result_match(ML_Exact): SymbolOperator("+", arity = 2, force_folding = False)(SymbolOperator("-", arity=1, force_folding = False)(SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1))), FO_Arg(2)),
            },
            lambda optree: optree.get_commutated(): {
                type_relax_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): RoundOperator(ML_Binary32)(SymbolOperator("-", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1)))),
                type_relax_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64): RoundOperator(ML_Binary64)(SymbolOperator("-", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1)))),
                type_result_match(ML_Exact): SymbolOperator("-", arity = 2, force_folding = False)(FO_Arg(2), SymbolOperator("*", arity = 2, no_parenthesis = True, force_folding = False)(FO_Arg(0), FO_Arg(1))),
            },
        },
    },
    Ceil: {
        None: {
            lambda optree: True: {
                type_relax_match(ML_Binary32, ML_Binary32): RoundOperator(ML_Int32, direction = ML_RoundTowardPlusInfty), 
                type_relax_match(ML_Binary64, ML_Binary64): RoundOperator(ML_Int64, direction = ML_RoundTowardPlusInfty), 
            },
        },
    },
    Floor: {
        None: {
            lambda optree: True: {
                type_relax_match(ML_Binary32, ML_Binary32): RoundOperator(ML_Int32, direction = ML_RoundTowardMinusInfty), 
                type_relax_match(ML_Binary64, ML_Binary64): RoundOperator(ML_Int64, direction = ML_RoundTowardMinusInfty), 
            },
        },
    },
    Trunc: {
        None: {
            lambda optree: True: {
                type_relax_match(ML_Binary32, ML_Binary32): RoundOperator(ML_Int32, direction = ML_RoundTowardZero), 
                type_relax_match(ML_Binary64, ML_Binary64): RoundOperator(ML_Int64, direction = ML_RoundTowardZero), 
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_relax_match(ML_Binary32, ML_Binary32): RoundOperator(ML_Int32), 
                type_relax_match(ML_Binary64, ML_Binary64): RoundOperator(ML_Int64), 
            },
        },
    },
    Comparison: {
      Comparison.NotEqual: {
        lambda optree: True: {
          type_relax_match(ML_Bool, ML_Exact, ML_Integer): SymbolOperator("<>", arity = 2, force_folding = False),
        },
      },
    },
    Conversion: {
        None: {
            lambda optree: True: {
                type_relax_match(ML_Binary32, ML_Binary64): RoundOperator(ML_Binary32), 
                type_relax_match(ML_Binary64, ML_Binary32): RoundOperator(ML_Binary64), 
            },
        },
    },
}


generic_inv_approx_table = ML_ApproxTable(
    dimensions = [2**7, 1], 
    storage_precision = ML_Binary32,
    init_data = [
        [((1.0 + (t_value / S2**9) ) * S2**-1)] for t_value in 
    [508, 500, 492, 485, 477, 470, 463, 455, 448, 441, 434, 428, 421, 414, 408, 401, 395, 389, 383, 377, 371, 365, 359, 353, 347, 342, 336, 331, 326, 320, 315, 310, 305, 300, 295, 290, 285, 280, 275, 271, 266, 261, 257, 252, 248, 243, 239, 235, 231, 226, 222, 218, 214, 210, 206, 202, 198, 195, 191, 187, 183, 180, 176, 172, 169, 165, 162, 158, 155, 152, 148, 145, 142, 138, 135, 132, 129, 126, 123, 120, 117, 114, 111, 108, 105, 102, 99, 96, 93, 91, 88, 85, 82, 80, 77, 74, 72, 69, 67, 64, 62, 59, 57, 54, 52, 49, 47, 45, 42, 40, 38, 35, 33, 31, 29, 26, 24, 22, 20, 18, 15, 13, 11, 9, 7, 5, 3, 0]
    ]
)

generic_approx_table_map = {
    None: { # language
        SpecificOperation: {
            SpecificOperation.DivisionSeed: {
                lambda optree: True: {
                    type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32): generic_inv_approx_table,
                    type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64): generic_inv_approx_table,
                },
                lambda optree: not optree.get_silent(): {
                    type_strict_match(ML_Binary32, ML_Binary32): generic_inv_approx_table,
                    type_strict_match(ML_Binary64, ML_Binary64): generic_inv_approx_table,
                },
                lambda optree: optree.get_silent(): {
                    type_strict_match(ML_Binary32, ML_Binary32): generic_inv_approx_table,
                    type_strict_match(ML_Binary64, ML_Binary64): generic_inv_approx_table,
                },
            },
            #SpecificOperation.InverseSquareRootSeed: {
            #    lambda optree: True: {
            #        type_strict_match(ML_Binary32, ML_Binary32): invsqrt_approx_table,
            #    },
            #},
        },
    },
}


## Generic C Capable Backend
class GenericProcessor(AbstractBackend):
    """ Generic class for instruction selection,
        corresponds to a portable C-implementation """

    target_name = "generic"
    default_compiler = "gcc"


    # code generation table map
    code_generation_table = {
        C_Code: c_code_generation_table,
        Gappa_Code: gappa_code_generation_table,
    }

    # approximation table map
    approx_table_map = generic_approx_table_map

    ## return the compiler command line program to use to build
    #  test programs
    def get_compiler(self):
      return GenericProcessor.default_compiler

    def get_execution_command(self, test_file):
      return "./%s" % test_file

if __name__ == "__main__":
    print FunctionOperator("ml_is_nan_or_inff", arity = 1).arg_map
