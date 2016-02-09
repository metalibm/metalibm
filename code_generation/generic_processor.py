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

from ..utility.log_report import *
from .generator_utility import *
from .code_element import *
from .complex_generator import *
from ..core.ml_formats import *
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *
from ..utility.common import Callable


def LibFunctionConstructor(require_header):
    def extend_kwords(kwords, ext_list):
        require_header_arg = [] if ((not "require_header" in kwords) or not kwords["require_header"]) else kwords["require_header"]
        require_header_arg += require_header
        kwords["require_header"] = require_header_arg
        return kwords
        
    return lambda *args, **kwords: FunctionOperator(*args, **extend_kwords(kwords, require_header))

Libm_Function       = LibFunctionConstructor(["math.h"])
Std_Function        = LibFunctionConstructor(["stdlib.h"])
Fenv_Function       = LibFunctionConstructor(["fenv.h"])
ML_Utils_Function   = LibFunctionConstructor(["support_lib/ml_utils.h"])
ML_Multi_Prec_Lib_Function   = LibFunctionConstructor(["support_lib/ml_multi_prec_lib.h"])


## Generate a unary symbol operator for integer to integer conversion
#  @param optree input DAG for the operator generator
#  @return code generation operator for generating optree implementation
def dynamic_integer_conversion(optree):
  in_format = optree.get_input(0).get_precision()
  out_format = optree.get_precision()
  if in_format == out_format:
    return IdentityOperator(force_folding = False, no_parenthesis = True, output_precision = out_format)
  else:
    return SymbolOperator("(%s)" % out_format.get_c_name(), arity = 1, force_folding = False, output_precision = out_format)

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
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Int32, ML_Int32, ML_Int32): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_UInt32, ML_Int32, ML_UInt32, ML_UInt32): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_Int64, ML_Int32, ML_Int64, ML_Int64): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_UInt64, ML_Int32, ML_UInt64, ML_UInt64): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_UInt32, ML_Int32, ML_UInt32, ML_UInt32): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_Binary32, ML_Int32, ML_Binary32, ML_Binary32): TemplateOperator("%s ? %s : %s", arity = 3),
                type_strict_match(ML_Binary64, ML_Int32, ML_Binary64, ML_Binary64): TemplateOperator("%s ? %s : %s", arity = 3),
            },
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
                # multidimentional tables, with integer indices:  first two types should be equals, and all the other should be integers
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
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32], 2, SymbolOperator("||", arity = 2)),
    },
    LogicalAnd: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32], 2, SymbolOperator("&&", arity = 2)),
    },
    LogicalNot: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32], 1, SymbolOperator("!", arity = 1)),
    },
    Negation: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 1, SymbolOperator("-", arity = 1)),
    },
    Addition: {
        None: {
          exclude_doubledouble: build_simplified_operator_generation_nomap([ML_Binary32, ML_Binary64, ML_Int8, ML_UInt8, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("+", arity = 2), cond = fp_std_cond),
          include_doubledouble: { 
            type_strict_match(ML_DoubleDouble, ML_Binary64, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_add_dd_d2", arity = 2, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_Binary64, ML_DoubleDouble): ML_Multi_Prec_Lib_Function("ml_add_dd_d_dd", arity = 2, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_DoubleDouble, ML_Binary64): ML_Multi_Prec_Lib_Function("ml_add_dd_d_dd", arity = 2, arg_map = {0: FO_Arg(1), 1: FO_Arg(0)}, output_precision = ML_DoubleDouble),
            type_strict_match(ML_DoubleDouble, ML_DoubleDouble, ML_DoubleDouble): ML_Multi_Prec_Lib_Function("ml_add_dd_dd2", arity = 2, output_precision = ML_DoubleDouble),
          },
        },
    },
    Subtraction: {
        None: build_simplified_operator_generation([ML_Binary32, ML_Binary64, ML_Int8, ML_UInt8, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("-", arity = 2), cond = fp_std_cond),
    },
    Multiplication: {
        None: {
          exclude_for_mult: build_simplified_operator_generation_nomap([ML_Binary32, ML_Binary64, ML_Int8, ML_UInt8, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("*", arity = 2), cond = exclude_doubledouble),
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
                type_strict_match(None): Fenv_Function("feclearexcept", arg_map = {0: "FE_ALL_EXCEPT"}, arity = 0),
            },
        },
        ExceptionOperation.RaiseException: {
            lambda optree: True: {
                type_strict_match(None): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
                type_strict_match(None,ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1, custom_generate_expr = gen_raise_custom_gen_expr),
                type_strict_match(None,ML_FPE_Type, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
            },
        },
        ExceptionOperation.RaiseReturn: {
            lambda optree: True: {
                type_strict_match(None): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
                type_strict_match(None, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1, custom_generate_expr = gen_raise_custom_gen_expr),
                type_strict_match(None, ML_FPE_Type, ML_FPE_Type): Fenv_Function("feraiseexcept", arity = 1)(SymbolOperator("|", output_precision = ML_UInt32, custom_generate_expr = gen_raise_custom_gen_expr)),
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
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 1, SymbolOperator("-", arity = 1), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Addition: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("+", arity = 2), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Subtraction: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("-", arity = 2), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
    },
    Multiplication: {
        None: build_simplified_operator_generation([ML_Int32, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator("*", arity = 2, no_parenthesis = True), explicit_rounding = True, match_function = type_relax_match, extend_exact = True),
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

class AbstractProcessor: 
    """ base abstract processor """
    pass


def test_is_processor(proc_class):
    """ return whether or not proc_class is a valid and non virtual processor class """
    return issubclass(proc_class, AbstractProcessor) and not proc_class is AbstractProcessor


def get_parent_proc_class_list(proc_class):
    return [parent for parent in proc_class.__bases__ if test_is_processor(parent)]
    

def create_proc_hierarchy(process_list, proc_class_list = []):
    """ create an ordered list of processor hierarchy """
    if process_list == []:
        return proc_class_list
    new_process_list = []
    for proc_class in process_list:
        if proc_class in proc_class_list: 
            continue
        else:
            proc_class_list.append(proc_class)
            new_process_list += get_parent_proc_class_list(proc_class)
    result = create_proc_hierarchy(new_process_list, proc_class_list)
    return result
    
class ML_FullySupported: pass

class GenericProcessor(AbstractProcessor):
    """ Generic class for instruction selection,
        corresponds to a portable C-implementation """

    target_name = "generic"

    # code generation table map
    code_generation_table = {
        C_Code: c_code_generation_table,
        Gappa_Code: gappa_code_generation_table,
    }

    # approximation table map
    approx_table_map = generic_approx_table_map

    def get_target_name(sef):
        return self.target_name

    def __init__(self, *args):
        """ processor class initialization """
        # create ordered list of parent architecture instances
        parent_class_list = get_parent_proc_class_list(self.__class__)
        self.parent_architecture = [parent(*args) for parent in create_proc_hierarchy(parent_class_list, [])]

        # create simplified of operation supported by the processor hierarchy
        self.simplified_rec_op_map = {}
        self.simplified_rec_op_map[C_Code] = self.generate_supported_op_map(language = C_Code)


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, **kwords): #folded = True, language = C_Code, result_var = None):
        """ processor generate expression """
        language = kwords["language"] if "language" in kwords else C_Code
        if self.is_local_supported_operation(optree, language = language):
            local_implementation = self.get_implementation(optree, language)
            return local_implementation.generate_expr(code_generator, code_object, optree, arg_tuple, **kwords)#folded = folded, result_var = result_var)
        else:
            for parent_proc in self.parent_architecture:
                if parent_proc.is_local_supported_operation(optree, language = language):
                    return parent_proc.get_implementation(optree, language).generate_expr(code_generator, code_object, optree, arg_tuple, **kwords)#folded = folded, result_var = result_var)
            # no implementation were found
            Log.report(Log.Verbose, "Tested architecture(s):")
            for parent_proc in self.parent_architecture:
              Log.report(Log.Verbose, "  %s " % parent_proc)

            Log.report(Log.Error, "the following operation is not supported by %s/%s: \n%s" % (self.__class__, language, optree.get_str(depth = 2, display_precision = True, memoization_map = {}))) 

    def generate_supported_op_map(self, language = C_Code, table_getter = lambda self: self.code_generation_table):
        """ generate a map of every operations supported by the processor hierarchy,
            to be used in OptimizationEngine step """
        op_map = {}
        self.generate_local_op_map(language, op_map)
        for parent_proc in self.parent_architecture:
            parent_proc.generate_local_op_map(language, op_map, table_getter = table_getter)
        return op_map


    def generate_local_op_map(self, language = C_Code, op_map = {}, table_getter = lambda self: self.code_generation_table):
        """ generate simplified map of locally supported operations """
        table = table_getter(self)
        local_map = table[language]
        for operation in local_map:
            if not operation in op_map: 
                op_map[operation] = {}
            for specifier in local_map[operation]:
                if not specifier in op_map[operation]: 
                    op_map[operation][specifier] = {}
                for condition in local_map[operation][specifier]:
                    if not condition in op_map[operation][specifier]:
                        op_map[operation][specifier][condition] = {}
                    for interface_format in local_map[operation][specifier][condition]:
                        op_map[operation][specifier][condition][interface_format] = ML_FullySupported
        return op_map
                    

    def get_implementation(self, optree, language = C_Code, table_getter = lambda self: self.code_generation_table):
        """ return <self> implementation of operation performed by <optree> """
        table = table_getter(self)
        op_class, interface, codegen_key = GenericProcessor.get_operation_keys(optree)
        for condition in table[language][op_class][codegen_key]:
            if condition(optree):
                for interface_condition in table[language][op_class][codegen_key][condition]:
                    if interface_condition(*interface, optree = optree):
                        return table[language][op_class][codegen_key][condition][interface_condition]
        return None

    def get_recursive_implementation(self, optree, language = None, table_getter = lambda self: self.code_generation_table):
        """ recursively search for an implementation of optree in the processor class hierarchy """
        if self.is_local_supported_operation(optree, language = language, table_getter = table_getter):
            local_implementation = self.get_implementation(optree, language, table_getter = table_getter)
            return local_implementation
        else:
            for parent_proc in self.parent_architecture:
                if parent_proc.is_local_supported_operation(optree, language = language, table_getter = table_getter):
                    return parent_proc.get_implementation(optree, language, table_getter = table_getter)
            # no implementation were found
            Log.report(Log.Error, "the following operation is not supported by %s: \n%s" % (self.__class__, optree.get_str(depth = 2, display_precision = True))) 
        

    def is_map_supported_operation(self, op_map, optree, language = C_Code, debug = False):
        """ return wheter or not the operation performed by optree has a local implementation """
        op_class, interface, codegen_key = self.get_operation_keys(optree)

        if not language in op_map: 
            # unsupported language
            if debug: Log.Report(Log.Info, "unsupported language for %s" % optree.get_str())
            return False
        else:
            if not op_class in op_map[language]:
                # unsupported operation
                if debug: Log.Report(Log.Info, "unsupported operation class for %s" % optree.get_str())
                return False
            else:
                if not codegen_key in op_map[language][op_class]:
                    if debug: Log.Report(Log.Info, "unsupported codegen key for %s" % optree.get_str())
                    # unsupported codegen key
                    return False
                else:
                    for condition in op_map[language][op_class][codegen_key]:
                        if condition(optree):
                            for interface_condition in op_map[language][op_class][codegen_key][condition]:
                                if interface_condition(*interface, optree = optree): return True
                    # unsupported condition or interface type
                    if debug: 
                      Log.report(Log.Info, "unsupported condition key for %s" % optree.get_str(display_precision = True))
                      for condition in op_map[language][op_class][codegen_key]:
                          if condition(optree):
                            print "verified by condition ", condition
                            for interface_condition in op_map[language][op_class][codegen_key][condition]:
                                print "ic: ", interface_condition.type_tuple, 
                                if interface_condition(*interface, optree = optree): 
                                  print "True"
                                else:
                                  print "False"
                      print op_map[language][op_class][codegen_key].keys()
                    return False
                        

    def is_local_supported_operation(self, optree, language = C_Code, table_getter = lambda self: self.code_generation_table, debug = False):
        """ return whether or not the operation performed by optree has a local implementation """
        table = table_getter(self)
        return self.is_map_supported_operation(table, optree, language, debug = debug)


    def is_supported_operation(self, optree, language = C_Code, debug = False):
        """ return whether or not the operation performed by optree is supported by any level of the processor hierarchy """
        return self.is_map_supported_operation(self.simplified_rec_op_map, optree, language, debug = debug)


    def get_operation_keys(optree):
        """ return code_generation_table key corresponding to the operation performed by <optree> """
        op_class = optree.__class__
        result_type = (optree.get_precision(),)
        arg_type = tuple(arg.get_precision() for arg in optree.inputs)
        interface = result_type + arg_type
        codegen_key = optree.get_codegen_key()
        return op_class, interface, codegen_key


    def get_local_implementation(proc_class, optree, language = C_Code, table_getter = lambda c: c.code_generation_table):
        """ return the implementation provided by <proc_class> of the operation performed by <optree> """
        op_class, interface, codegen_key = proc_class.get_operation_keys(optree)
        table = table_getter(proc_class)
        for condition in table[language][op_class][codegen_key]:
            if condition(optree):
                for interface_condition in table[language][op_class][codegen_key][condition]:
                    if interface_condition(*interface, optree = optree):
                        return table[language][op_class][codegen_key][condition][interface_condition]
        raise Exception()


    # static member function binding
    get_operation_keys          = Callable(get_operation_keys) 
    get_local_implementation    = Callable(get_local_implementation)



if __name__ == "__main__":
    print FunctionOperator("ml_is_nan_or_inff", arity = 1).arg_map
