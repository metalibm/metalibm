# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Oct 6th, 2015
# last-modified:    
#
# description: implement a fixed point backend for Metalibm
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import *
from .generator_utility import SymbolOperator, FunctionOperator, TemplateOperator, C_Code, Gappa_Code, build_simplified_operator_generation, IdentityOperator, FO_Arg, RoundOperator, type_strict_match, type_relax_match, type_result_match, type_function_match, FunctionObjectOperator, type_all_match, build_simplified_operator_generation_nomap
from .complex_generator import *
from ..core.ml_formats import *
from ..core.ml_operations import *
from ..utility.common import Callable
from .generic_processor import GenericProcessor

from metalibm_core.core.target import TargetRegister


def get_fixed_support_format(precision):
  """ return the ML's integer format to contains
      the fixed-point format precision """
  format_map = {
    # signed
    True: {
      8: ML_Int8, 
      16: ML_Int16, 
      32: ML_Int32, 
      64: ML_Int64, 
      128: ML_Int128, 
    },
    # unsigned
    False: {
      8: ML_UInt8, 
      16: ML_UInt16, 
      32: ML_UInt32, 
      64: ML_UInt64, 
      128: ML_UInt128, 
    },
  }
  return format_map[precision.get_signed()][precision.get_c_bit_size()]

def add_modifier(optree):
  """ modify addition optree to be mapped on standard integer format 
      every operand is assumed to be in fixed-point precision """
  op0, op1 = optree.get_inputs()
  op0_format = op0.get_precision()
  op1_format = op1.get_precision()
  optree_format = optree.get_precision()

  # make sure formats are as assumed
  assert (isinstance(op0_format, ML_Fixed_Format) and isinstance(op1_format, ML_Fixed_Format) and isinstance(optree_format, ML_Fixed_Format)), "operands format must be fixed-point in add_modifier"

  # result format frac_size
  rf_fs = max(min(op0_format.get_frac_size(), op1_format.get_frac_size()), optree.get_precision().get_frac_size())
  # result format integer size
  rf_is = optree_format.get_integer_size()

  tmp_format = ML_Custom_FixedPoint_Format(rf_is, rf_fs)
  support_format = get_fixed_support_format(tmp_format) 
  op0_conv = TypeCast(Conversion(op0, precision = tmp_format), precision = support_format)
  op1_conv = TypeCast(Conversion(op1, precision = tmp_format), precision = support_format)

  tmp_result = TypeCast(Addition(op0_conv, op1_conv, precision = support_format), precision = tmp_format)
  return Conversion(tmp_result, precision = optree.get_precision())


def CI(value):
  return Constant(value, precision = ML_UInt32)

def conv_modifier(optree):
  """ lower the Conversion optree to std integer formats """
  op0 = optree.get_input(0)

  in_format = op0.get_precision()
  out_format = optree.get_precision()

  # support format
  in_sformat = get_fixed_support_format(in_format)
  out_sformat = get_fixed_support_format(out_format)

  # conversion when the output format is large than the input format
  if out_sformat.get_bit_size() >= in_sformat.get_bit_size():
    in_ext = Conversion(TypeCast(op0, precision = in_sformat), precision = out_sformat)
    shift = out_format.get_frac_size() - in_format.get_frac_size()
    if shift > 0:
      return TypeCast(BitLogicLeftShift(in_ext, CI(shift), precision = out_sformat), precision = out_format)
    elif shift < 0:
      return TypeCast(BitLogicRightShift(in_ext, CI(-shift), precision = out_sformat), precision = out_format)
    else:
      return TypeCast(in_ext, precision = out_format)
  else:
    in_s = TypeCast(op0, precision = in_sformat)
    shift = out_format.get_frac_size() - in_format.get_frac_size()
    if shift > 0:
      return TypeCast(Conversion(BitLogicLeftShift(in_s, CI(shift), precision = in_sformat), precision = out_sformat), precision = out_format)
    elif shift < 0:
      return TypeCast(Conversion(BitLogicRightShift(in_s, CI(-shift), precision = in_sformat), precision = out_sformat), precision = out_format)
    else:
      return TypeCast(Conversion(in_s, precision = out_sformat), precision = out_format)


def round_down_check(optree):
  return optree.get_rounding_mode() in [None, ML_RoundTowardMinusInfty] 


# class Match custom fixed point format
MCFIPF = TCM(ML_Custom_FixedPoint_Format)

fixed_c_code_generation_table = {
  Addition: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, MCFIPF, MCFIPF) : ComplexOperator(optree_modifier = add_modifier), 
      },
    },
  },
  Conversion: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, MCFIPF) : ComplexOperator(optree_modifier = conv_modifier), 
      },
    },
  },
  TypeCast: {
      None: {
          lambda optree: True: {
              type_custom_match(MCFIPF, FSM(ML_Int64)):    IdentityOperator(),
              type_custom_match(MCFIPF, FSM(ML_Int32)):    IdentityOperator(),
              type_custom_match(MCFIPF, FSM(ML_UInt64)):   IdentityOperator(),
              type_custom_match(MCFIPF, FSM(ML_UInt32)):   IdentityOperator(),
              type_custom_match(FSM(ML_Int32), MCFIPF):    IdentityOperator(),
              type_custom_match(FSM(ML_UInt32), MCFIPF):    IdentityOperator(),
              type_custom_match(FSM(ML_Int64), MCFIPF):    IdentityOperator(),
              type_custom_match(FSM(ML_UInt64), MCFIPF):    IdentityOperator(),
          },
      },
  },
}

class FixedPointBasckend(GenericProcessor):
  target_name = "fixed_point"
  TargetRegister.register_new_target(target_name, lambda _: FixedPointBasckend)

  code_generation_table = {
    C_Code: fixed_c_code_generation_table
  }
