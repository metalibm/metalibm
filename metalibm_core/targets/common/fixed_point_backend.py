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
# All rights reserved
# created:          Oct  6th, 2015
# last-modified:    Mar  7th, 2018
#
# description: implement a fixed point backend for Metalibm
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.utility.log_report import *
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.complex_generator import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import *
from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.core.target import UniqueTargetDecorator

from metalibm_core.opt.opt_utils import forward_attributes


def fixed_modifier(optree, op_class = Addition):
  """ modify addition optree to be mapped on standard integer format 
      every operand is assumed to be in fixed-point precision """
  op0, op1 = optree.get_inputs()
  op0_format = op0.get_precision()
  op1_format = op1.get_precision()
  optree_format = optree.get_precision()

  # make sure formats are as assumed
  assert (isinstance(op0_format, ML_Fixed_Format) and isinstance(op1_format, ML_Fixed_Format) and isinstance(optree_format, ML_Fixed_Format)), "operands format must be fixed-point in fixed_modifier"

  # result format frac_size
  rf_fs = max(min(op0_format.get_frac_size(), op1_format.get_frac_size()), optree.get_precision().get_frac_size())
  # result format integer size
  rf_is = optree_format.get_integer_size()

  tmp_format = ML_Custom_FixedPoint_Format(rf_is, rf_fs, signed = optree_format.get_signed())
  support_format = get_std_integer_support_format(tmp_format) 
  op0_conv = TypeCast(Conversion(op0, precision = tmp_format), precision = support_format)
  op1_conv = TypeCast(Conversion(op1, precision = tmp_format), precision = support_format)

  tmp_result = TypeCast(op_class(op0_conv, op1_conv, precision = support_format, tag = optree.get_tag()), precision = tmp_format, tag = optree.get_tag())
  return Conversion(tmp_result, precision = optree.get_precision())


def add_modifier(optree):
  return fixed_modifier(optree, op_class = Addition)

def sub_modifier(optree):
  return fixed_modifier(optree, op_class = Subtraction)

def mul_modifier(optree):
  """ modify addition optree to be mapped on standard integer format 
      every operand is assumed to be in fixed-point precision """
  op0, op1 = optree.get_inputs()
  op0_format = op0.get_precision()
  op1_format = op1.get_precision()
  optree_format = optree.get_precision()

  # make sure formats are as assumed
  assert (isinstance(op0_format, ML_Fixed_Format) and isinstance(op1_format, ML_Fixed_Format) and isinstance(optree_format, ML_Fixed_Format)), "operands format must be fixed-point in add_modifier"
  
  tmp_format = ML_Custom_FixedPoint_Format(
    min(optree_format.get_integer_size(), op0_format.get_integer_size() + op1_format.get_integer_size()),
    op0_format.get_frac_size() + op1_format.get_frac_size(),
    op0_format.get_signed() or op1_format.get_signed()
  )

  Log.report(Log.Verbose, "mul_modifier tmp_format=%s" % tmp_format)
  
  op0_conv = TypeCast(op0, precision = get_std_integer_support_format(op0_format))
  op1_conv = TypeCast(op1, precision = get_std_integer_support_format(op1_format))
  tmp_conv = Multiplication(op0_conv, op1_conv, precision = get_std_integer_support_format(tmp_format), tag = optree.get_tag())
  tmp = TypeCast(tmp_conv, precision = tmp_format)
  result = Conversion(tmp, precision = optree_format)
  Log.report(Log.Verbose, "result of mul_modifier on\n%s IS\n %s" % (optree.get_str(depth = 2, display_precision = True, memoization_map = {}), result.get_str(depth = 4, display_precision = True)))

  return result


def CI(value):
  return Constant(value, precision = ML_UInt32)

def conv_modifier(optree):
  """ lower the Conversion optree to std integer formats """
  op0 = optree.get_input(0)

  in_format = op0.get_precision()
  out_format = optree.get_precision()

  # support format
  in_sformat = get_std_integer_support_format(in_format)
  out_sformat = get_std_integer_support_format(out_format)

  result = None

  Log.report(Log.Verbose, "in_format is %s | in_sformat is %s" % (in_format, in_sformat))
  Log.report(Log.Verbose, "out_format is %s | out_sformat is %s" % (out_format, out_sformat))
  # conversion when the output format is large than the input format
  if in_format == out_format:
    result = optree
  if out_sformat.get_bit_size() >= in_sformat.get_bit_size():
    in_ext = Conversion(TypeCast(op0, precision = in_sformat), precision = out_sformat)
    shift = out_format.get_frac_size() - in_format.get_frac_size()
    if shift > 0:
      result = TypeCast(BitLogicLeftShift(in_ext, CI(shift), precision = out_sformat), precision = out_format)
    elif shift < 0:
      result =  TypeCast(BitLogicRightShift(in_ext, CI(-shift), precision = out_sformat), precision = out_format)
    else:
      result = TypeCast(in_ext, precision = out_format)
  else:
    in_s = TypeCast(op0, precision = in_sformat)
    shift = out_format.get_frac_size() - in_format.get_frac_size()
    if shift > 0:
      result = TypeCast(Conversion(BitLogicLeftShift(in_s, CI(shift), precision = in_sformat), precision = out_sformat), precision = out_format)
    elif shift < 0:
      result = TypeCast(Conversion(BitLogicRightShift(in_s, CI(-shift), precision = in_sformat), precision = out_sformat), precision = out_format)
    else:
      result = TypeCast(Conversion(in_s, precision = out_sformat), precision = out_format)

  result.set_tag(optree.get_tag())
  Log.report(Log.Verbose, "result of conv_modifier on \n %s IS: \n  %s " % (optree.get_str(display_precision = True, depth = 3, memoization_map = {}), result.get_str(display_precision = True, depth = 4)))
  return result


## Lower the conversion from a floating-point input to a fixed-point result
#  @param optree Conversion to be lowered
#  @return lowered Operation DAG
def conv_from_fp_modifier(optree):
  """ lower the Conversion optree to from floaint-point to std integer formats """
  op = optree.get_input(0)

  in_format = op.get_precision()
  out_format = optree.get_precision()

  # support format
  out_sformat = get_std_integer_support_format(out_format)

  scaling_factor = Constant(S2**out_format.get_frac_size(), precision = in_format)
  scaling_input = NearestInteger(Multiplication(scaling_factor, op, precision = in_format), precision = out_sformat)
  result = TypeCast(scaling_input, precision = out_format)

  Log.report(Log.Verbose, "result of conv_from_fp_modifier on %s IS\n %s " % (optree.get_str(display_precision = True, depth = 2, memoization_map = {}), result.get_str(display_precision = True, depth = 3)))
  return result



def legalize_fixed_point_comparison(optree):
    """ Legalization a fixed-point comparison into integer operation """
    predicate = optree.specifier
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)

    if lhs.get_precision() == rhs.get_precision():
        # if format match, a single cast will suffice
        # get_std_integer_support_format should managed signedness properly
        support_format = get_std_integer_support_format(lhs.get_precision())
        new_node = Comparison(
            TypeCast(lhs, precision=support_format),
            TypeCast(rhs, precision=support_format),
            specifier=predicate
        )
        forward_attributes(optree, new_node)
        return new_node
    else:
        encapsulating_integer_size = max(
            lhs.get_precision().get_integer_size(),
            rhs.get_precision().get_integer_size()
        )
        encapsulating_frac_size = max(
            lhs.get_precision().get_frac_size(),
            rhs.get_precision().get_frac_size()
        )
        encapsulating_signedness = lhs.get_signed() or rhs.get_signed()
        encapsulating_format = ML_Custom_FixedPoint_Format(
            encapsulating_integer_size,
            encapsulating_frac_size,
            signed=encapsulating_signedness
        )
        if encapsulating_format != lhs.get_precision():
            lhs = Conversion(lhs, precision=encapsulating_format)
        if encapsulating_format != rhs.get_precision():
            rhs = Conversion(rhs, precision=encapsulating_format)

        new_node = Comparison(
            lhs,
            rhs,
            specifier=predicate
        )
        forward_attributes(optree, new_node)
        return new_node

## Lower the Conversion optree which converts a fixed-point value
#  into a floating-point results
#  @return the lowered conversion
#  @param optree Conversion optree to be converted
def conv_fixed_to_fp_modifier(optree):
  op = optree.get_input(0)
  in_support_format = get_std_integer_support_format(op.get_precision())

  casted_input = TypeCast(op, precision = in_support_format)
  converted_input = Conversion(casted_input, precision = optree.get_precision())
  scaled_factor = Constant(S2**-op.get_precision().get_frac_size(), precision = optree.get_precision())
  scaled_result = Multiplication(converted_input, scaled_factor, precision = optree.get_precision())

  Log.report(Log.Verbose, "result of conv_fixed_to_fp_modifier on %s IS\n %s " % (optree.get_str(display_precision = True, depth = 2, memoization_map = {}), scaled_result.get_str(display_precision = True, depth = 3)))

  return scaled_result


def round_down_check(optree):
  return optree.get_rounding_mode() in [None, ML_RoundTowardMinusInfty] 


def fixed_cast_modifier(optree):
  """ lower the Conversion optree to std integer formats """
  op0 = optree.get_input(0)

  in_format = op0.get_precision()
  out_format = optree.get_precision()
  # support format
  in_sformat = get_std_integer_support_format(in_format)
  out_sformat = get_std_integer_support_format(out_format)
  if out_sformat == in_sformat:
    return op0
  else:
    return None

FixedCastOperator = ComplexOperator(optree_modifier = fixed_cast_modifier, backup_operator = IdentityOperator(force_folding = False, no_parenthesis = True))


def is_cast_simplification_valid(dst_type, src_type, **kwords):
  if not isinstance(dst_type, ML_Fixed_Format) or not isinstance(src_type, ML_Fixed_Format): return False
  src_support_type = get_std_integer_support_format(src_type)
  dst_support_type = get_std_integer_support_format(dst_type)

  #return src_support_type.get_bit_size() == dst_support_type.get_bite_size()
  return dst_type.get_c_bit_size() == src_type.get_c_bit_size()

def is_cast_simplification_invalid(dst_type, src_type, **kwords):
  return not is_cast_simplification_valid(dst_type, src_type, **kwords)

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
  Subtraction: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, MCFIPF, MCFIPF) : ComplexOperator(optree_modifier = sub_modifier), 
      },
    },
  },
  Multiplication: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, MCFIPF, MCFIPF) : ComplexOperator(optree_modifier = mul_modifier),
      },
    },
  },
  Conversion: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, MCFIPF) : ComplexOperator(optree_modifier = conv_modifier), 
        type_custom_match(MCFIPF, FSM(ML_Binary32)) : ComplexOperator(optree_modifier = conv_from_fp_modifier), 
        type_custom_match(FSM(ML_Binary32), MCFIPF) : ComplexOperator(optree_modifier = conv_fixed_to_fp_modifier),

        type_custom_match(MCFIPF, FSM(ML_Binary64)):
            ComplexOperator(optree_modifier=conv_from_fp_modifier),
        type_custom_match(FSM(ML_Binary64), MCFIPF):
            ComplexOperator(optree_modifier=conv_fixed_to_fp_modifier),
      },

    },
  },
  TypeCast: {
      None: {
          lambda optree: isinstance(optree.get_precision(), ML_Fixed_Format) and isinstance(optree.get_input(0).get_precision(), ML_Fixed_Format): {
              # type cast between two FixedPoint is the same as TypeCast between the support type
              is_cast_simplification_valid: IdentityOperator(force_folding = False, no_parenthesis = True),
              #is_cast_simplification_invalid: IdentityOperator(),
          },
      },
  },
  Comparison: {
    Comparison.NotEqual: {
        lambda optree: True: {
            type_custom_match(FSM(ML_Bool), MCFIPF, MCFIPF):
                ComplexOperator(optree_modifier=legalize_fixed_point_comparison),
        },
    },
    Comparison.GreaterOrEqual: {
        lambda optree: True: {
            type_custom_match(FSM(ML_Bool), MCFIPF, MCFIPF):
                ComplexOperator(optree_modifier=legalize_fixed_point_comparison),
        },
    },
    Comparison.LessOrEqual: {
        lambda optree: True: {
            type_custom_match(FSM(ML_Bool), MCFIPF, MCFIPF):
                ComplexOperator(optree_modifier=legalize_fixed_point_comparison),
        },
    },
  },
}

fixed_gappa_code_generation_table = {
  Conversion: {
    None: {
      round_down_check: {
        type_custom_match(MCFIPF, FSM(ML_Binary64)) : RoundOperator(ML_Int32), 
      },
    },
  },
}

@UniqueTargetDecorator
class FixedPointBackend(GenericProcessor):
  target_name = "fixed_point"

  code_generation_table = {
    C_Code: fixed_c_code_generation_table, 
    Gappa_Code: fixed_gappa_code_generation_table,
  }

# debug message
Log.report(LOG_BACKEND_INIT, "Initializing fixed-point backend target")
