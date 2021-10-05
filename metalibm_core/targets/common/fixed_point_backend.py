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
    if in_format == out_format:
        # no transformation need when input and output formats are the same
        result = optree
    elif out_sformat.get_bit_size() >= in_sformat.get_bit_size():
        # conversion when the output format is larger than the input format
        # 1st step: start by the conversion between support format
        in_ext = Conversion(TypeCast(op0, precision=in_sformat), precision=out_sformat)
        # 2nd step: fix the alignment
        shift = out_format.get_frac_size() - in_format.get_frac_size()
        if shift > 0:
            result = BitLogicLeftShift(in_ext, CI(shift), precision=out_sformat)
        elif shift < 0:
            result =  BitLogicRightShift(in_ext, CI(-shift), precision=out_sformat)
        else:
            result = in_ext
        # 3rd step: mask the bits which are not part of the output format
        mask_size = out_format.get_integer_size() + out_format.get_frac_size()
        # TODO/FIXME only support right aligned fixed-point format
        assert out_format.support_right_align == 0
        # TODO/FIXME: Interval=None required to bypass default interval building issue
        # for Constant node (2**mask_size-1) may be too big to be converted
        # to SollyaObject successfully
        # TODO/FIXME manage signed cased properly
        if not out_format.get_signed():
            result = BitLogicAnd(result, Constant(2**mask_size-1, precision=out_sformat, interval=None), precision=out_sformat)
        # 4th step: cast towards output format
        result = TypeCast(result, precision=out_format)
    else:
        # conversion when the output format is smaller than the input format
        # 1st step: cast to input support format
        in_s = TypeCast(op0, precision = in_sformat)
        # 2nd step shift
        shift = out_format.get_frac_size() - in_format.get_frac_size()
        if shift > 0:
            result = BitLogicLeftShift(in_s, CI(shift), precision = in_sformat)
        elif shift < 0:
            result = BitLogicRightShift(in_s, CI(-shift), precision = in_sformat)
        else:
            result = in_s

        # 3rd step conversion and masking out excess bits
        result = Conversion(result, precision=out_sformat)
        mask_size = out_format.get_integer_size() + out_format.get_frac_size()
        # TODO/FIXME only support right aligned fixed-point format
        assert out_format.support_right_align == 0
        # TODO/FIXME manage signed cased properly
        if not out_format.get_signed():
            result = BitLogicAnd(result, Constant(2**mask_size-1, precision=out_sformat), precision=out_sformat)
        # 4th step
        result = TypeCast(result, precision=out_format)

    #result.set_tag(optree.get_tag())
    forward_attributes(optree, result)
    Log.report(Log.Debug, "result of conv_modifier on \n %s IS: \n  %s " % (optree.get_str(display_precision = True, depth = 3, memoization_map = {}), result.get_str(display_precision = True, depth = 4)))
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

def unary_io_format_mismtach(optree):
    assert len(optree.inputs) == 1
    return optree.get_precision() != optree.get_input(0).get_precision()


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
    """ cast between two fixed-point are considered valid if the
        size of their support format matches """
    if not isinstance(dst_type, ML_Fixed_Format) or not isinstance(src_type, ML_Fixed_Format): return False
    src_support_type = get_std_integer_support_format(src_type)
    dst_support_type = get_std_integer_support_format(dst_type)

    #return src_support_type.get_bit_size() == dst_support_type.get_bite_size()
    return dst_type.get_c_bit_size() == src_type.get_c_bit_size()

def is_cast_simplification_invalid(dst_type, src_type, **kwords):
  return not is_cast_simplification_valid(dst_type, src_type, **kwords)

def legalize_float2fix_copy_sign(node):
    """ legalize CopySign node from floating-point to fixed-point """
    op = node.get_input(0)
    op_format = op.get_precision().get_base_format()
    int_format = op_format.get_integer_format()
    casted_node = TypeCast(op, precision=int_format)
    shift = op_format.get_field_size() + op_format.get_exponent_size()
    assert shift <= 64
    # TODO/FIXME: shift should be implemented on top of unsigned format
    # to ensure logical shift behavior (and not arihmetic shift)
    return Conversion(
        BitLogicRightShift(
            casted_node,
            Constant(shift, precision=int_format),
            precision=int_format
        ),
        precision=node.get_precision())


def legalize_float2fix_cast(node):
    op = node.get_input(0)
    return TypeCast(
        TypeCast(op, precision=op.get_precision().get_base_format().get_integer_format()),
        precision=node.get_precision()
    )

def legalize_fix2float_cast(node):
    op = node.get_input(0)
    return TypeCast(
        TypeCast(op, precision=node.get_precision().get_base_format().get_integer_format()),
        precision=node.get_precision()
    )

# class Match custom fixed point format
MCFIPF = TCM(ML_Custom_FixedPoint_Format)

def match_specific_fixed_point_format(int_size, frac_size, is_signed):
    def match_fct(candidate_format):
        return MCFIPF(candidate_format) and candidate_format.get_integer_size() == int_size and \
               candidate_format.get_frac_size() == frac_size and candidate_format.get_signed() == is_signed
    return match_fct


def legalize_fixed_point_clz(node):
    assert isinstance(node, CountLeadingZeros)
    op = node.get_input(0)
    op_format = op.get_precision()
    support_format = get_fixed_point_support_format(op.get_precision().get_support_format().get_bit_size(), False)

    raw_lzc = CountLeadingZeros(
        TypeCast(op, precision=support_format),
        precision=support_format
    )
    # TODO/FIXME: does not support non-zero alignment of fixed-point value
    # within support format
    assert op_format.support_right_align == 0
    offset = support_format.get_bit_size() - (op_format.get_integer_size() + op_format.get_frac_size())
    unbiased_lzc = Subtraction(
        raw_lzc,
        Constant(offset, precision=support_format),
        precision=support_format)
    # conversion towards output result
    result = TypeCast(
        Conversion(
            unbiased_lzc,
            precision=node.get_precision().get_support_format()
        ),
        precision=node.get_precision()
    )
    # TODO/FIXME should be "Legalization" verbose level
    Log.report(Log.Info, "legalizing {} to {}", node, result)
    return result

# alis
MSFPF = match_specific_fixed_point_format

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
      lambda op: (round_down_check(op) and unary_io_format_mismtach(op)): {
            type_custom_match(MCFIPF, MCFIPF) : ComplexOperator(optree_modifier=conv_modifier), 
      },
      lambda op: (round_down_check(op) and not unary_io_format_mismtach(op)): {
            # identical input and output format
            type_custom_match(MCFIPF, MCFIPF):
                IdentityOperator(),
      },
      round_down_check: {
        type_custom_match(MCFIPF, FSM(ML_Binary32)) : ComplexOperator(optree_modifier = conv_from_fp_modifier), 
        type_custom_match(FSM(ML_Binary32), MCFIPF) : ComplexOperator(optree_modifier = conv_fixed_to_fp_modifier),

        type_custom_match(MCFIPF, FSM(ML_Binary64)):
            ComplexOperator(optree_modifier=conv_from_fp_modifier),
        type_custom_match(FSM(ML_Binary64), MCFIPF):
            ComplexOperator(optree_modifier=conv_fixed_to_fp_modifier),

      },
      lambda optree: True: {
        # TODO/FIXME: dirty
        type_custom_match(MCFIPF, FSM(ML_Int32)):
            IdentityOperator(),
        type_custom_match(MSFPF(1, 0, False), FSM(ML_Bool)):
            TemplateOperator(" %s ? 1 : 0 ", arity=1),
      },
    },
  },
  TypeCast: {
      None: {
          lambda optree: isinstance(optree.get_precision(), ML_Fixed_Format) and isinstance(optree.get_input(0).get_precision(), ML_Fixed_Format): {
              # type cast between two FixedPoint is the same as TypeCast between the support type
              is_cast_simplification_valid: IdentityOperator(force_folding = False, no_parenthesis = True),
              # FIXME: arbitrary fix
              is_cast_simplification_invalid: IdentityOperator(),
          },
          lambda optree: not (isinstance(optree.get_precision(), ML_Fixed_Format) and isinstance(optree.get_input(0).get_precision(), ML_Fixed_Format)): {
                type_custom_match(MCFIPF, FSM(ML_Binary32)):
                    ComplexOperator(optree_modifier=legalize_float2fix_cast),
                type_custom_match(FSM(ML_Binary32), MCFIPF):
                    ComplexOperator(optree_modifier=legalize_fix2float_cast),

          }
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
    Comparison.Greater: {
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
    Comparison.Less: {
        lambda optree: True: {
            type_custom_match(FSM(ML_Bool), MCFIPF, MCFIPF):
                ComplexOperator(optree_modifier=legalize_fixed_point_comparison),
        },
    },
    Comparison.Equal: {
        lambda optree: True: {
            type_custom_match(FSM(ML_Bool), MCFIPF, MCFIPF):
                ComplexOperator(optree_modifier=legalize_fixed_point_comparison),
        },
    },
  },
    BitLogicLeftShift: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator("<<", arity=2),
            },
        },
    },
    # TODO/FIXME: should make sure right logic shifts are implemented
    # over unsigned support format
    BitLogicRightShift: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator(">>", arity=2),
            type_custom_match(MCFIPF, MCFIPF, FSM(ML_Int32)):
                SymbolOperator(">>", arity=2),
            },
        },
    },
    # TODO/FIXME: should make sure arithmetic shifts are implemented
    # over signed support format
    BitArithmeticRightShift: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator(">>", arity=2),
            type_custom_match(MCFIPF, MCFIPF, FSM(ML_Int32)):
                SymbolOperator(">>", arity=2),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator("|", arity=2),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator("&", arity=2),
            },
        },
    },
    BitLogicXor: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, MCFIPF, MCFIPF):
                SymbolOperator("^", arity=2),
            },
        },
    },
    Select: {
        None: {
            lambda _: True: {
            type_custom_match(MCFIPF, FSM(ML_Bool), MCFIPF, MCFIPF):
                TemplateOperator(" %s ? %s : %s ", arity=3),
            # TODO/FIXME: could be moved into generic processor target
            type_strict_match(ML_Bool, ML_Bool, ML_Bool, ML_Bool):
                TemplateOperator(" %s ? %s : %s ", arity=3),
            },
        },
    },
    CountLeadingZeros: {
        None: {
            # All wrong: should be based on optree.get_input(0) precision and also
            # should take into account non extra leading zeros due to support format
            #lambda optree: (optree.get_precision().get_bit_size() <= 32): {
            #    type_custom_match(MCFIPF, MCFIPF):
            #        FunctionOperator("__builtin_clz", arity = 1),
            #},
            #lambda optree: (32 < optree.get_precision().get_bit_size() <= 64): {
            #    type_custom_match(MCFIPF, MCFIPF):
            #        FunctionOperator("__builtin_clzl", arity = 1),
            #},
            lambda optree: True: {
                type_custom_match(MCFIPF, MCFIPF):
                    ComplexOperator(optree_modifier=legalize_fixed_point_clz),
                type_strict_match(ML_Int32, ML_UInt32):
                    FunctionOperator("__builtin_clzl", arity=1),
                type_strict_match(ML_Int32, ML_UInt64):
                    FunctionOperator("__builtin_clzl", arity=1),
                type_strict_match(ML_UInt64, ML_UInt64):
                    FunctionOperator("__builtin_clzl", arity=1),
            }
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
