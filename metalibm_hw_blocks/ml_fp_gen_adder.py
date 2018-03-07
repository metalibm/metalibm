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
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD, cbrt
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *


from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter

## Helper for debug enabling
debug_std          = ML_Debug(display_format = " -radix 2 ")
debug_dec          = ML_Debug(display_format = " -radix 10 ")
debug_dec_unsigned = ML_Debug(display_format = " -decimal -unsigned ")

## Wrapper for zero extension
# @param op the input operation tree
# @param s integer size of the extension
# @return the Zero extended operation node
def zext(op,s):
  s = int(s)
  op_size = op.get_precision().get_bit_size() 
  ext_precision  = ML_StdLogicVectorFormat(op_size + s)
  return ZeroExt(op, s, precision = ext_precision)

## Generate the right zero extended output from @p optree
def rzext(optree, ext_size):
  ext_size = int(ext_size)
  op_size = optree.get_precision().get_bit_size()
  ext_format = ML_StdLogicVectorFormat(ext_size)
  out_format = ML_StdLogicVectorFormat(op_size + ext_size)
  return Concatenation(optree, Constant(0, precision = ext_format), precision = out_format)

class FP_FMA(ML_Entity("fp_fma")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = VHDLBackend(), 
             output_file = "fp_fma.vhd", 
             entity_name = "fp_fma",
             language = VHDL_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "fp_fma",
      entity_name = entity_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,

      backend = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      language = language,
      arg_template = arg_template
    )

    self.accuracy  = accuracy
    self.precision = precision

  def generate_scheme(self):

    def get_virtual_cst(prec, value, language):
      return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))

    ## convert @p value from an input floating-point precision
    #  @p in_precision to an output support format @p out_precision
    io_precision = VirtualFormat(base_format = self.precision, support_format = ML_StdLogicVectorFormat(self.precision.get_bit_size()), get_cst = get_virtual_cst)
    # declaring standard clock and reset input signal
    #clk = self.implementation.add_input_signal("clk", ML_StdLogic)
    # reset = self.implementation.add_input_signal("reset", ML_StdLogic)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision) 
    vy = self.implementation.add_input_signal("y", io_precision) 

    vx_precision     = self.precision
    vy_precision     = self.precision
    result_precision = self.precision

    # precision for first operand vx which is to be statically 
    # positionned
    p = vx_precision.get_mantissa_size()
    # precision for second operand vy which is to be dynamically shifted
    q = vy_precision.get_mantissa_size()
    # precision of output
    o = result_precision.get_mantissa_size()

    # vx must be aligned with vy
    # the largest shit amount (in absolute value) is precision + 2
    # (1 guard bit and 1 rounding bit)
    exp_vx_precision     = ML_StdLogicVectorFormat(vx_precision.get_exponent_size())
    exp_vy_precision     = ML_StdLogicVectorFormat(vy_precision.get_exponent_size())

    mant_vx_precision    = ML_StdLogicVectorFormat(p-1)
    mant_vy_precision    = ML_StdLogicVectorFormat(q-1)

    mant_vx = MantissaExtraction(vx, precision = mant_vx_precision)
    mant_vy = MantissaExtraction(vy, precision = mant_vy_precision)
    
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision)
    exp_vy = ExponentExtraction(vy, precision = exp_vy_precision)

    # Maximum number of leading zero for normalized <vx>
    L_x = 0
    # Maximum number of leading zero for normalized <vy>
    L_y = 0

    sign_vx = CopySign(vx, precision = ML_StdLogic)
    sign_vy = CopySign(vy, precision = ML_StdLogic)

    # determining if the operation is an addition (effective_op = '0')
    # or a subtraction (effective_op = '1')
    effective_op = BitLogicXor(sign_vx, sign_vy, precision = ML_StdLogic, tag = "effective_op", debug = ML_Debug(display_format = "-radix 2"))

    exp_vx_bias = vx_precision.get_bias()
    exp_vy_bias = vy_precision.get_bias()

    exp_offset = max(o+L_y,q)+2
    exp_bias = exp_offset + exp_vx_bias - exp_vy_bias
    # Determine a working precision to accomodate exponent difference
    # FIXME: check interval and exponent operations size
    exp_precision_ext_size = max(vx_precision.get_exponent_size(), vy_precision.get_exponent_size()) + 2
    exp_precision_ext = ML_StdLogicVectorFormat(exp_precision_ext_size)
    # Y is first aligned offset = max(o+L_y,q) + 2 bits to the left of x 
    # and then shifted right by 
    # exp_diff = exp_x - exp_y + offset
    # exp_vx in [emin, emax]
    # exp_vx - exp_vx + p +2 in [emin-emax + p + 2, emax - emin + p + 2]
    exp_diff = Subtraction(
                Addition(
                  zext(exp_vx, exp_precision_ext_size - vx_precision.get_exponent_size()), 
                  Constant(exp_bias, precision = exp_precision_ext),
                  precision = exp_precision_ext
                ),
                zext(exp_vy, exp_precision_ext_size - vy_precision.get_exponent_size()), 
                precision = exp_precision_ext,
                tag = "exp_diff",
                debug = debug_std
    )
    signed_exp_diff = SignCast(
      exp_diff, 
      specifier = SignCast.Signed, 
      precision = exp_precision_ext
    )
    datapath_full_width = exp_offset + max(o + L_x, p) + 2 + q
    max_exp_diff = datapath_full_width - q 
    exp_diff_lt_0 = Comparison(
      signed_exp_diff,
      Constant(0, precision = exp_precision_ext), 
      specifier = Comparison.Less, 
      precision = ML_Bool, 
      tag = "exp_diff_lt_0", 
      debug = debug_std
    )
    exp_diff_gt_max_diff = Comparison(signed_exp_diff, Constant(max_exp_diff, precision = exp_precision_ext), specifier = Comparison.Greater, precision = ML_Bool)

    shift_amount_prec = ML_StdLogicVectorFormat(int(floor(log2(max_exp_diff))+1))

    mant_shift = Select(
      exp_diff_lt_0,
      Constant(0, precision = shift_amount_prec),
      Select(
        exp_diff_gt_max_diff,
        Constant(max_exp_diff, precision = shift_amount_prec),
        Truncate(exp_diff, precision = shift_amount_prec),
        precision = shift_amount_prec
      ),
      precision = shift_amount_prec,
      tag = "mant_shift",
      debug = ML_Debug(display_format = "-radix 10")
    )

    mant_ext_size = max_exp_diff
    shift_prec = ML_StdLogicVectorFormat(datapath_full_width)
    shifted_mant_vy = BitLogicRightShift(rzext(mant_vy, mant_ext_size), mant_shift, precision = shift_prec, tag = "shifted_mant_vy", debug = debug_std)
    # vx is right-extended by q+2 bits
    # and left extend by exp_offset
    mant_vx_ext = zext(rzext(mant_vx, q+2), exp_offset+1)

    add_prec = ML_StdLogicVectorFormat(datapath_full_width + 1)

    mant_vx_add_op = Select(
      Comparison(
        effective_op,
        Constant(1, precision = ML_StdLogic),
        precision = ML_Bool,
        specifier = Comparison.Equal
      ),
      Negation(mant_vx_ext, precision = add_prec, tag = "neg_mant_vx"),
      mant_vx_ext,
      precision = add_prec,
      tag = "mant_vx_add_op",
      debug = ML_Debug(display_format = " ")
    )
      

    mant_add = Addition(
                 zext(shifted_mant_vy, 1),
                 mant_vx_add_op,
                 precision = add_prec,
                 tag = "mant_add",
                 debug = ML_Debug(display_format = " -radix 2")
              )

    # if the addition overflows, then it meant vx has been negated and
    # the 2's complement addition cancelled the negative MSB, thus
    # the addition result is positive, and the result is of the sign of Y
    # else the result is of opposite sign to Y
    add_is_negative = BitLogicAnd(
        CopySign(mant_add, precision = ML_StdLogic),
        effective_op,
        precision = ML_StdLogic,
        tag = "add_is_negative",
        debug = ML_Debug(" -radix 2")
      )
    # Negate mantissa addition result if it is negative
    mant_add_abs = Select(
      Comparison(
        add_is_negative,
        Constant(1, precision = ML_StdLogic),
        specifier = Comparison.Equal,
        precision = ML_Bool
      ),
      Negation(mant_add, precision = add_prec, tag = "neg_mant_add", debug = debug_std),
      mant_add,
      precision = add_prec,
      tag = "mant_add_abs",
      debug = debug_std
    )

    res_sign = BitLogicXor(add_is_negative, sign_vy, precision = ML_StdLogic, tag = "res_sign")

    # Precision for leading zero count
    lzc_width = int(floor(log2(datapath_full_width + 1)) + 1)
    lzc_prec = ML_StdLogicVectorFormat(lzc_width)

    lzc_args = ML_LeadingZeroCounter.get_default_args(width = (datapath_full_width + 1))
    LZC_entity = ML_LeadingZeroCounter(lzc_args)
    lzc_entity_list = LZC_entity.generate_scheme()
    lzc_implementation = LZC_entity.get_implementation()

    lzc_component = lzc_implementation.get_component_object()

    #lzc_in = SubSignalSelection(mant_add, p+1, 2*p+3)
    lzc_in = mant_add_abs # SubSignalSelection(mant_add_abs, 0, 3*p+3, precision = ML_StdLogicVectorFormat(3*p+4))

    add_lzc = Signal("add_lzc", precision = lzc_prec, var_type = Signal.Local, debug = debug_dec)
    add_lzc = PlaceHolder(add_lzc, lzc_component(io_map = {"x": lzc_in, "vr_out": add_lzc}))

    # Index of output mantissa least significant bit
    mant_lsb_index = datapath_full_width - o + 1

    #add_lzc = CountLeadingZeros(mant_add, precision = lzc_prec)
    # CP stands for close path, the data path where X and Y are within 1 exp diff
    res_normed_mant = BitLogicLeftShift(mant_add_abs, add_lzc, precision = add_prec, tag = "res_normed_mant", debug = debug_std)
    pre_mant_field = SubSignalSelection(res_normed_mant, mant_lsb_index, datapath_full_width - 1, precision = ML_StdLogicVectorFormat(o-1))

    ## Helper function to extract a single bit
    #  from a vector of bits signal
    def BitExtraction(optree, index, **kw):
      return VectorElementSelection(optree, index, precision = ML_StdLogic, **kw)
    def IntCst(value):
      return Constant(value, precision = ML_Integer)


    round_bit = BitExtraction(res_normed_mant, IntCst(mant_lsb_index - 1))
    mant_lsb  = BitExtraction(res_normed_mant, IntCst(mant_lsb_index))
    sticky_prec = ML_StdLogicVectorFormat(datapath_full_width - o)
    sticky_input = SubSignalSelection(
      res_normed_mant, 0, datapath_full_width - o - 1, 
      precision =  sticky_prec
    )
    sticky_bit = Select(
      Comparison(
        sticky_input,
        Constant(0, precision = sticky_prec),
        specifier = Comparison.NotEqual,
        precision = ML_Bool
      ),
      Constant(1, precision = ML_StdLogic),
      Constant(0, precision = ML_StdLogic),
      precision = ML_StdLogic,
      tag = "sticky_bit",
      debug = debug_std
    )

    # increment selection for rouding to nearest (tie to even)
    round_increment_RN = BitLogicAnd(
      round_bit,
      BitLogicOr(
        sticky_bit,
        mant_lsb,
        precision = ML_StdLogic
      ),
      precision = ML_StdLogic,
      tag = "round_increment_RN",
      debug = debug_std
    )

    rounded_mant = Addition(
      zext(pre_mant_field, 1),
      round_increment_RN,
      precision = ML_StdLogicVectorFormat(o),
      tag = "rounded_mant",
      debug = debug_std
    )
    rounded_overflow = BitExtraction(
      rounded_mant, 
      IntCst(o-1), 
      tag = "rounded_overflow", 
      debug = debug_std
    )
    res_mant_field = Select(
      Comparison(
        rounded_overflow,
        Constant(1, precision = ML_StdLogic),
        specifier = Comparison.Equal,
        precision = ML_Bool
      ),
      SubSignalSelection(rounded_mant, 1, o-1),
      SubSignalSelection(rounded_mant, 0, o-2),
      precision = ML_StdLogicVectorFormat(o-1),
      tag = "final_mant",
      debug = debug_std
    )


    res_exp_tmp_size = max(vx_precision.get_exponent_size(), vy_precision.get_exponent_size()) + 2

    res_exp_tmp_prec = ML_StdLogicVectorFormat(res_exp_tmp_size)

    exp_vy_biased = Addition(
      zext(exp_vy, res_exp_tmp_size - vy_precision.get_exponent_size()),
      Constant(vy_precision.get_bias() + 1, precision = res_exp_tmp_prec),
      precision = res_exp_tmp_prec,
      tag = "exp_vy_biased",
      debug = debug_dec
    )
    # vx's exponent is biased with the format bias
    # plus the exponent offset so it is left align to datapath MSB
    exp_vx_biased = Addition(
      zext(exp_vx, res_exp_tmp_size - vx_precision.get_exponent_size()),
      Constant(
        vx_precision.get_bias() + exp_offset + 1, 
        precision = res_exp_tmp_prec
      ),
      precision = res_exp_tmp_prec,
      tag = "exp_vx_biased",
      debug = debug_dec
    )

    # If exp diff is less than 0, then we must consider that vy's exponent is
    # the meaningful one and thus compute result exponent with respect
    # to vy's exponent value
    res_exp_base = Select(
      exp_diff_lt_0,
      exp_vy_biased,
      exp_vx_biased,
      precision = res_exp_tmp_prec,
      tag = "res_exp_base",
      debug = debug_dec
    )

    # Eventually we add the result exponent base 
    # with the exponent offset and the leading zero count
    res_exp_ext = Addition(
      Subtraction(
        Addition(
          zext(res_exp_base, 0),
          Constant(-result_precision.get_bias(), precision = res_exp_tmp_prec),
          precision = res_exp_tmp_prec
        ),
        zext(add_lzc, res_exp_tmp_size - lzc_width), 
        precision = res_exp_tmp_prec
      ),
      rounded_overflow,
      precision = res_exp_tmp_prec,
      tag = "res_exp_ext",
      debug = debug_std
    )

    res_exp_prec = ML_StdLogicVectorFormat(result_precision.get_exponent_size())

    res_exp = Truncate(res_exp_ext, precision = res_exp_prec, tag = "res_exp", debug = debug_dec_unsigned)

    vr_out = TypeCast(
      FloatBuild(
        res_sign, 
        res_exp, 
        res_mant_field, 
        precision = self.precision,
      ),
      precision = io_precision,
      tag = "result",
      debug = debug_std
    )

    self.implementation.add_output_signal("vr_out", vr_out)

    return lzc_entity_list + [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    vy = io_map["y"]
    result = {}
    print "vx, vy"
    print vx, vx.__class__
    print vy, vy.__class__
    result["vr_out"] = sollya.round(vx + vy, self.precision.get_sollya_object(), sollya.RN)
    return result

  # standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
  standard_test_cases = [
    ({"x": 1.0, "y": (S2**-53 + S2**-54)}, None),
    ({
      "y": ML_Binary64.get_value_from_integer_coding("47d273e91e2c9048", base = 16),
      "x": ML_Binary64.get_value_from_integer_coding("c7eea5670485a5ec", base = 16)
    }, None),
    ({
      "y": ML_Binary64.get_value_from_integer_coding("75164a1df94cd488", base = 16),
      "x": ML_Binary64.get_value_from_integer_coding("5a7567b08508e5b4", base = 16)
    }, None)
  ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_fp_fma", default_output_file = "ml_fp_fma.vhd" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_fma      = FP_FMA(args)

    ml_hw_fma.gen_implementation()
