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

from sollya import Interval, ceil, floor, round, log2
S2 = sollya.SollyaObject(2)
from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_cst_dec)
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

from metalibm_core.utility.rtl_debug_utils import *


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *


from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter


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

class FP_FIXED_MPFMA(ML_Entity("fp_fixed_mpfma")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             precision = ML_Binary32, 
             target = VHDLBackend(), 
             debug_flag = False,
             output_file = "fp_fixed_mpfma.vhd", 
             entity_name = "fp_fixed_mpfma",
             language = VHDL_Code,
             vector_size = 1,
             ):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "fp_fixed_mpfma",
      entity_name = entity_name,
      output_file = output_file,

      io_precisions = io_precisions,

      backend = target,

      debug_flag = debug_flag,
      language = language,
      arg_template = arg_template
    )

    self.precision = precision.get_base_format()
    self.io_precision = precision
    # number of extra bits to add to the accumulator fixed precision
    self.extra_digit = arg_template.extra_digit

    min_prod_exp = self.precision.get_emin_subnormal() * 2
    self.acc_lsb_index = min_prod_exp
    # select sign-magintude encoded accumulator
    self.sign_magnitude = arg_template.sign_magnitude
    # enable/disable operator pipelining
    self.pipelined      = arg_template.pipelined

  @staticmethod
  def get_default_args(**kw):
    default_mapping = {
      "extra_digit" : 0,
      "sign_magnitude" : False,
      "pipelined" : False
    }
    default_mapping.update(kw)
    return DefaultEntityArgTemplate(
      **default_mapping
    )

  def get_acc_lsb_index(self):
    return self.acc_lsb_index

  def generate_scheme(self):
    ## Generate Fused multiply and add comput <x> . <y> + <z>
    Log.report(Log.Info, "generating fixed MPFMA with {ed} extra digit(s) and sign-magnitude accumulator: {sm}".format(ed = self.extra_digit, sm = self.sign_magnitude))

    def get_virtual_cst(prec, value, language):
      return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))

    ## convert @p value from an input floating-point precision
    #  @p in_precision to an output support format @p out_precision
    io_precision = self.io_precision
    # declaring standard clock and reset input signal
    #clk = self.implementation.add_input_signal("clk", ML_StdLogic)
    # reset = self.implementation.add_input_signal("reset", ML_StdLogic)
    # declaring main input variable

    # maximum weigth for a mantissa product digit
    max_prod_exp = self.precision.get_emax() * 2 + 1
    # minimum wieght for a mantissa product digit
    min_prod_exp = self.precision.get_emin_subnormal() * 2

    ## Most and least significant digit index for the
    #  accumulator
    acc_msb_index = max_prod_exp + self.extra_digit
    acc_lsb_index = min_prod_exp

    acc_width = acc_msb_index - min_prod_exp + 1
    # precision of the accumulator
    acc_prec = ML_StdLogicVectorFormat(acc_width)

    reset = self.implementation.add_input_signal("reset", ML_StdLogic)

    vx = self.implementation.add_input_signal("x", io_precision) 
    vy = self.implementation.add_input_signal("y", io_precision) 

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()

    acc = self.implementation.add_input_signal("acc", acc_prec)
    if self.sign_magnitude:
      # the accumulator is in sign-magnitude representation
      sign_acc = self.implementation.add_input_signal("sign_acc", ML_StdLogic)
    else:
      sign_acc = CopySign(acc, precision = ML_StdLogic, tag = "sign_acc", debug = debug_std)

    vx_precision     = self.precision
    vy_precision     = self.precision
    result_precision = acc_prec

    # precision for first operand vx which is to be statically 
    # positionned
    p = vx_precision.get_mantissa_size()
    # precision for second operand vy which is to be dynamically shifted
    q = vy_precision.get_mantissa_size()

    # vx must be aligned with vy
    # the largest shit amount (in absolute value) is precision + 2
    # (1 guard bit and 1 rounding bit)
    exp_vx_precision     = ML_StdLogicVectorFormat(vx_precision.get_exponent_size())
    exp_vy_precision     = ML_StdLogicVectorFormat(vy_precision.get_exponent_size())

    mant_vx_precision    = ML_StdLogicVectorFormat(p-1)
    mant_vy_precision    = ML_StdLogicVectorFormat(q-1)

    mant_vx = MantissaExtraction(vx, precision = mant_vx_precision)
    mant_vy = MantissaExtraction(vy, precision = mant_vy_precision)
    
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision, tag = "exp_vx", debug = debug_dec)
    exp_vy = ExponentExtraction(vy, precision = exp_vy_precision, tag = "exp_vy", debug = debug_dec)

    # Maximum number of leading zero for normalized <vx> mantissa
    L_x = 0
    # Maximum number of leading zero for normalized <vy> mantissa
    L_y = 0
    # Maximum number of leading zero for the product of <x>.<y>
    # mantissa. 
    L_xy = L_x + L_y + 1

    sign_vx = CopySign(vx, precision = ML_StdLogic)
    sign_vy = CopySign(vy, precision = ML_StdLogic)

    # determining if the operation is an addition (effective_op = '0')
    # or a subtraction (effective_op = '1')
    sign_xy = BitLogicXor(sign_vx, sign_vy, precision = ML_StdLogic, tag = "sign_xy", debug = debug_std)
    effective_op = BitLogicXor(sign_xy, sign_acc, precision = ML_StdLogic, tag = "effective_op", debug = debug_std)

    exp_vx_bias = vx_precision.get_bias()
    exp_vy_bias = vy_precision.get_bias()

    # <acc> is statically positionned in the datapath,
    # it may even constitute the whole datapath
    #
    # the product is shifted with respect to the fix accumulator

    exp_bias = (exp_vx_bias + exp_vy_bias) 

    # because of the mantissa range [1, 2[, the product exponent
    # is located one bit to the right (lower) of the product MSB
    prod_exp_offset = 1

    # Determine a working precision to accomodate exponent difference
    # FIXME: check interval and exponent operations size
    exp_precision_ext_size = max(
      vx_precision.get_exponent_size(), 
      vy_precision.get_exponent_size(),
      abs(ceil(log2(abs(acc_msb_index)))),
      abs(ceil(log2(abs(acc_lsb_index)))),
      abs(ceil(log2(abs(exp_bias + prod_exp_offset)))),
    ) + 2
    Log.report(Log.Info, "exp_precision_ext_size={}".format(exp_precision_ext_size))
    exp_precision_ext = ML_StdLogicVectorFormat(exp_precision_ext_size)

    # static accumulator exponent
    exp_acc = Constant(acc_msb_index, precision = exp_precision_ext, tag = "exp_acc", debug = debug_cst_dec) 

    # Y is first aligned offset = max(o+L_y,q) + 2 bits to the left of x 
    # and then shifted right by 
    # exp_diff = exp_x - exp_y + offset
    # exp_vx in [emin, emax]
    # exp_vx - exp_vx + p +2 in [emin-emax + p + 2, emax - emin + p + 2]
    exp_diff = UnsignedSubtraction(
                exp_acc,
                UnsignedAddition(
                  UnsignedAddition(
                    zext(exp_vy, exp_precision_ext_size - vy_precision.get_exponent_size()), 
                    zext(exp_vx, exp_precision_ext_size - vx_precision.get_exponent_size()), 
                    precision = exp_precision_ext
                  ),
                  Constant(exp_bias + prod_exp_offset, precision = exp_precision_ext, tag = "diff_bias", debug = debug_cst_dec),
                  precision = exp_precision_ext,
                  tag = "pre_exp_diff",
                  debug = debug_dec
                ),
                precision = exp_precision_ext,
                tag = "exp_diff",
                debug = debug_dec
    )
    exp_precision_ext_signed = get_signed_precision(exp_precision_ext)
    signed_exp_diff = SignCast(
      exp_diff,
      specifier = SignCast.Signed,
      precision = exp_precision_ext_signed
    )
    datapath_full_width = acc_width
    # the maximum exp diff is the size of the datapath
    # minus the bit size of the product
    max_exp_diff = datapath_full_width - (p + q)
    exp_diff_lt_0 = Comparison(
      signed_exp_diff,
      Constant(0, precision = exp_precision_ext_signed),
      specifier = Comparison.Less,
      precision = ML_Bool,
      tag = "exp_diff_lt_0",
      debug = debug_std
    )
    exp_diff_gt_max_diff = Comparison(signed_exp_diff, Constant(max_exp_diff, precision = exp_precision_ext_signed), specifier = Comparison.Greater, precision = ML_Bool)

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
      debug = debug_dec
    )

    prod_prec = ML_StdLogicVectorFormat(p+q)
    prod = UnsignedMultiplication(
      mant_vx,
      mant_vy,
      precision = prod_prec,
      tag = "prod",
      debug = debug_std
    )

    # attempt at pipelining the operator
    # self.implementation.start_new_stage()

    mant_ext_size = datapath_full_width - (p+q)
    shift_prec = ML_StdLogicVectorFormat(datapath_full_width)
    shifted_prod = BitLogicRightShift(rzext(prod, mant_ext_size), mant_shift, precision = shift_prec, tag = "shifted_prod", debug = debug_std)

    ## Inserting a pipeline stage after the product shifting
    if self.pipelined: self.implementation.start_new_stage()


    if self.sign_magnitude:
      # the accumulator is in sign-magnitude representation

      acc_negated = Select(
        Comparison(
          sign_xy,
          sign_acc,
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        acc,
        BitLogicNegate(acc, precision = acc_prec),
        precision = acc_prec
      )

      # one extra MSB bit is added to the final addition
      # to detect overflows
      add_width = acc_width + 1
      add_prec = ML_StdLogicVectorFormat(add_width)
     
      # FIXME: implement with a proper compound adder
      mant_add_p0_ext = UnsignedAddition(
        zext(shifted_prod, 1),
        zext(acc_negated, 1),
        precision = add_prec
      )
      mant_add_p1_ext = UnsignedAddition(
        mant_add_p0_ext,
        Constant(1, precision = ML_StdLogic),
        precision = add_prec,
        tag = "mant_add",
        debug = debug_std
      )
      # discarding carry overflow bit
      mant_add_p0 = SubSignalSelection(mant_add_p0_ext, 0, acc_width - 1, precision = acc_prec)
      mant_add_p1 = SubSignalSelection(mant_add_p1_ext, 0, acc_width - 1, precision = acc_prec)

      mant_add_pre_sign = CopySign(mant_add_p1_ext, precision = ML_StdLogic, tag = "mant_add_pre_sign", debug = debug_std)
      mant_add = Select(
        Comparison(
          sign_xy,
          sign_acc,
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        mant_add_p0,
        Select(
          Comparison(
            mant_add_pre_sign,
            Constant(1, precision = ML_StdLogic),
            specifier = Comparison.Equal,
            precision = ML_Bool
          ),
          mant_add_p1,
          BitLogicNegate(
            mant_add_p0,
            precision = acc_prec
          ),
          precision = acc_prec,
        ),
        precision = acc_prec,
        tag = "mant_add"
      )


      # if both operands had the same sign, then
      # mant_add is necessarily positive and the result
      # sign matches the input sign
      # if both operands had opposite signs, then
      # the result sign matches the product sign
      # if mant_add is positive, else the accumulator sign
      output_sign = Select(
        Comparison(
          effective_op,
          Constant(1, precision = ML_StdLogic),
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        # if the effective op is a subtraction (prod - acc)
        BitLogicXor(
          sign_acc,
          mant_add_pre_sign,
          precision = ML_StdLogic
        ),
        # the effective op is an addition, thus result and
        # acc share sign
        sign_acc,
        precision = ML_StdLogic,
        tag = "output_sign"
      )

      if self.pipelined: self.implementation.start_new_stage()
        
      # adding output
      self.implementation.add_output_signal("vr_sign", output_sign)
      self.implementation.add_output_signal("vr_acc", mant_add)

    else:
      # 2s complement encoding of the accumulator,
      # the accumulator is never negated, only the producted
      # is negated if negative
     
      # negate shifted prod when required
      shifted_prod_op = Select(
        Comparison(
          sign_xy,
          Constant(1, precision = ML_StdLogic),
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        Negation(shifted_prod, precision = shift_prec),
        shifted_prod,
        precision = shift_prec
      )

      add_prec = shift_prec # ML_StdLogicVectorFormat(datapath_full_width + 1)


      mant_add = UnsignedAddition(
                   shifted_prod_op,
                   acc,
                   precision = acc_prec,
                   tag = "mant_add",
                   debug = debug_std
                )

      if self.pipelined: self.implementation.start_new_stage()

      self.implementation.add_output_signal("vr_acc", mant_add)

    return [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    vy = io_map["y"]
    acc = io_map["acc"]
    result = {}
    acc_lsb_index = self.get_acc_lsb_index()
    if self.sign_magnitude:
      sign_acc = io_map["sign_acc"]
      acc = -acc if sign_acc else acc
      result_value = int(sollya.nearestint((vx * vy + acc *S2**acc_lsb_index)*S2**-acc_lsb_index))
      result_sign = 1 if result_value < 0 else 0
      result["vr_sign"] = result_sign
      result["vr_acc"]  = abs(result_value)
    else:
      result_value = int(sollya.nearestint((vx * vy + acc *S2**acc_lsb_index)*S2**-acc_lsb_index))
      result["vr_acc"] = result_value
    return result

  standard_test_cases = [
    #({
      #"y": ML_Binary16.get_value_from_integer_coding("bab9", base = 16),
      #"x": ML_Binary16.get_value_from_integer_coding("bbff", base = 16),
      #"acc": int("1000000011111001011000111000101000101101110110001010011000101001001111100010101001", 2),
      #"sign_acc": 0
      #}, None),
      ({
        "y": ML_Binary16.get_value_from_integer_coding("bbff", base = 16),
        "x": ML_Binary16.get_value_from_integer_coding("bbfa", base = 16),
        "acc": int("1000100010100111001111000001000001101100110110011010001001011011000010010111111001", 2),
        "sign_acc": 1}, None),
  ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_fp_fixed_mpfma", default_output_file = "ml_fp_fixed_mpfma.vhd" )
    # extra digit command line argument
    arg_template.parser.add_argument("--extra-digit", dest = "extra_digit", type=int, default = 0, help = "set the number of accumulator extra digits")
    arg_template.parser.add_argument("--sign-magnitude", dest = "sign_magnitude", action = "store_const", default = False, const = True, help = "set sign-magnitude encoding for the accumulator")
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_fp_fixed_mpfma      = FP_FIXED_MPFMA(args)

    ml_hw_fp_fixed_mpfma.gen_implementation()
