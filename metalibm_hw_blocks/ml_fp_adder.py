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

from sollya import Interval, floor, round, log2
S2 = sollya.SollyaObject(2)
from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *


from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter

from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_dec_unsigned, debug_cst_dec
)

class FP_Adder(ML_Entity("fp_adder")):
  def __init__(self, arg_template=DefaultEntityArgTemplate):
    # initializing I/O precision

    # initializing base class
    ML_EntityBasis.__init__(self,
      arg_template = arg_template
    )

    self.precision = arg_template.precision

  ## Generate default arguments structure (before any user / test overload)
  @staticmethod
  def get_default_args(**kw):
    default_arg_map = {
      "precision": ML_Binary32,
      "pipelined": False,
      "output_file": "fp_adder.vhd",
      "entity_name": "fp_adder",
      "language": VHDL_Code,
      "passes": [("beforecodegen:size_datapath")],
    }
    default_arg_map.update(**kw)
    return DefaultEntityArgTemplate(**default_arg_map)

  def generate_scheme(self):

    def get_virtual_cst(prec, value, language):
      return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))
    ## convert @p value from an input floating-point precision
    #  @p in_precision to an output support format @p out_precision
    io_precision = HdlVirtualFormat(self.precision)
    # declaring standard clock and reset input signal
    #clk = self.implementation.add_input_signal("clk", ML_StdLogic)
    reset = self.implementation.add_input_signal("reset", ML_StdLogic)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision)
    vy = self.implementation.add_input_signal("y", io_precision)

    base_precision = self.precision.get_base_format()
    p = base_precision.get_mantissa_size()

    # vx must be aligned with vy
    # the largest shit amount (in absolute value) is precision + 2
    # (1 guard bit and 1 rounding bit)
    exp_precision     = ML_StdLogicVectorFormat(base_precision.get_exponent_size())

    mant_precision    = ML_StdLogicVectorFormat(base_precision.get_mantissa_size())

    mant_vx = MantissaExtraction(vx, precision = mant_precision)
    mant_vy = MantissaExtraction(vy, precision = mant_precision)

    exp_vx = ExponentExtraction(vx, precision = exp_precision)
    exp_vy = ExponentExtraction(vy, precision = exp_precision)

    sign_vx = CopySign(vx, precision = ML_StdLogic)
    sign_vy = CopySign(vy, precision = ML_StdLogic)

    # determining if the operation is an addition (effective_op = '0')
    # or a subtraction (effective_op = '1')
    effective_op = BitLogicXor(sign_vx, sign_vy, precision = ML_StdLogic, tag = "effective_op", debug=debug_std)

    ## Wrapper for zero extension
    # @param op the input operation tree
    # @param s integer size of the extension
    # @return the Zero extended operation node
    def zext(op,s):
      op_size = op.get_precision().get_bit_size()
      ext_precision  = ML_StdLogicVectorFormat(op_size + s)
      return ZeroExt(op, s, precision = ext_precision)
    ## Generate the right zero extended output from @p optree
    def rzext(optree, ext_size):
      op_size = optree.get_precision().get_bit_size()
      ext_format = ML_StdLogicVectorFormat(ext_size)
      out_format = ML_StdLogicVectorFormat(op_size + ext_size)
      return Concatenation(optree, Constant(0, precision = ext_format), precision = out_format)

    exp_bias = p + 2
    exp_precision_ext = fixed_point(base_precision.get_exponent_size() + 2, 0)
    exp_precision = fixed_point(base_precision.get_exponent_size(), 0, signed=False)
    # Y is first aligned p+2 bit to the left of x
    # and then shifted right by
    # exp_diff = exp_x - exp_y + precision + 2
    # exp_vx in [emin, emax]
    # exp_vx - exp_vx + p +2 in [emin-emax + p + 2, emax - emin + p + 2]
    exp_diff = Subtraction(
        Addition(
            TypeCast(exp_vx, precision=exp_precision),
            Constant(exp_bias, precision=exp_precision_ext),
        ),
        TypeCast(exp_vy, precision=exp_precision),
    )
    exp_diff_lt_0 = Comparison(exp_diff, Constant(0, precision=exp_precision_ext), specifier = Comparison.Less, precision = ML_Bool)
    exp_diff_gt_2pp4 = Comparison(exp_diff, Constant(2*p+4, precision = exp_precision_ext), specifier = Comparison.Greater, precision = ML_Bool)

    shift_amount_size = int(floor(log2(2*p+4))+1)
    shift_amount_prec = ML_StdLogicVectorFormat(shift_amount_size)

    mant_shift = Select(
      exp_diff_lt_0,
      0,
      Select(
        exp_diff_gt_2pp4,
        Constant(2*p+4),
        exp_diff,
      ),
      tag = "mant_shift",
      debug = debug_dec
    )

    mant_shift = TypeCast(
        Conversion(mant_shift, precision=fixed_point(shift_amount_size, 0, signed=False)),
        precision=shift_amount_prec
    )

    mant_ext_size = 2*p+4
    shift_prec = ML_StdLogicVectorFormat(3*p+4)
    shifted_mant_vy = BitLogicRightShift(rzext(mant_vy, mant_ext_size), mant_shift, precision = shift_prec, tag = "shifted_mant_vy", debug = debug_std)
    mant_vx_ext = zext(rzext(mant_vx, p+2), p+2+1)
    mant_vx_ext.set_attributes(tag="mant_vx_ext")

    add_prec = ML_StdLogicVectorFormat(3*p+5)

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
      debug=debug_cst_dec
    )


    mant_add = UnsignedAddition(
                 zext(shifted_mant_vy, 1),
                 mant_vx_add_op,
                 precision = add_prec,
                 tag = "mant_add",
                 debug=debug_std
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
        debug = debug_std
      )
    # Negate mantissa addition result if it is negative
    mant_add_abs = Select(
      Comparison(
        add_is_negative,
        Constant(1, precision = ML_StdLogic),
        specifier = Comparison.Equal,
        precision = ML_Bool
      ),
      Negation(mant_add, precision = add_prec, tag = "neg_mant_add"),
      mant_add,
      precision = add_prec,
      tag = "mant_add_abs"
    )

    res_sign = BitLogicXor(add_is_negative, sign_vy, precision = ML_StdLogic, tag = "res_sign")

    # Precision for leading zero count
    lzc_width = int(floor(log2(3*p+5)) + 1)
    lzc_prec = ML_StdLogicVectorFormat(lzc_width)


    add_lzc = CountLeadingZeros(
        mant_add_abs,
        precision=lzc_prec,
        tag="add_lzc",
        debug=debug_dec_unsigned
    )

    #add_lzc = CountLeadingZeros(mant_add, precision = lzc_prec)
    # CP stands for close path, the data path where X and Y are within 1 exp diff
    res_normed_mant = BitLogicLeftShift(mant_add, add_lzc, precision = add_prec, tag = "res_normed_mant", debug = debug_std)
    pre_mant_field = SubSignalSelection(res_normed_mant, 2*p+5, 3*p+3, precision = ML_StdLogicVectorFormat(p-1))

    ## Helper function to extract a single bit
    #  from a vector of bits signal
    def BitExtraction(optree, index, **kw):
      return VectorElementSelection(optree, index, precision = ML_StdLogic, **kw)
    def IntCst(value):
      return Constant(value, precision = ML_Integer)

    round_bit = BitExtraction(res_normed_mant, IntCst(2*p+4))
    mant_lsb  = BitExtraction(res_normed_mant, IntCst(2*p+5))
    sticky_prec = ML_StdLogicVectorFormat(2*p+4)
    sticky_input = SubSignalSelection(
      res_normed_mant, 0, 2*p+3, 
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

    rounded_mant = UnsignedAddition(
      zext(pre_mant_field, 1),
      round_increment_RN,
      precision = ML_StdLogicVectorFormat(p),
      tag = "rounded_mant",
      debug = debug_std
    )
    rounded_overflow = BitExtraction(
      rounded_mant, 
      IntCst(p-1), 
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
      SubSignalSelection(rounded_mant, 1, p-1),
      SubSignalSelection(rounded_mant, 0, p-2),
      precision = ML_StdLogicVectorFormat(p-1),
      tag = "final_mant",
      debug = debug_std
    )

    res_exp_prec_size = base_precision.get_exponent_size() + 2
    res_exp_prec = ML_StdLogicVectorFormat(res_exp_prec_size)

    res_exp_ext = UnsignedAddition(
      UnsignedSubtraction(
        UnsignedAddition(
          zext(exp_vx, 2),
          Constant(3+p, precision = res_exp_prec),
          precision = res_exp_prec
        ),
        zext(add_lzc, res_exp_prec_size - lzc_width), 
        precision = res_exp_prec
      ),
      rounded_overflow,
      precision = res_exp_prec,
      tag = "res_exp_ext",
      debug = debug_std
    )

    res_exp = Truncate(res_exp_ext, precision = ML_StdLogicVectorFormat(base_precision.get_exponent_size()), tag = "res_exp", debug = debug_dec)

    vr_out = TypeCast(
      FloatBuild(
        res_sign,
        res_exp,
        res_mant_field,
        precision = base_precision,
      ),
      precision = io_precision,
      tag = "result",
      debug = debug_std
    )

    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    vy = io_map["y"]
    result = {}
    result["vr_out"] = sollya.round(vx + vy, self.precision.get_sollya_object(), sollya.RN)
    return result

  standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="new_fp_adder", default_output_file="ml_fp_adder.vhd",
        default_arg=FP_Adder.get_default_args()
    )
    # argument extraction 
    args = arg_template.arg_extraction()

    ml_hw_adder      = FP_Adder(args)

    ml_hw_adder.gen_implementation()
