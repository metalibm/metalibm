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

from sollya import Interval, ceil, floor, round
S2 = sollya.SollyaObject(2)
from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

from metalibm_hw_blocks.rtl_blocks import *

from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter

from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_hex
)


## Generate the code for a single step of a newton-Raphson
#  iteration
def generate_NR_iteration(
        recp_input, previous_approx,
        mult_int_frac_size,
        error_int_frac_size,
        tmp_approx_int_frac_size,
        approx_int_frac_size,
        implementation, pipelined=0, tag_suffix=""):
    # unpacking input tuple (python2 / python3 compatibility)
    (mult_int_size, mult_frac_size) = mult_int_frac_size
    (error_int_size, error_frac_size) = error_int_frac_size
    (tmp_approx_int_size, tmp_approx_frac_size) = tmp_approx_int_frac_size
    (approx_int_size, approx_frac_size) = approx_int_frac_size
    # creating required formats
    it_mult_precision = RTL_FixedPointFormat(
      mult_int_size, mult_frac_size,
      support_format = ML_StdLogicVectorFormat(mult_int_size + mult_frac_size)
    )
    error_precision = RTL_FixedPointFormat(
      error_int_size,
      error_frac_size,
      support_format = ML_StdLogicVectorFormat(error_int_size + error_frac_size)
    )
    tmp_approx_precision = RTL_FixedPointFormat(
      tmp_approx_int_size,
      tmp_approx_frac_size,
      support_format = ML_StdLogicVectorFormat(tmp_approx_int_size + tmp_approx_frac_size)
    )
    new_approx_precision = RTL_FixedPointFormat(
      approx_int_size,
      approx_frac_size,
      support_format = ML_StdLogicVectorFormat(approx_int_size + approx_frac_size)
    )
    # computing error
    it_mult = Multiplication(
      recp_input,
      previous_approx,
      precision = it_mult_precision,
      debug = debug_fixed,
      tag = "it_mult" + tag_suffix
    )
    if pipelined >= 2: implementation.start_new_stage()
    it_error = Subtraction(
      Constant(1, precision = it_mult_precision),
      it_mult,
      precision = error_precision,
      tag = "it_error" + tag_suffix,
      debug = debug_fixed
    )
    if pipelined >= 1: implementation.start_new_stage()
 
    # computing new approximation
    approx_mult = Multiplication(
      it_error,
      previous_approx,
      precision = tmp_approx_precision,
      tag = "approx_mult" + tag_suffix,
      debug = debug_fixed
    )
    if pipelined >= 2: implementation.start_new_stage()
 
    new_approx = Addition(
      previous_approx,
      approx_mult,
      precision = new_approx_precision, 
      tag = "new_approx",
      debug = debug_fixed
    )
    return new_approx

class FP_Divider(ML_Entity("fp_div")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             ):

    # initializing base class
    ML_EntityBasis.__init__(self, 
      arg_template = arg_template
    )

    self.pipelined = arg_template.pipelined

  ## default argument template generation
  @staticmethod
  def get_default_args(**kw):
    default_dict = {
        "precision": ML_Binary32,
        "target": VHDLBackend(),
        "output_file": "my_fp_div.vhd", 
        "entity_name": "my_fp_div",
        "language": VHDL_Code,
        "pipelined": False,
    }
    default_dict.update(kw)
    return DefaultEntityArgTemplate(
        **default_dict
    )

  def generate_scheme(self):

    def get_virtual_cst(prec, value, language):
      return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))

    ## convert @p value from an input floating-point precision
    #  @p in_precision to an output support format @p out_precision
    io_precision = HdlVirtualFormat(self.precision)

    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision)

    if self.pipelined:
      self.implementation.add_input_signal("reset", ML_StdLogic)

    vx_precision = self.precision

    p = vx_precision.get_mantissa_size()

    exp_vx_precision     = ML_StdLogicVectorFormat(vx_precision.get_exponent_size())
    mant_vx_precision    = ML_StdLogicVectorFormat(p)

    # mantissa extraction
    mant_vx = MantissaExtraction(vx, precision = mant_vx_precision, tag = "mant_vx")
    # exponent extraction
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision, tag = "exp_vx", debug = debug_dec)

    approx_index_size = 8

    approx_precision = RTL_FixedPointFormat(
      2, approx_index_size,
      support_format = ML_StdLogicVectorFormat(approx_index_size + 2),
    )

    # selecting table index from input mantissa MSBs
    tab_index = SubSignalSelection(mant_vx, p-2 - approx_index_size +1, p-2, tag = "tab_index")

    # declaring reciprocal approximation table
    inv_approx_table = ML_NewTable(dimensions = [2**approx_index_size], storage_precision = approx_precision, tag = "inv_approx_table")
    for i in range(2**approx_index_size):
      num_input = 1 + i * S2**-approx_index_size
      table_value = io_precision.get_base_format().round_sollya_object(1 / num_input)
      inv_approx_table[i] = table_value

    # extracting initial reciprocal approximation
    inv_approx_value = TableLoad(inv_approx_table, tab_index, precision = approx_precision, tag = "inv_approx_value", debug = debug_fixed)


    #inv_approx_value = TypeCast(inv_approx_value, precision = approx_precision)
    pre_it0_input = zext(SubSignalSelection(mant_vx, p-1 - approx_index_size , p-1, tag = "it0_input"), 1)
    it0_input = TypeCast(pre_it0_input, precision = approx_precision, tag = "it0_input", debug = debug_fixed)

    it1_precision = RTL_FixedPointFormat(
      2,
      2 * approx_index_size,
      support_format = ML_StdLogicVectorFormat(2 + 2 * approx_index_size)
    )

    pre_it1_input = zext(SubSignalSelection(mant_vx, p - 1 - 2 * approx_index_size, p -1, tag = "it1_input"), 1)
    it1_input = TypeCast(pre_it1_input, precision = it1_precision, tag = "it1_input", debug = debug_fixed)

    final_approx = generate_NR_iteration(
      it0_input,
      inv_approx_value,
      (2, approx_index_size * 2), # mult precision
      (-3, 2 * approx_index_size), # error precision
      (2, approx_index_size * 3), # new-approx mult
      (2, approx_index_size * 2), # new approx precision
      self.implementation,
      pipelined = 0, #1 if self.pipelined else 0,
      tag_suffix = "_first"
    )

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()

    final_approx = generate_NR_iteration(
      it1_input,
      final_approx,
      # mult precision
      (2, approx_index_size * 3),
      # error precision
      (-6, approx_index_size * 3),
      # approx mult precision
      (2, approx_index_size * 3),
      # new approx precision
      (2, approx_index_size * 3),
      self.implementation,
      pipelined = 1 if self.pipelined else 0,
      tag_suffix = "_second"
    )

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()

    last_it_precision = RTL_FixedPointFormat(
      2,
      p - 1,
      support_format=ML_StdLogicVectorFormat(2 + p - 1)
    )

    pre_last_it_input = zext(mant_vx, 1)
    last_it_input = TypeCast(
        pre_last_it_input, precision=last_it_precision,
        tag="last_it_input", debug=debug_fixed
    )

    final_approx = generate_NR_iteration(
      last_it_input,
      final_approx,
      # mult-precision
      (2, 2 * p - 1),
      # error precision
      (int(- (3 * approx_index_size) / 2), approx_index_size * 2 + p - 1),
      # mult approx mult precision
      (2, approx_index_size * 2 + p - 1),
      # approx precision
      (2, p),
      self.implementation,
      pipelined = 2 if self.pipelined else 0,
      tag_suffix = "_third"
    )

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()

    final_approx = generate_NR_iteration(
      last_it_input,
      final_approx,
      (2, 2 * p),
      (int(-(4 * p)/5), 2 * p),
      (2, 2 * p),
      (2, 2 * p),
      self.implementation,
      pipelined = 2 if self.pipelined else 0,
      tag_suffix = "_last"
    )

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()


    final_approx.set_attributes(tag = "final_approx", debug = debug_fixed)

    # bit indexes to select mantissa from final_approximation
    pre_mant_size = min(self.precision.get_field_size(), final_approx.get_precision().get_frac_size()) 
    final_approx_frac_msb_index = final_approx.get_precision().get_frac_size() - 1
    final_approx_frac_lsb_index = final_approx.get_precision().get_frac_size() - pre_mant_size

    # extracting bit to determine if result should be left-shifted and
    # exponent incremented
    cst_index = Constant(final_approx.get_precision().get_frac_size(), precision = ML_Integer)
    final_approx_casted = TypeCast(final_approx, precision = ML_StdLogicVectorFormat(final_approx.get_precision().get_bit_size()))
    not_decrement = final_approx_casted[cst_index] 
    not_decrement.set_attributes(precision = ML_StdLogic, tag = "not_decrement", debug = debug_std)
    logic_1 = Constant(1, precision = ML_StdLogic)

    result = Select(
      Comparison( not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
      SubSignalSelection(
        TypeCast(
          final_approx, 
          precision = ML_StdLogicVectorFormat(final_approx.get_precision().get_bit_size())
        ),
        final_approx_frac_lsb_index,
        final_approx_frac_msb_index,
      ),
      SubSignalSelection(
        TypeCast(
          final_approx, 
          precision = ML_StdLogicVectorFormat(final_approx.get_precision().get_bit_size())
        ),
        final_approx_frac_lsb_index - 1,
        final_approx_frac_msb_index - 1,
      ),
      precision = ML_StdLogicVectorFormat(pre_mant_size),
      tag = "result"
    )
    def get_bit(optree, bit_index):
      bit_index_cst = Constant(bit_index, precision = ML_Integer)
      bit_sel = VectorElementSelection(
        optree,
        bit_index_cst,
        precision = ML_StdLogic)
      return bit_sel

    least_bit = Select(
      Comparison(not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
      get_bit(final_approx_casted, final_approx_frac_lsb_index),
      get_bit(final_approx_casted, final_approx_frac_lsb_index - 1),
      precision = ML_StdLogic,
      tag = "least_bit",
      debug = debug_std,
    )
    round_bit = Select(
      Comparison(not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
      get_bit(final_approx_casted, final_approx_frac_lsb_index - 1),
      get_bit(final_approx_casted, final_approx_frac_lsb_index - 2),
      precision = ML_StdLogic,
      tag = "round_bit",
      debug = debug_std,
    )
    sticky_bit_input = Select( 
      Comparison(not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
      SubSignalSelection(
        final_approx_casted, 0, 
        final_approx_frac_lsb_index - 2, 
        precision = ML_StdLogicVectorFormat(final_approx_frac_lsb_index - 1)
      ),
      zext(
        SubSignalSelection(
          final_approx_casted, 0, 
          final_approx_frac_lsb_index - 3, 
          precision = ML_StdLogicVectorFormat(final_approx_frac_lsb_index - 2)
        ),
        1
      ),
      precision = ML_StdLogicVectorFormat(final_approx_frac_lsb_index - 1)
    )
    sticky_bit = Select(
      Equal(
        sticky_bit_input, 
        Constant(0, precision = ML_StdLogicVectorFormat(final_approx_frac_lsb_index - 1))
      ),
      Constant(0, precision = ML_StdLogic),
      Constant(1, precision = ML_StdLogic),
      precision = ML_StdLogic,
      tag = "sticky_bit",
      debug = debug_std
    )
    # if mantissa require extension
    if pre_mant_size < self.precision.get_mantissa_size() - 1:
      result = rzext(result, self.precision.get_mantissa_size() - 1 - pre_mant_size) 

    res_mant_field = result

    # real_exp = exp_vx - bias
    # - real_exp = bias - exp_vx
    # encoded negated exp = bias - exp_vx + bias = 2 * bias - exp_vx
    fp_io_precision = io_precision.get_base_format()
    exp_op_precision = ML_StdLogicVectorFormat(fp_io_precision.get_exponent_size() + 2)
    biasX2 = Constant(- 2 * fp_io_precision.get_bias(), precision = exp_op_precision)

    neg_exp = Subtraction(
      SignCast(
        biasX2,
        specifier = SignCast.Unsigned,
        precision = get_unsigned_precision(exp_op_precision)
      ),
      SignCast(
        zext(exp_vx, 2),
        specifier = SignCast.Unsigned,
        precision = get_unsigned_precision(exp_op_precision),
      ),
      precision = exp_op_precision,
      tag = "neg_exp",
      debug = debug_dec
    )
    neg_exp_field = SubSignalSelection(
      neg_exp,
      0,
      fp_io_precision.get_exponent_size() - 1,
      precision = ML_StdLogicVectorFormat(fp_io_precision.get_exponent_size())
    )


    res_exp = Addition(
      SignCast(
        neg_exp_field,
        precision = get_unsigned_precision(exp_vx.get_precision()),
        specifier = SignCast.Unsigned
      ),
      SignCast(
        Select(
          Comparison(not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
          Constant(0, precision = exp_vx_precision),
          Constant(-1, precision = exp_vx_precision),
          precision = exp_vx_precision
        ),
        precision = get_unsigned_precision(exp_vx_precision),
        specifier = SignCast.Unsigned
      ),
      precision = exp_vx_precision,
      tag = "result_exp",
      debug = debug_dec
    )

    res_sign = CopySign(vx, precision = ML_StdLogic)

    exp_mant_precision = ML_StdLogicVectorFormat(io_precision.get_bit_size() - 1)

    round_incr = Select(
      LogicalAnd(
        Equal(round_bit, Constant(1, precision = ML_StdLogic)),
        LogicalOr(
          Equal(sticky_bit, Constant(1, precision = ML_StdLogic)),
          Equal(least_bit, Constant(1, precision = ML_StdLogic)),
          precision = ML_Bool,
        ),
        precision = ML_Bool,
      ),
      Constant(1, precision = ML_StdLogic),
      Constant(0, precision = ML_StdLogic),
      tag = "round_incr",
      precision = ML_StdLogic,
      debug = debug_std
    )

    exp_mant = Concatenation(
      res_exp,
      res_mant_field,
      precision = exp_mant_precision
    )

    exp_mant_rounded = Addition(
      SignCast(
        exp_mant,
        SignCast.Unsigned,
        precision = get_unsigned_precision(exp_mant_precision)
      ),
      round_incr,
      precision = exp_mant_precision,
      tag = "exp_mant_rounded"
    )
    vr_out = TypeCast(
      Concatenation(
        res_sign,
        exp_mant_rounded,
        precision = ML_StdLogicVectorFormat(io_precision.get_bit_size())
      ),
      precision = io_precision,
      debug = debug_hex,
      tag = "vr_out"
    )

    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    result = {}
    result["vr_out"] = sollya.round(1.0 / vx, self.precision.get_sollya_object(), sollya.RN)
    return result

  #standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
  standard_test_cases = [
    ({"x": sollya.parse("0x1.24f608p0")}, None),
    ({"x": 1.5}, None),
  ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="new_fp_div",
        default_output_file="ml_fp_div.vhd",
        default_arg=FP_Divider.get_default_args() )
    # extra command line arguments

    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_div      = FP_Divider(args)

    ml_hw_div.gen_implementation()
