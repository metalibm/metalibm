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
import random

import sollya

from sollya import S2, Interval, ceil, floor, round
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate

from metalibm_core.core.random_gen import FPRandomGen
from metalibm_core.core.advanced_operations import (
    FixedPointPosition
)
from metalibm_core.core.special_values import (
    FP_SpecialValue,
    is_nan,
    is_plus_infty, is_minus_infty, is_sv_omega,
    is_plus_zero, is_minus_zero,
    FP_QNaN, FP_PlusInfty,
)

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

from metalibm_core.targets.kalray.k1c_fp_utils import (
    rnd_mode_format, rnd_rne, rnd_ru, rnd_rd, rnd_rz
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
    it_mult_precision = fixed_point(
      mult_int_size, mult_frac_size,
      signed=True
    )
    error_precision = fixed_point(
      error_int_size,
      error_frac_size,
      signed=True
    )
    tmp_approx_precision = fixed_point(
      tmp_approx_int_size,
      tmp_approx_frac_size,
      signed=True
    )
    new_approx_precision = fixed_point(
      approx_int_size,
      approx_frac_size,
      signed=True
    )
    # computing error
    it_mult = Multiplication(
      recp_input,
      previous_approx,
      precision=it_mult_precision,
      debug=debug_fixed,
      tag="it_mult" + tag_suffix
    )
    if pipelined >= 2: implementation.start_new_stage()
    it_error = Subtraction(
      Constant(1, precision = it_mult_precision),
      it_mult,
      tag="it_error" + tag_suffix,
      precision=error_precision,
      debug=debug_fixed
    )
    if pipelined >= 1: implementation.start_new_stage()

    # computing new approximation
    approx_mult = Multiplication(
      it_error,
      previous_approx,
      tag="approx_mult" + tag_suffix,
      debug=debug_fixed
    )
    if pipelined >= 2: implementation.start_new_stage()

    new_approx = Addition(
      previous_approx,
      approx_mult,
      tag="new_approx",
      debug=debug_fixed
    )
    return Conversion(
        new_approx,
        precision=new_approx_precision
    )


class FP_Divider(ML_Entity("fp_div")):
  def __init__(self,
             arg_template = DefaultEntityArgTemplate,
             ):

    # initializing base class
    ML_EntityBasis.__init__(self,
      arg_template = arg_template
    )
    self.disable_sub_testing = arg_template.disable_sub_testing
    self.disable_sv_testing = arg_template.disable_sv_testing

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
    io_precision = VirtualFormat(
        base_format=self.precision,
        support_format=ML_StdLogicVectorFormat(
            self.precision.get_bit_size()
        ),
        get_cst=get_virtual_cst)

    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision)
    # rounding mode input
    rnd_mode = self.implementation.add_input_signal("rnd_mode", rnd_mode_format)

    if self.pipelined:
      self.implementation.add_input_signal("reset", ML_StdLogic)

    vx_precision = self.precision

    p = vx_precision.get_mantissa_size()
    exp_size = vx_precision.get_exponent_size()


    exp_vx_precision     = ML_StdLogicVectorFormat(vx_precision.get_exponent_size())
    mant_vx_precision    = ML_StdLogicVectorFormat(p)
    # fixed-point precision for operand's exponent
    exp_fixed_precision =fixed_point(exp_size, 0, signed=False)

    # mantissa extraction
    mant_vx = TypeCast(
        MantissaExtraction(vx, precision=mant_vx_precision, tag="extracted_mantissa"),
        precision=fixed_point(1,p-1,signed=False),
        debug=debug_fixed,
        tag="mant_vx"
    )
    # exponent extraction
    exp_vx = TypeCast(
        ExponentExtraction(vx, precision=exp_vx_precision, tag="exp_vx"),
        precision=exp_fixed_precision
    )

    approx_index_size = 8
    approx_precision = fixed_point(
      2, approx_index_size,
    )

    # selecting table index from input mantissa MSBs
    tab_index = SubSignalSelection(
        mant_vx,
        p-2 - approx_index_size +1,
        p-2, tag="tab_index"
    )

    # declaring reciprocal approximation table
    inv_approx_table = ML_NewTable(
        dimensions=[2**approx_index_size],
        storage_precision=approx_precision, tag="inv_approx_table")
    for i in range(2**approx_index_size):
      num_input = 1 + i * S2**-approx_index_size
      table_value = io_precision.get_base_format().round_sollya_object(1 / num_input)
      inv_approx_table[i] = table_value

    # extracting initial reciprocal approximation
    inv_approx_value = TableLoad(
        inv_approx_table, tab_index,
        precision=approx_precision,
        tag="inv_approx_value",
        debug=debug_fixed)

    #inv_approx_value = TypeCast(inv_approx_value, precision = approx_precision)
    pre_it0_input = zext(SubSignalSelection(mant_vx, p-1 - approx_index_size , p-1, tag = "it0_input"), 1)
    it0_input = TypeCast(pre_it0_input, precision = approx_precision, tag = "it0_input", debug = debug_fixed)

    it1_precision = RTL_FixedPointFormat(
      2,
      2 * approx_index_size,
      support_format = ML_StdLogicVectorFormat(2 + 2 * approx_index_size)
    )

    it1_input = mant_vx

    final_approx = generate_NR_iteration(
      mant_vx,
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
      mant_vx,
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

    final_approx = generate_NR_iteration(
      mant_vx,
      final_approx,
      # mult-precision
      (2, 2 * p - 1),
      # error precision
      (- (3 * approx_index_size) / 2, approx_index_size * 2 + p - 1),
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
      mant_vx,
      final_approx,
      (2, 2 * p),
      (-(4 * p)/5, 2 * p),
      (2, 2 * p),
      (2, 2 * p),
      self.implementation,
      pipelined = 2 if self.pipelined else 0,
      tag_suffix = "_last"
    )

    # Inserting post-input pipeline stage
    if self.pipelined: self.implementation.start_new_stage()

    final_approx.set_attributes(tag="final_approx", debug=debug_hex)

    last_approx_norm = final_approx

    offset_bit = BitSelection(
        last_approx_norm,
        FixedPointPosition(
            last_approx_norm,
            0,
            align=FixedPointPosition.FromPointToLSB
        ),
        tag="offset_bit",
        debug=debug_std
    )

    # extracting bit to determine if result should be left-shifted and
    # exponent incremented
    not_decrement = offset_bit

    final_approx_reduced = SubSignalSelection(
          final_approx,
          FixedPointPosition(
            final_approx, -(p-1),
            align=FixedPointPosition.FromPointToLSB
          ),
          FixedPointPosition(
            final_approx, 0,
            align=FixedPointPosition.FromPointToLSB
          ),
          precision=fixed_point(p,0,signed=False)
    )
    final_approx_reduced_shifted = SubSignalSelection(
          final_approx,
          FixedPointPosition(
            final_approx, -p,
            align=FixedPointPosition.FromPointToLSB
          ),
          FixedPointPosition(
            final_approx, -1,
            align=FixedPointPosition.FromPointToLSB
          ),
          precision=fixed_point(p,0,signed=False)
    )

    # unrounded mantissa field excluding leading digit
    unrounded_mant_field = Select(
      equal_to(not_decrement, 1),
      final_approx_reduced,
      final_approx_reduced_shifted,
      precision=fixed_point(p, 0, signed=False),
      tag="unrounded_mant_field",
      debug=debug_hex,
    )
    def get_bit(optree, bit_index):
        bit_sel = BitSelection(
            optree,
            FixedPointPosition(
                optree,
                -bit_index,
                align=FixedPointPosition.FromPointToLSB
            )
        )
        return bit_sel

    mant_lsb = Select(
      equal_to(not_decrement, 1),
      get_bit(final_approx, p-1),
      get_bit(final_approx, p),
      precision = ML_StdLogic,
      tag = "mant_lsb",
      debug = debug_std,
    )
    round_bit = Select(
      equal_to(not_decrement, 1),
      get_bit(final_approx, p),
      get_bit(final_approx, p+1),
      precision = ML_StdLogic,
      tag = "round_bit",
      debug = debug_std,
    )
    sticky_bit_input = Select(
        equal_to(not_decrement, 1),
        SubSignalSelection(
            final_approx,
            0,
            FixedPointPosition(
                    final_approx,
                    -(p+1),
                    align=FixedPointPosition.FromPointToLSB
            ),
            precision=None,
            tag="sticky_bit_input"
        ),
        SubSignalSelection(
            final_approx,
            0,
            FixedPointPosition(
                    final_approx,
                    -(p+2),
                    align=FixedPointPosition.FromPointToLSB
            ),
            precision=None,
            tag="sticky_bit_input"
        ),
    )
    sticky_bit = Select(
        Equal(
            sticky_bit_input,
            Constant(0, precision=None)
        ),
        Constant(0, precision = ML_StdLogic),
        Constant(1, precision = ML_StdLogic),
        precision = ML_StdLogic,
        tag="sticky_bit",
        debug=debug_std
    )
    # TODO: manage leading digit (in case of subnormal result)
    pre_result = unrounded_mant_field

    # real_exp = exp_vx - bias
    # - real_exp = bias - exp_vx
    # encoded negated exp = bias - exp_vx + bias = 2 * bias - exp_vx
    fp_io_precision = io_precision.get_base_format()

    neg_exp = -2 * fp_io_precision.get_bias() - exp_vx
    neg_exp.set_attributes(tag="neg_exp", debug=debug_fixed)
    res_exp = Subtraction(
        neg_exp,
        Select(
          equal_to(not_decrement, 1),
          Constant(0, precision = exp_fixed_precision),
          Constant(1, precision = exp_fixed_precision),
          precision=None,
          tag="exp_offset",
          debug=debug_fixed
        ),
        tag="res_exp",
        debug=debug_fixed
    )
    res_exp_field = SubSignalSelection(
        res_exp,
        FixedPointPosition(
            res_exp,
            0,
            align=FixedPointPosition.FromPointToLSB,
            tag="res_exp_field LSB"
        ),
        FixedPointPosition(
            res_exp,
            exp_size - 1,
            align=FixedPointPosition.FromPointToLSB,
            tag="res_exp_field MSB"
        ),
        precision=None,
        tag="res_exp_field",
        # debug=debug_fixed
    )

    result_sign = CopySign(vx, precision=ML_StdLogic)

    exp_mant_precision = ML_StdLogicVectorFormat(io_precision.get_bit_size() - 1)

    rnd_mode_is_rne = Equal(rnd_mode, rnd_rne, precision=ML_Bool)
    rnd_mode_is_ru  = Equal(rnd_mode, rnd_ru, precision=ML_Bool)
    rnd_mode_is_rd  = Equal(rnd_mode, rnd_rd, precision=ML_Bool)
    rnd_mode_is_rz  = Equal(rnd_mode, rnd_rz, precision=ML_Bool)

    round_incr = Conversion(
        logical_or_reduce([
            logical_and_reduce([rnd_mode_is_rne, equal_to(round_bit, 1), equal_to(sticky_bit,1)]),
            logical_and_reduce([rnd_mode_is_rne, equal_to(round_bit, 1), equal_to(sticky_bit,0), equal_to(mant_lsb, 1)]),
            logical_and_reduce([rnd_mode_is_ru, equal_to(result_sign, 0), LogicalOr(equal_to(round_bit, 1), equal_to(sticky_bit,1), precision = ML_Bool)]),
            logical_and_reduce([rnd_mode_is_rd, equal_to(result_sign, 1), LogicalOr(equal_to(round_bit, 1), equal_to(sticky_bit,1), precision = ML_Bool)]),
        ]),
        precision = fixed_point(1, 0, signed=False),
        tag="round_incr",
        #debug=debug_fixed
    )

    # Precision for result without sign
    unsigned_result_prec = fixed_point((p-1)+exp_size, 0)

    unrounded_mant_field_nomsb = Conversion(
        unrounded_mant_field,
        precision=fixed_point(p-1,0,signed=False),
        tag="unrounded_mant_field_nomsb",
        debug=debug_hex
    )

    pre_rounded_unsigned_result = Concatenation(
        res_exp_field,
        unrounded_mant_field_nomsb,
        precision=unsigned_result_prec,
        tag="pre_rounded_unsigned_result"
    )
    unsigned_result_rounded = Addition(
        pre_rounded_unsigned_result,
        round_incr,
        precision=unsigned_result_prec,
        tag="unsigned_result"
    )

    vr_out = TypeCast(
      Concatenation(
        result_sign,
        TypeCast(
            unsigned_result_rounded,
            precision=ML_StdLogicVectorFormat(p-1+exp_size)
        ),
        precision=ML_StdLogicVectorFormat(io_precision.get_bit_size())
      ),
      precision = io_precision,
      debug = debug_hex,
      tag = "vr_out"
    )

    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  def init_test_generator(self):
      """ Initialize test case generator """
      weight_map = {
          FPRandomGen.Category.SpecialValues: 0.0 if self.disable_sv_testing else 0.1,
          FPRandomGen.Category.Subnormal: 0.0 if self.disable_sub_testing else 0.2,
          FPRandomGen.Category.Normal: 0.7,
      }
      self.input_generator = FPRandomGen(self.precision, weight_map=weight_map)

  def generate_test_case(self, input_signals, io_map, index, test_range = None):
      """ specific test case generation for K1C TCA BLAU """
      rnd_mode = random.randrange(4)

      input_values = {
          "rnd_mode": rnd_mode,
          "x": self.input_generator.get_new_value()
      }
      return input_values


  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    rnd_mode_i = io_map["rnd_mode"]

    def div_numeric_emulate(vx):
        sollya_format = self.precision.get_sollya_object()
        return sollya.round(1.0 / vx, sollya_format, rnd_mode)

    rnd_mode = {
        0: sollya.RN,
        1: sollya.RU,
        2: sollya.RD,
        3: sollya.RZ
    }[rnd_mode_i]
    value_mapping = {
        is_plus_infty: lambda _: 0.0,
        is_nan: lambda _: FP_QNaN(self.precision),
        is_minus_infty: lambda _: FP_QNaN(self.precision),
        is_plus_zero: lambda _: FP_PlusInfty(self.precision),
        is_minus_zero: lambda _: FP_MinusInfty(self.precision),
        is_sv_omega: lambda op: lambda _: div_numeric_emulate(op.get_value()),
        lambda op: not(FP_SpecialValue.is_special_value(op)): div_numeric_emulate,
    }
    result = {}
    for predicate in value_mapping:
        if predicate(vx):
            result["vr_out"] = value_mapping[predicate](vx)
            return result
    Log.report(Log.Error, "no predicate fits {} in numeric_emulate\n".format(vx))

  #standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
  standard_test_cases = [
    ({"x": 2.0, "rnd_mode": 0}, None),
    ({"x": sollya.parse("0x1.24f608p0"), "rnd_mode": 0}, None),
    ({"x": 1.5, "rnd_mode": 0}, None),
  ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="new_fp_div",
        default_output_file="ml_fp_div.vhd",
        default_arg=FP_Divider.get_default_args() )
    # extra command line arguments
    arg_template.parser.add_argument(
        "--disable-sub-test", dest="disable_sub_testing", action="store_const",
        const=True, default=False,
        help="disabling generation of subnormal input during testing")
    arg_template.parser.add_argument(
        "--disable-sv-test", dest="disable_sv_testing", action="store_const",
        const=True, default=False,
        help="disabling generation of special values input during testing")

    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_div      = FP_Divider(args)

    ml_hw_div.gen_implementation()
