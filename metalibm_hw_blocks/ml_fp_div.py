# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD, cbrt
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
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

## Helper for debug enabling
debug_std          = ML_Debug(display_format = " -radix 2 ")
debug_hex          = ML_Debug(display_format = " -radix 16 ")
debug_dec          = ML_Debug(display_format = " -radix 10 ")
debug_dec_unsigned = ML_Debug(display_format = " -decimal -unsigned ")


## debug pre-process function for
#  fixed-point value
def fixed_debug_pre_process(value_name, optree):
  fixed_prec = optree.get_precision()
  signed_attr = "-signed" if fixed_prec.get_signed() else "-unsigned"
  return "echo [get_fixed_value [examine -value {signed_attr} {value}] {weight}]".format(signed_attr = signed_attr, value = value_name, weight = -fixed_prec.get_frac_size())

## Debug attributes specific for Fixed-Point values
debug_fixed = ML_AdvancedDebug(pre_process = fixed_debug_pre_process)

class FP_Divider(ML_Entity("fp_div")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             ):

    # initializing base class
    ML_EntityBasis.__init__(self, 
      arg_template = arg_template
    )

  @staticmethod
  def get_default_args(width = 32):
    return DefaultEntityArgTemplate( 
             precision = ML_Binary32, 
             debug_flag = False, 
             target = VHDLBackend(), 
             output_file = "my_fp_div.vhd", 
             entity_name = "my_fp_div",
             language = VHDL_Code
           )

  def generate_scheme(self):

    def get_virtual_cst(prec, value, language):
      return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))

    ## convert @p value from an input floating-point precision
    #  @p in_precision to an output support format @p out_precision
    io_precision = VirtualFormat(base_format = self.precision, support_format = ML_StdLogicVectorFormat(self.precision.get_bit_size()), get_cst = get_virtual_cst)

    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision) 
    vy = self.implementation.add_input_signal("y", io_precision) 

    vx_precision = self.precision
    vy_precision = self.precision

    p = vx_precision.get_mantissa_size()
    assert p == vy_precision.get_mantissa_size()

    exp_vx_precision     = ML_StdLogicVectorFormat(vx_precision.get_exponent_size())
    exp_vy_precision     = ML_StdLogicVectorFormat(vy_precision.get_exponent_size())

    mant_vx_precision    = ML_StdLogicVectorFormat(p)
    mant_vy_precision    = ML_StdLogicVectorFormat(p)

    # mantissa extraction
    mant_vx = MantissaExtraction(vx, precision = mant_vx_precision, tag = "mant_vx")
    mant_vy = MantissaExtraction(vy, precision = mant_vy_precision, tag = "mant_vy")
    # exponent extraction 
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision, tag = "exp_vx", debug = debug_dec)
    exp_vy = ExponentExtraction(vy, precision = exp_vy_precision, tag = "exp_vy", debug = debug_dec)

    approx_index_size = 6

    approx_precision = RTL_FixedPointFormat(
      2, approx_index_size,
      support_format = ML_StdLogicVectorFormat(approx_index_size + 2),
    )
    #approx_precision_unsigned = RTL_FixedPointFormat(
    #  2, approx_index_size,
    #  support_format = ML_StdLogicVectorFormat(approx_index_size + 1),
    #  signed = False
    #)

    # selecting table index from input mantissa MSBs
    tab_index = SubSignalSelection(mant_vx, p-2 - approx_index_size +1, p-2, tag = "tab_index")

    # declaring reciprocal approximation table
    inv_approx_table = ML_NewTable(dimensions = [2**approx_index_size], storage_precision = approx_precision, tag = "inv_approx_table")
    for i in xrange(2**approx_index_size):
      num_input = 1 + i * S2**-approx_index_size
      table_value = io_precision.get_base_format().round_sollya_object(1 / num_input)
      inv_approx_table[i] = table_value

    # extracting initial reciprocal approximation
    inv_approx_value = TableLoad(inv_approx_table, tab_index, precision = approx_precision, tag = "inv_approx_value", debug = debug_fixed)


    #inv_approx_value = TypeCast(inv_approx_value, precision = approx_precision)
    pre_it0_input = zext(SubSignalSelection(mant_vx, p-1 - approx_index_size , p-1, tag = "it0_input"), 1)
    it0_input = TypeCast(pre_it0_input, precision = approx_precision, tag = "it0_input", debug = debug_fixed)

    #it0_input = TypeCast(it0_input, precision = approx_precision_unsigned, tag = "it0_input", debug = debug_fixed)

    def generate_NR_iteration(recp_input, previous_approx, (mult_int_size, mult_frac_size), (approx_int_size, approx_frac_size)):
      # creating required formats
      it_mult_precision = RTL_FixedPointFormat(
        mult_int_size, mult_frac_size,
        support_format = ML_StdLogicVectorFormat(mult_int_size + mult_frac_size)
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
        tag = "it_mult"
      )
      it_error = Subtraction(
        Constant(1, precision = it_mult_precision),
        it_mult,
        precision = it_mult_precision,
        tag = "it_error",
        debug = debug_fixed
      )
      # computing new approximation
      approx_mult = Multiplication(
        it_error,
        previous_approx,
        precision = new_approx_precision,
        tag = "approx_mult",
        debug = debug_fixed
      )
      new_approx = Addition(
        previous_approx,
        approx_mult,
        precision = new_approx_precision, 
        tag = "new_approx",
        debug = debug_fixed
      )
      return new_approx

    it0_mult_precision = RTL_FixedPointFormat(
      3, approx_index_size * 2,
      support_format = ML_StdLogicVectorFormat(3 + 2 * approx_index_size),
    )

    it0_mult = Multiplication(it0_input, inv_approx_value, precision = it0_mult_precision, tag = "it0_mult", debug = debug_fixed)
    it0_error = Subtraction(
      Constant(1, precision = it0_mult_precision),
      it0_mult,
      precision = it0_mult_precision,
      tag = "it0_error",
      debug = debug_fixed
    )
    it1_approx_precision = RTL_FixedPointFormat(
      4, 3 * approx_index_size,
      support_format = ML_StdLogicVectorFormat(4 + 3 * approx_index_size),
    )
    it1_mult = Multiplication(
      it0_error,
      inv_approx_value,
      precision = it1_approx_precision 
    )

    it1_approx = Addition(
      inv_approx_value,
      it1_mult,
      precision = it1_approx_precision,
      tag = "it1_approx",
      debug = debug_fixed
    )

    final_approx = generate_NR_iteration(
      it0_input,
      it1_approx,
      (2, approx_index_size * 3),
      (2, approx_index_size * 3)
    )

    last_it_precision = RTL_FixedPointFormat(
      2,
      p - 1,
      support_format = ML_StdLogicVectorFormat(2 + p - 1)
    )

    pre_last_it_input = zext(mant_vx, 1)
    last_it_input = TypeCast(pre_last_it_input, precision = last_it_precision, tag = "last_it_input", debug = debug_fixed)

    final_approx = generate_NR_iteration(
      last_it_input,
      final_approx,
      (2, approx_index_size * 3 + p - 1),
      (2, approx_index_size * 3 + p - 1)
    )

    final_approx = generate_NR_iteration(
      last_it_input,
      final_approx,
      (2, approx_index_size * 3 + p - 1),
      (2, approx_index_size * 3 + p - 1)
    )

    final_approx = generate_NR_iteration(
      last_it_input,
      final_approx,
      (2, 2 * p),
      (2, 2 * p)
    )

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
        precision = exp_op_precision
      ),
      SignCast(
        zext(exp_vx, 2), 
        specifier = SignCast.Unsigned,
        precision = exp_op_precision,
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
        precision = exp_vx.get_precision(),
        specifier = SignCast.Unsigned
      ),
      SignCast(
        Select(
          Comparison(not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
          Constant(0, precision = exp_vx_precision),
          Constant(-1, precision = exp_vx_precision),
          precision = exp_vx_precision
        ),
        precision = exp_vx_precision,
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
        precision = exp_mant_precision
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
    #vr_out = FloatBuild(
    #  res_sign,
    #  res_exp,
    #  res_mant_field,
    #  precision = io_precision,
    #  tag = "vr_out",
    #  debug = debug_hex
    #)

    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    vy = io_map["y"]
    result = {}
    result["vr_out"] = sollya.round(1.0 / vx, self.precision.get_sollya_object(), sollya.RN)
    return result

  #standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
  standard_test_cases = [({"x": 1.5, "y": 0.0}, None)]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_fp_div", default_output_file = "ml_fp_div.vhd", default_arg = FP_Divider.get_default_args() )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_div      = FP_Divider(args)

    ml_hw_div.gen_implementation()
