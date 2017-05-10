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

    mant_vx_precision    = ML_StdLogicVectorFormat(p-1)
    mant_vy_precision    = ML_StdLogicVectorFormat(p-1)

    # mantissa extraction
    mant_vx = MantissaExtraction(vx, precision = mant_vx_precision, tag = "mant_vx")
    mant_vy = MantissaExtraction(vy, precision = mant_vy_precision, tag = "mant_vy")
    # exponent extraction 
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision, tag = "exp_vx", debug = debug_dec)
    exp_vy = ExponentExtraction(vy, precision = exp_vy_precision, tag = "exp_vy", debug = debug_dec)

    approx_index_size = 6

    approx_precision = RTL_FixedPointFormat(
      1, approx_index_size,
      support_format = ML_StdLogicVectorFormat(approx_index_size + 1),
    )
    approx_precision_unsigned = RTL_FixedPointFormat(
      1, approx_index_size,
      support_format = ML_StdLogicVectorFormat(approx_index_size + 1),
      signed = False
    )

    tab_index = SubSignalSelection(mant_vx, p-2 - approx_index_size +1, p-2, tag = "tab_index")

    # declaring reciprocal approximation table
    inv_approx_table = ML_NewTable(dimensions = [2**approx_index_size], storage_precision = approx_precision, tag = "inv_approx_table")
    for i in xrange(2**approx_index_size):
      num_input = 1 + i * S2**-approx_index_size
      table_value = io_precision.get_base_format().round_sollya_object(1 / num_input)
      inv_approx_table[i] = table_value



    inv_approx_value = TableLoad(inv_approx_table, tab_index, precision = approx_precision_unsigned, tag = "inv_approx_value", debug = debug_fixed)

    #inv_approx_value = TypeCast(inv_approx_value, precision = approx_precision)
    it0_input = SubSignalSelection(mant_vx, p-1 - approx_index_size , p-1, tag = "it0_input")
    it0_input = TypeCast(it0_input, precision = approx_precision_unsigned, tag = "it0_input", debug = debug_fixed)

    it0_mult_precision = RTL_FixedPointFormat(
      2, approx_index_size * 2,
      support_format = ML_StdLogicVectorFormat(2 + 2 * approx_index_size),
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
      support_format = ML_StdLogicVectorFormat(3 + 3 * approx_index_size),
    )

    it1_approx = Addition(
      inv_approx_value,
      Multiplication(
        it0_error,
        inv_approx_value,
        precision = it1_approx_precision 
      ),
      precision = it1_approx_precision,
      tag = "it1_approx",
      debug = debug_fixed
    )

    cst_index = Constant(3 * approx_index_size, precision = ML_Integer)
    not_decrement = TypeCast(it1_approx, precision = ML_StdLogicVectorFormat(it1_approx_precision.get_bit_size()))[cst_index] 
    not_decrement.set_attributes(precision = ML_StdLogic, tag = "not_decrement")
    logic_1 = Constant(1, precision = ML_StdLogic)

    pre_mant_size = min(self.precision.get_mantissa_size(), 3 * approx_index_size) - 1
    result = Select(
      Comparison( not_decrement, logic_1, specifier = Comparison.Equal, precision = ML_Bool),
      SubSignalSelection(
        TypeCast(
          it1_approx, 
          precision = ML_StdLogicVectorFormat(it1_approx.get_precision().get_bit_size())
        ),
        3 * approx_index_size - pre_mant_size,
        3 * approx_index_size - 1
      ),
      SubSignalSelection(
        TypeCast(
          it1_approx, 
          precision = ML_StdLogicVectorFormat(it1_approx.get_precision().get_bit_size())
        ),
        3 * approx_index_size - pre_mant_size - 1,
        3 * approx_index_size - 2
      ),
      precision = ML_StdLogicVectorFormat(pre_mant_size)
    )
    # if mantissa require extension
    if pre_mant_size < self.precision.get_mantissa_size() - 1:
      result = rzext(result, self.precision.get_mantissa_size() - 1 - pre_mant_size) 

    res_mant_field = result

    res_exp = Addition(
      SignCast(
        exp_vx, 
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
      tag = "result_exp"
    )

    res_sign = CopySign(vx, precision = ML_StdLogic)

    vr_out = FloatBuild(
      res_sign,
      res_exp,
      res_mant_field,
      precision = io_precision
    )

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
