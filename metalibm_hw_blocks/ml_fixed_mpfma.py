# -*- coding: utf-8 -*-

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

class FP_FIXED_MPFMA(ML_Entity("fp_fixed_mpfma")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = VHDLBackend(), 
             output_file = "fp_fixed_mpfma.vhd", 
             entity_name = "fp_fixed_mpfma",
             language = VHDL_Code,
             vector_size = 1,
             extra_digit = 0):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "fp_fixed_mpfma",
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
    # number of extra bits to add to the accumulator fixed precision
    self.extra_digit = extra_digit

  def generate_scheme(self):
    ## Generate Fused multiply and add comput <x> . <y> + <z>

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

    vx = self.implementation.add_input_signal("x", io_precision) 
    vy = self.implementation.add_input_signal("y", io_precision) 

    acc = self.implementation.add_input_signal("acc", acc_prec)
    sign_acc = CopySign(acc, precision = ML_StdLogic)

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
    
    exp_vx = ExponentExtraction(vx, precision = exp_vx_precision)
    exp_vy = ExponentExtraction(vy, precision = exp_vy_precision)

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
    sign_xy = BitLogicXor(sign_vx, sign_vy, precision = ML_StdLogic, tag = "sign_xy", debug = ML_Debug(display_format = "-radix 2"))
    effective_op = BitLogicXor(sign_xy, sign_acc, precision = ML_StdLogic, tag = "effective_op", debug = ML_Debug(display_format = "-radix 2"))

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
    ) + 2
    exp_precision_ext = ML_StdLogicVectorFormat(exp_precision_ext_size)

    # static accumulator exponent
    exp_acc = Constant(acc_msb_index, precision = exp_precision_ext) 

    # Y is first aligned offset = max(o+L_y,q) + 2 bits to the left of x 
    # and then shifted right by 
    # exp_diff = exp_x - exp_y + offset
    # exp_vx in [emin, emax]
    # exp_vx - exp_vx + p +2 in [emin-emax + p + 2, emax - emin + p + 2]
    exp_diff = Subtraction(
                Addition(
                  Addition(
                    zext(exp_vy, exp_precision_ext_size - vy_precision.get_exponent_size()), 
                    zext(exp_vx, exp_precision_ext_size - vx_precision.get_exponent_size()), 
                    precision = exp_precision_ext
                  ),
                  Constant(exp_bias + prod_exp_offset, precision = exp_precision_ext),
                  precision = exp_precision_ext
                ),
                exp_acc,
                precision = exp_precision_ext,
                tag = "exp_diff",
                debug = debug_std
    )
    signed_exp_diff = SignCast(
      exp_diff, 
      specifier = SignCast.Signed, 
      precision = exp_precision_ext
    )
    datapath_full_width = acc_width
    # the maximum exp diff is the size of the datapath
    # minus the bit size of the product
    max_exp_diff = datapath_full_width - (p + q)
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

    prod_prec = ML_StdLogicVectorFormat(p+q)
    prod = Multiplication(
      mant_vx,
      mant_vy,
      precision = prod_prec,
      tag = "prod",
      debug = debug_std
    )

    mant_ext_size = datapath_full_width - (p+q)
    shift_prec = ML_StdLogicVectorFormat(datapath_full_width)
    shifted_prod = BitLogicRightShift(rzext(prod, mant_ext_size), mant_shift, precision = shift_prec, tag = "shifted_prod", debug = debug_std)

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

    mant_add = Addition(
                 shifted_prod_op,
                 acc,
                 precision = acc_prec,
                 tag = "mant_add",
                 debug = ML_Debug(display_format = " -radix 2")
              )

    self.implementation.add_output_signal("vr_acc", mant_add)

    return [self.implementation]

  def numeric_emulate(self, io_map):
    vx = io_map["x"]
    vy = io_map["y"]
    vz = io_map["z"]
    result = {}
    result["vr_out"] = sollya.round(vx * vy + vz, self.precision.get_sollya_object(), sollya.RN)
    return result

  # standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
  standard_test_cases = [
    #({"x": 2.0, "y": 4.0, "z": 16.0}, None),
    ({
      "y": ML_Binary16.get_value_from_integer_coding("2cdc", base = 16),
      "x": ML_Binary16.get_value_from_integer_coding("1231", base = 16),
      "z": ML_Binary16.get_value_from_integer_coding("5b5e", base = 16),
    }, None),
    #({
    #  "y": ML_Binary64.get_value_from_integer_coding("47d273e91e2c9048", base = 16),
    #  "x": ML_Binary64.get_value_from_integer_coding("c7eea5670485a5ec", base = 16)
    #}, None),
    #({
    #  "y": ML_Binary64.get_value_from_integer_coding("75164a1df94cd488", base = 16),
    #  "x": ML_Binary64.get_value_from_integer_coding("5a7567b08508e5b4", base = 16)
    #}, None)
  ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_fp_fixed_mpfma", default_output_file = "ml_fp_fixed_mpfma.vhd" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_fp_fixed_mpfma      = FP_FIXED_MPFMA(args)

    ml_hw_fp_fixed_mpfma.gen_implementation()
