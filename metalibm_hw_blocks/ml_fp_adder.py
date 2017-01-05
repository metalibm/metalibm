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

debug_std = ML_Debug(display_format = " -radix 2 ")
debug_dec = ML_Debug(display_format = " -radix 10 ")

class FP_Adder(ML_Entity("fp_adder")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = VHDLBackend(), 
             output_file = "fp_adder.vhd", 
             entity_name = "fp_adder",
             language = VHDL_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "fp_adder",
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
    reset = self.implementation.add_input_signal("reset", ML_StdLogic)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", io_precision) 
    vy = self.implementation.add_input_signal("y", io_precision) 

    p = self.precision.get_mantissa_size()

    # vx must be aligned with vy
    # the largest shit amount (in absolute value) is precision + 2
    # (1 guard bit and 1 rounding bit)
    exp_precision     = ML_StdLogicVectorFormat(self.precision.get_exponent_size())

    mant_precision    = ML_StdLogicVectorFormat(self.precision.get_field_size())

    mant_vx = MantissaExtraction(vx, precision = mant_precision)
    mant_vy = MantissaExtraction(vy, precision = mant_precision)
    
    exp_vx = ExponentExtraction(vx, precision = exp_precision)
    exp_vy = ExponentExtraction(vy, precision = exp_precision)

    sign_vx = CopySign(vx, precision = ML_StdLogic)
    sign_vy = CopySign(vy, precision = ML_StdLogic)

    # determining if the operation is an addition (effective_op = '0')
    # or a subtraction (effective_op = '1')
    effective_op = BitLogicXor(sign_vx, sign_vy, precision = ML_StdLogic, tag = "effective_op", debug = ML_Debug(display_format = "-radix 2"))

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
    exp_precision_ext = ML_StdLogicVectorFormat(self.precision.get_exponent_size() + 2)
    # Y is first aligned p+2 bit to the left of x 
    # and then shifted right by 
    # exp_diff = exp_x - exp_y + precision + 2
    # exp_vx in [emin, emax]
    # exp_vx - exp_vx + p +2 in [emin-emax + p + 2, emax - emin + p + 2]
    exp_diff = Subtraction(
                Addition(
                  zext(exp_vx, 2), 
                  Constant(exp_bias, precision = exp_precision_ext),
                  precision = exp_precision_ext
                ),
                zext(exp_vy, 2), 
                precision = exp_precision_ext,
                tag = "exp_diff"
    )
    exp_diff_lt_0 = Comparison(exp_diff, Constant(0, precision = exp_precision_ext), specifier = Comparison.Less, precision = ML_Bool)
    exp_diff_gt_2pp4 = Comparison(exp_diff, Constant(2*p+4, precision = exp_precision_ext), specifier = Comparison.Greater, precision = ML_Bool)

    shift_amount_prec = ML_StdLogicVectorFormat(int(floor(log2(2*p+4))+1))

    mant_shift = Select(
      exp_diff_lt_0,
      Constant(0, precision = shift_amount_prec),
      Select(
        exp_diff_gt_2pp4,
        Constant(2*p+4, precision = shift_amount_prec),
        Truncate(exp_diff, precision = shift_amount_prec),
        precision = shift_amount_prec
      ),
      precision = shift_amount_prec,
      tag = "mant_shift",
      debug = ML_Debug(display_format = "-radix 10")
    )

    mant_ext_size = 2*p+4
    shift_prec = ML_StdLogicVectorFormat(3*p+4)
    shifted_mant_vy = BitLogicRightShift(rzext(mant_vy, mant_ext_size), mant_shift, precision = shift_prec, tag = "shifted_mant_vy", debug = debug_std)
    mant_vx_ext = zext(rzext(mant_vx, p+2), p+2+1)

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
      Negation(mant_add, precision = add_prec, tag = "neg_mant_add"),
      mant_add,
      precision = add_prec,
      tag = "mant_add_abs"
    )

    res_sign = BitLogicXor(add_is_negative, sign_vy, precision = ML_StdLogic, tag = "res_sign")

    # Precision for leading zero count
    lzc_width = int(floor(log2(3*p+5)) + 1)
    lzc_prec = ML_StdLogicVectorFormat(lzc_width)

    lzc_args = ML_LeadingZeroCounter.get_default_args(width = (3*p+5))
    LZC_entity = ML_LeadingZeroCounter(lzc_args)
    lzc_entity_list = LZC_entity.generate_scheme()
    lzc_implementation = LZC_entity.get_implementation()

    lzc_component = lzc_implementation.get_component_object()

    #lzc_in = SubSignalSelection(mant_add, p+1, 2*p+3)
    lzc_in = mant_add_abs # SubSignalSelection(mant_add_abs, 0, 3*p+3, precision = ML_StdLogicVectorFormat(3*p+4))

    add_lzc = Signal("add_lzc", precision = lzc_prec, var_type = Signal.Local, debug = debug_dec)
    add_lzc = PlaceHolder(add_lzc, lzc_component(io_map = {"x": lzc_in, "vr_out": add_lzc}))

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

    rounded_mant = Addition(
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

    res_exp_prec_size = self.precision.get_exponent_size() + 2
    res_exp_prec = ML_StdLogicVectorFormat(res_exp_prec_size)

    res_exp_ext = Addition(
      Subtraction(
        Addition(
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

    res_exp = Truncate(res_exp_ext, precision = ML_StdLogicVectorFormat(self.precision.get_exponent_size()), tag = "res_exp", debug = debug_dec)

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
    result["vr_out"] = sollya.round(vx + vy, self.precision.get_sollya_object(), sollya.RN)
    return result

  standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_fp_adder", default_output_file = "ml_fp_adder.vhd" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_adder      = FP_Adder(args)

    ml_hw_adder.gen_implementation()
