# -*- coding: utf-8 -*-

import sys

import sollya

# from sollya import S2, Interval, inf, sup, pi, log, exp, cos, sin, guessdegree

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable


from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.log_report import Log
from metalibm_core.utility.arg_utils import extract_option_value  

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import *
from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate


## Newton-Raphson iteration object
class NR_Iteration: 
  def __init__(self, value, approx, half_value):
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    Attributes.set_default_silent(True)

    self.square = approx * approx
    error_mult = self.square * half_value
    self.error = 0.5 - error_mult
    approx_mult = self.error * approx
    self.new_approx = approx + approx_mult 

    Attributes.unset_default_rounding_mode()
    Attributes.unset_default_silent()


  def get_new_approx(self):
    return self.new_approx

  def get_hint_rules(self, gcg, gappa_code, exact):
    pass


## propagate @p precision on @p optree on all operands with 
#  no precision (None) set, applied recursively
def propagate_format(optree, precision):
  if optree.get_precision() is None:
    optree.set_precision(precision)
    if not isinstance(optree, ML_LeafNode):
      for op_input in optree.get_inputs():
        propagate_format(op_input, precision)


def compute_sqrt(vx, init_approx, num_iter, debug_lftolx = None, precision = ML_Binary64):

    h = 0.5 * vx
    h.set_attributes(tag = "h", debug = debug_multi, silent = True, rounding_mode = ML_RoundToNearest)

    current_approx = init_approx 
    # correctly-rounded inverse computation
    num_iteration = num_iter
    inv_iteration_list = []
    for i in xrange(num_iteration):
        new_iteration = NR_Iteration(vx, current_approx, h)
        inv_iteration_list.append(new_iteration)
        current_approx = new_iteration.get_new_approx()
        current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_multi)

    final_approx = current_approx
    final_approx.set_attributes(tag = "final_approx", debug = debug_multi)

    # multiplication correction iteration
    # to get correctly rounded full square root
    Attributes.set_default_silent(True)
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    S = vx * final_approx
    S.set_attributes(tag = "S", debug = debug_multi)
    t5 = final_approx * h
    t5.set_attributes(tag = "t5", debug = debug_multi)
    H = 0.5 * final_approx
    H.set_attributes(tag = "H", debug = debug_multi)
    d = vx - S * S
    #d = FMSN(S, S, vx)
    d.set_attributes(tag = "d", debug = debug_multi)
    t6 = 0.5 - t5 * final_approx
    t6.set_attributes(tag = "t6", debug = debug_multi)
    S1 = S + d * H
    S1.set_attributes(tag = "S1", debug = debug_multi)
    H1 = H + t6 * H
    H1.set_attributes(tag = "H1", debug = debug_multi)
    #d1 = vx - S1 * S1
    d1 = FMSN(S1, S1, vx) #, clearprevious = True)
    d1.set_attributes(tag = "d1", debug = debug_multi)

    pR = FMA(d1, H1, S1)

    #t7 = 0.5 - t6 * final_approx
    #H2 = H1 + t6 * H1
    d_last = FMSN(pR, pR, vx, silent = True, tag = "d_last", debug = debug_multi)


    Attributes.unset_default_silent()
    Attributes.unset_default_rounding_mode()
    #R  = S1 + d1 * H1 

    #R = FMA(d1, H1, S1, rounding_mode = ML_GlobalRoundMode)
    R = FMA(d_last, H1, pR, rounding_mode = ML_GlobalRoundMode)

    # set precision
    propagate_format(R, precision)
    propagate_format(S1, precision)
    propagate_format(H1, precision)
    propagate_format(d1, precision)

    return R, S1, H1, d1


class ML_Sqrt(ML_Function("ml_sqrt")):
  def __init__(self, 
               arg_template = DefaultArgTemplate,
               precision = ML_Binary32, 
               accuracy = ML_CorrectlyRounded, 
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               num_iter = 3,
               fast_path_extract = True,
               target = GenericProcessor(), 
               output_file = "sqrtf.c", 
               function_name = "sqrtf",
               dot_product_enabled = False, 
               vector_size = 1, 
               language = C_Code):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    num_iter  = ArgDefault.select_value([arg_template.num_iter, num_iter])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "exp",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      vector_size = vector_size,
      language = language,
      arg_template = arg_template
    )

    self.accuracy  = accuracy
    self.precision = precision
    self.integer_precision = self.precision.get_integer_format()
    self.num_iter = num_iter

  def generate_scheme(self):
    pre_vx = self.implementation.add_input_variable("x", self.precision) 


    # local overloading of RaiseReturn operation
    def SqrtRaiseReturn(*args, **kwords):
        kwords["arg_value"] = pre_vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    ex = ExponentExtraction(pre_vx, tag = "ex", debug = debug_multi, precision = ML_Int32)
    equal_comp = Equal(Modulo(ex, 2, precision = ML_Int32), 0, precision = ML_Bool)
    even_ex = Select(equal_comp, ex, ex - 1, tag = "even_ex", debug = debug_multi, precision = ML_Int32)
    pre_scale_factor = ExponentInsertion(-(even_ex/2), tag = "pre_scale_factor", debug = debug_multi, precision = self.precision) 
    pre_scale_mult = (pre_vx * pre_scale_factor)
    pre_scale_mult.set_attributes(silent = True, rounding_mode = ML_RoundToNearest, precision = self.precision)
    slow_vx = pre_scale_mult * pre_scale_factor
    slow_vx.set_attributes(tag = "vx", silent = True, rounding_mode = ML_RoundToNearest, debug = debug_multi, precision = self.precision)
    slow_scale_factor = ExponentInsertion(even_ex / 2, tag = "scale_factor", debug = debug_multi, precision = self.precision)

    vx = slow_vx
    scale_factor = slow_scale_factor

    # computing the inverse square root
    init_approx = None
    # forcing vx precision to make processor support test
    vx.set_precision(self.precision)
    init_approx_precision = InverseSquareRootSeed(vx, precision = self.precision, tag = "seed", debug = debug_multi)
    if not self.processor.is_supported_operation(init_approx_precision):
        if self.precision != ML_Binary32:
            px = Conversion(vx, precision = ML_Binary32, tag = "px", debug = debug_multi) 
            init_approx_fp32 = Conversion(InverseSquareRootSeed(px, precision = ML_Binary32, tag = "seed_fp32", debug = debug_multi), precision = self.precision, tag = "seed_ext", debug = debug_multi)
            if not self.processor.is_supported_operation(init_approx_fp32):
                Log.report(Log.Error, "The target %s does not implement inverse square root seed" % self.processor)
            else:
                init_approx = init_approx_fp32
        else:
            Log.report(Log.Error, "The target %s does not implement inverse square root seed" % self.processor)
    else:
        init_approx = init_approx_precision

    result, S1, H1, d1 = compute_sqrt(vx, init_approx, self.num_iter, precision = self.precision)
    result = result * scale_factor
    result.set_attributes(tag = "result", debug = debug_multi, clearprevious = True)


    def bit_match(fp_optree, bit_id, likely = False, ** kwords):
        # return NotEqual(BitLogicAnd(TypeCast(fp_optree, precision = self.integer_precision), 1 << bit_id), 0, likely = likely, **kwords)
        return BitLogicAnd(TypeCast(fp_optree, precision = self.integer_precision), 1 << bit_id)

    x_qnan = Test(pre_vx, specifier = Test.IsQuietNaN, likely = False, tag = "x_qnan", debug = debug_multi, precision = ML_Bool)
    x_zero = Test(pre_vx, specifier = Test.IsZero, likely = False, tag = "x_zero", debug = debug_multi, precision = ML_Bool)
    x_neg = Comparison(pre_vx, 0, specifier = Comparison.Less, likely = False, precision = ML_Bool)
    x_nan = Test(pre_vx, specifier = Test.IsNaN, likely = False, tag = "x_nan", debug = debug_multi, precision = ML_Bool)
    x_nan_or_neg = LogicalOr(x_nan, x_neg, precision = ML_Bool)
    x_plus_inf = Test(pre_vx, specifier = Test.IsPositiveInfty, likely = False, tag = "x_plus_inf", debug = debug_multi, precision = ML_Bool)

    return_neg = Statement(ClearException(), SqrtRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    specific_case_bit = bit_match(init_approx, 0, tag = "specific_case_bit")

    # x inf and y inf 
    pre_scheme = ConditionBlock(NotEqual(specific_case_bit, 0, tag = "specific_case", debug = debug_multi, likely = True),
        Return(result),
        ConditionBlock(x_zero,
            Statement(ClearException(), Return(pre_vx)),
            ConditionBlock(x_nan_or_neg,
                ConditionBlock(x_qnan, 
                    Statement(ClearException(), Return(FP_QNaN(self.precision))),
                    Statement(ClearException(), SqrtRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))
                ),
                ConditionBlock(x_plus_inf,
                    Return(pre_vx),
                    Statement(
                        #ConditionBlock(Comparison(d_last, 0, specifier = Comparison.NotEqual, likely = True),
                        #    Raise(ML_FPE_Inexact)
                        #),
                        Return(result)
                    )
                )
            )
        )
    )
    scheme = None
    rnd_mode = GetRndMode()
    scheme = Statement(rnd_mode, SetRndMode(ML_RoundToNearest, precision = ML_Void), S1, H1, d1, SetRndMode(rnd_mode, precision = ML_Void), result, pre_scheme)

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
      """ generate the emulation code for ML_Log2 functions
          mpfr_x is a mpfr_t variable which should have the right precision
          mpfr_rnd is the rounding mode
      """
      emulate_func_name = "mpfr_sqrt"
      emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
      emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
      mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

      return mpfr_call

  def numeric_emulate(self, input):
        return sollya.sqrt(input)



if __name__ == "__main__":
  # auto-test
  # num_iter        = int(extract_option_value("--num-iter", "3"))

  arg_template = ML_NewArgTemplate(default_function_name = "new_sqrt", default_output_file = "new_sqrt.c")
  arg_template.parser.add_argument("--num-iter", dest = "num_iter", action = "store", default = ArgDefault(3), help = "number of Newton-Raphson iterations")
  args = parse_arg_index_list = arg_template.arg_extraction()


  ml_sqrt  = ML_Sqrt(args)
  ml_sqrt.gen_implementation()
  
