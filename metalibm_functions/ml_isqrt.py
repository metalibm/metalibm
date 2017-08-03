
 # -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

## Newton-Raphson iteration object
class NR_Iteration:
  def __init__(self, value, approx, half_value):
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    Attributes.set_default_silent(True)

    self.square = approx * approx
    mult = FMSN(half_value, self.square, 0.5)
    self.new_approx =  FMA(approx, mult, approx)

    Attributes.unset_default_rounding_mode()
    Attributes.unset_default_silent()


  def get_new_approx(self):
    return self.new_approx

## propagate @p precision on @p optree on all operands with
#  no precision (None) set, applied recursively
def propagate_format(optree, precision):
  if optree.get_precision() is None:
    optree.set_precision(precision)
    if not isinstance(optree, ML_LeafNode):
      for op_input in optree.get_inputs():
        propagate_format(op_input, precision)


def compute_isqrt(vx, init_approx, num_iter, debug_lftolx = None, precision = ML_Binary64):

    h = 0.5 * vx
    h.set_attributes(tag = "h", debug = debug_multi, silent = True, rounding_mode = ML_RoundToNearest)

    current_approx = init_approx
    # correctly-rounded inverse computation

    for i in xrange(num_iter):
        new_iteration = NR_Iteration(vx, current_approx, h)
        current_approx = new_iteration.get_new_approx()
        current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_multi)

    final_approx = current_approx
    final_approx.set_attributes(tag = "final_approx", debug = debug_multi)

    # multiplication correction iteration
    # to get correctly rounded full square root
    Attributes.set_default_silent(True)
    Attributes.set_default_rounding_mode(ML_RoundToNearest)

    return final_approx



class ML_Isqrt(ML_Function("ml_isqrt")):
  def __init__(self,
             arg_template = DefaultArgTemplate,
             precision = ML_Binary32,
             accuracy  = ML_CorrectlyRounded,
             libm_compliant = True,
             debug_flag = False,
             fuse_fma = True,
             fast_path_extract = True,
             target = GenericProcessor(),
             output_file = "my_isqrt.c",
             function_name = "my_isqrt",
             language = C_Code,
             vector_size = 1,
             num_iter = 3):

    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    num_iter  = ArgDefault.select_value([arg_template.num_iter, num_iter])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self,
      base_name = "isqrt",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      language = language,
      vector_size = vector_size,
      arg_template = arg_template
    )

    self.accuracy  = accuracy
    self.precision = precision
    self.num_iter = num_iter

  def generate_scheme(self):
    # declaring target and instantiating optimization engine

    vx = self.implementation.add_input_variable("x", self.precision)
    vx.set_attributes(precision = self.precision, tag = "vx", debug =debug_multi)
    Log.set_dump_stdout(True)

    Log.report(Log.Info, "\033[33;1m Generating implementation scheme \033[0m")
    if self.debug_flag:
        Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

    # local overloading of RaiseReturn operation
    def SqrtRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)


    test_zero = Comparison(vx, 0, specifier = Comparison.Equal, likely = False, debug = debug_multi, tag = "Is_Zero", precision = ML_Bool)
    test_NaN = Test(vx, specifier = Test.IsNaN, likely = False, debug = debug_multi, tag = "is_NaN", precision = ML_Bool)
    test_negative = Comparison(vx, 0, specifier = Comparison.Less, debug = debug_multi, tag = "is_Negative", precision = ML_Bool, likely = False)
    test_inf = Test(vx, specifier = Test.IsInfty, likely = False, debug = debug_multi, tag = "is_Inf", precision = ML_Bool)
    test_NaN_or_Neg = LogicalOr(test_NaN, test_negative, precision = ML_Bool)

    return_PosZero = Statement(Return(FP_PlusInfty(self.precision)))
    return_NaN_or_neg = Statement(Return(FP_QNaN(self.precision)))
    return_inf = Statement(Return(FP_PlusZero(self.precision)))


    NR_init = InverseSquareRootSeed(vx, precision = self.precision, tag = "sqrt_seed", debug = debug_multi)

    result = compute_isqrt(vx, NR_init, int(self.num_iter), precision = self.precision)

    scheme = ConditionBlock(
                test_zero,
                return_PosZero,
                ConditionBlock(
                    test_NaN_or_Neg,
                    return_NaN_or_neg,
                    ConditionBlock(
                        test_inf,
                        return_inf,
                        Return(result)
                    )
                )
            )

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
      """ generate the emulation code for ML_Log2 functions
          mpfr_x is a mpfr_t variable which should have the right precision
          mpfr_rnd is the rounding mode
      """
      emulate_func_name = "mpfr_isqrt"
      emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"])
      emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
      mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

      return mpfr_call

  def numeric_emulate(self, input):
        return 1/sollya.sqrt(input)


  standard_test_cases = [(1.651028399744791652636877188342623412609100341796875,)] # [sollya.parse(x)] for x in  ["+0.0", "-1*0.0", "2.0"]]

if __name__ == "__main__":

  arg_template = ML_NewArgTemplate(default_function_name = "new_isqrt", default_output_file = "new_isqrt.c")
  arg_template.parser.add_argument("--num-iter", dest = "num_iter", action = "store", default = ArgDefault(3), help = "number of Newton-Raphson iterations")
  args = parse_arg_index_list = arg_template.arg_extraction()


  ml_isqrt  = ML_Isqrt(args)
  ml_isqrt.gen_implementation()

