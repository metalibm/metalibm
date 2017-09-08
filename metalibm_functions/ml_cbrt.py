# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD
try:
    from sollya import cbrt
except ImportError:
    from sollya_extra_functions import cbrt
from sollya import parse as sollya_parse

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



class ML_Cbrt(ML_Function("ml_cbrt")):
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = GenericProcessor(), 
             output_file = "my_cbrt.c", 
             function_name = "my_cbrt",
             language = C_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "cbrt",
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

  def generate_scheme(self):
    # declaring main input variable
    vx = self.implementation.add_input_variable("x", self.precision) 

    # declaring approximation parameters
    index_size = 6
    num_iteration = 8

    Log.set_dump_stdout(True)

    Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
    if self.debug_flag: 
        Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)



    def cbrt_newton_iteration(current_approx, input_value, input_inverse):
      # Cubic root of A is approximated by a Newton-Raphson iteration
      # on f(x) = 1 - A / x^3
      # x_n+1 = 4/3 * x_n - x_n^4 / (3 * A)
      # x_n+1 = 1/3 * (x_n * (1 - x_n^3/A) + x_n)

      approx_triple = Multiplication(current_approx, Multiplication(current_approx, current_approx))

      diff      = FMSN(approx_triple, input_inverse, Constant(1, precision = self.precision))
      injection = FMA(
        Multiplication(
          current_approx, 
          Constant(1/3.0, precision = self.precision),
        ),
        diff, current_approx)

      new_approx = injection

      return new_approx


    reduced_vx = MantissaExtraction(vx, precision = self.precision)

    int_precision = self.precision.get_integer_format()


    cbrt_approx_table = ML_NewTable(dimensions = [2**index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cbrt_approx_table"))
    for i in range(2**index_size):
      input_value = 1 + i / SollyaObject(2**index_size) 

      cbrt_approx = cbrt(input_value)
      cbrt_approx_table[i][0] = round(cbrt_approx, self.precision.get_sollya_object(), RN)

    # approximation of cbrt(1), cbrt(2), cbrt(4)
    cbrt_mod_table = ML_NewTable(dimensions = [3, 1], storage_precision = self.precision, tag = self.uniquify_name("cbrt_mod_table"))
    for i in range(3):
      input_value = SollyaObject(2)**i
      cbrt_mod_table[i][0] = round(cbrt(input_value), self.precision.get_sollya_object(), RN)

    vx_int = TypeCast(reduced_vx, precision = int_precision)
    mask = BitLogicRightShift(vx_int, self.precision.get_precision() - index_size, precision = int_precision)
    mask = BitLogicAnd(mask, Constant(2**index_size - 1, precision = int_precision), precision = int_precision, tag = "table_index")
    table_index = mask

    exp_vx = ExponentExtraction(vx, precision = ML_Int32, tag = "exp_vx")
    exp_vx_third = Division(exp_vx, Constant(3, precision = ML_Int32), precision = ML_Int32, tag = "exp_vx_third")
    exp_vx_mod   = Modulo(exp_vx, Constant(3, precision = ML_Int32), precision = ML_Int32, tag = "exp_vx_mod")

    cbrt_mod = TableLoad(cbrt_mod_table, exp_vx_mod, Constant(0), tag = "cbrt_mod")

    init_approx = Multiplication(
      Multiplication(
        # approx cbrt(mantissa)
        TableLoad(cbrt_approx_table, table_index, Constant(0, precision = ML_Int32)),
        # approx cbrt(2^(e%3))
        cbrt_mod,
        precision = self.precision
      ),
      # 2^(e/3)
      ExponentInsertion(exp_vx_third, precision = self.precision),
      precision = self.precision
    )

    inverse_red_vx = Division(Constant(1, precision = self.precision), reduced_vx)
    inverse_vx = Division(Constant(1, precision = self.precision), vx)

    current_approx = init_approx

    for i in xrange(num_iteration):
      #current_approx = cbrt_newton_iteration(current_approx, reduced_vx, inverse_red_vx) 
      current_approx = cbrt_newton_iteration(current_approx, vx, inverse_vx) 

    result = current_approx

    # last iteration
    ext_precision = ML_DoubleDouble
    xn_2 = Multiplication(current_approx, current_approx, precision = ext_precision)
    xn_3 = Multiplication(current_approx, xn_2, precision = ext_precision)

    FourThird = Constant(4/SollyaObject(3), precision = ext_precision)

    # main scheme
    Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    scheme = Statement(
                Return(
                    result
                  ))

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cbrt"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call

  def numeric_emulate(self, input_value):
    return cbrt(input_value)

  standard_test_cases =[sollya_parse(x) for x in  ["1.1", "1.5"]]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_cosh", default_output_file = "new_cosh.c" )
    # argument extraction 
    args = arg_template.arg_extraction()

    ml_cbrt          = ML_Cbrt(args)

    ml_cbrt.gen_implementation()
