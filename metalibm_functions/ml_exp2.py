# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
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



class ML_Exp2(ML_Function("ml_exp2")):
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = GenericProcessor(), 
             output_file = "my_exp2.c", 
             function_name = "my_exp2",
             language = C_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "exp2",
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
    # declaring target and instantiating optimization engine

    vx = self.implementation.add_input_variable("x", self.precision) 

    Log.set_dump_stdout(True)

    Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
    if self.debug_flag: 
        Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    index_size = 3

    approx_interval = Interval(0.0, 2**-index_size)
    error_goal_approx = 2**-(self.precision.get_precision())
    int_precision = {ML_Binary32: ML_Int32, ML_Binary64: ML_Int64}[self.precision]


    vx_int = Floor(vx * 2**index_size, precision = self.precision, tag = "vx_int", debug = debug_multi)
    vx_frac = vx - (vx_int * 2**-index_size)
    vx_frac.set_attributes(tag = "vx_frac", debug = debug_multi, unbreakable = True)
    poly_degree = sup(guessdegree(2**(sollya.x), approx_interval, error_goal_approx)) + 1
    precision_list = [1] + [self.precision] * (poly_degree)

    vx_integer = Conversion(vx_int, precision = int_precision, tag = "vx_integer", debug = debug_multi)
    vx_int_hi = BitLogicRightShift(vx_integer, Constant(index_size), tag = "vx_int_hi", debug = debug_multi)
    vx_int_lo = Modulo(vx_integer, 2**index_size, tag = "vx_int_lo", debug = debug_multi)
    pow_exp = ExponentInsertion(Conversion(vx_int_hi, precision = int_precision), precision = self.precision, tag = "pow_exp", debug = debug_multi)

    exp2_table = ML_Table(dimensions = [2 * 2**index_size, 2], storage_precision = self.precision, tag = self.uniquify_name("exp2_table"))
    for i in range(2 * 2**index_size):
      input_value = i - 2**index_size if i >= 2**index_size else i 
      exp2_value = SollyaObject(2)**((input_value)* 2**-index_size)
      hi_value = round(exp2_value, self.precision.get_sollya_object(), RN)
      lo_value = round(exp2_value - hi_value, self.precision.get_sollya_object(), RN)
      exp2_table[i][0] = lo_value
      exp2_table[i][1] = hi_value

    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(2**(sollya.x), poly_degree, precision_list, approx_interval, sollya.absolute, error_function = error_function)

    print "poly_approx_error: ", poly_approx_error, float(log2(poly_approx_error))

    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme
    poly = polynomial_scheme_builder(poly_object.sub_poly(start_index = 1), vx_frac, unified_precision = self.precision)
    poly.set_attributes(tag = "poly", debug = debug_multi)

    table_index = Addition(vx_int_lo, Constant(2**index_size, precision = int_precision), precision = int_precision, tag = "table_index", debug = debug_multi)

    lo_value_load = TableLoad(exp2_table, table_index, 0, tag = "lo_value_load", debug = debug_multi)
    hi_value_load = TableLoad(exp2_table, table_index, 1, tag = "hi_value_load", debug = debug_multi)

    result = (hi_value_load + (hi_value_load * poly + (lo_value_load + lo_value_load * poly))) * pow_exp
    ov_flag = Comparison(vx_int_hi, Constant(self.precision.get_emax(), precision = self.precision), specifier = Comparison.Greater)

    # main scheme
    Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    scheme = Statement(
                Return(
                  Select(
                    ov_flag,
                    FP_PlusInfty(self.precision),
                    result
                  )))

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_exp"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call

  def numeric_emulate(self, input_value):
    return sollya.SollyaObject(2)**(input_value)


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_exp2", default_output_file = "new_exp2.c" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_exp2          = ML_Exp2(args)

    ml_exp2.gen_implementation()
