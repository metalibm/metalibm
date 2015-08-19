# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed






class ML_SinCos(ML_Function("ml_cos")):
  """ Implementation of cosinus function """
  def __init__(self, 
               precision = ML_Binary32, 
               accuracy  = ML_Faithful,
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               fast_path_extract = True,
               target = GenericProcessor(), 
               output_file = "cosf.c", 
               function_name = "cosf", 
               cos_output = True):
    # initializing I/O precision
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "cos",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag
    )
    self.precision = precision
    self.cos_output = cos_output



  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cos" if self.cos_output else "mpfr_sin"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self): 
    # declaring CodeFunction and retrieving input variable
    vx = Abs(self.implementation.add_input_variable("x", self.precision), tag = "vx") 


    Log.report(Log.Info, "generating implementation scheme")
    if self.debug_flag: 
        Log.report(Log.Info, "debug has been enabled")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    debug_precision = {ML_Binary32: debug_ftox, ML_Binary64: debug_lftolx}[self.precision]


    test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
    test_nan        = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
    test_positive   = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

    test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
    return_snan        = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    # return in case of infinity input
    infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(self.precision)), Return(FP_PlusZero(self.precision))))
    # return in case of specific value input (NaN or inf)
    specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(self.precision))), infty_return)
    # return in case of standard (non-special) input

    sollya_precision = self.precision.get_sollya_object()
    hi_precision = self.precision.get_field_size() - 3


    # argument reduction
    frac_pi_index = 6
    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, RN)
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, RN)
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = debugd)
    fk = Conversion(k, precision = self.precision, tag = "fk")

    inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
    inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

    red_vx_hi = (vx - inv_frac_pi_cst * fk)
    red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
    red_vx_lo_sub = inv_frac_pi_lo_cst * fk
    red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
    vx_d = Conversion(vx, precision = ML_Binary64, tag = "vx_d")
    pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d_hi = (vx_d - inv_frac_pi_cst * fk)
    pre_red_vx_d_hi.set_attributes(tag = "pre_red_vx_d_hi", precision = ML_Binary64, debug = debug_lftolx)
    pre_red_vx_d = pre_red_vx_d_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d.set_attributes(tag = "pre_red_vx_d", debug = debug_lftolx, precision = ML_Binary64)


    modk = Modulo(k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "switch_value", debug = True)

    sel_c = Equal(BitLogicAnd(modk, 2**(frac_pi_index-1)), 2**(frac_pi_index-1))
    sel_c.set_attributes(tag = "sel_c", debug = debugd)
    red_vx = pre_red_vx # Select(sel_c, -pre_red_vx, pre_red_vx)
    red_vx.set_attributes(tag = "red_vx", debug = debug_precision, precision = self.precision)

    red_vx_d = Select(sel_c, -pre_red_vx_d, pre_red_vx_d)
    red_vx_d.set_attributes(tag = "red_vx_d", debug = debug_lftolx, precision = ML_Binary64)

    approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

    Log.report(Log.Info, "approx interval: %s\n" % approx_interval)

    error_goal_approx = S2**-self.precision.get_precision()


    Log.report(Log.Info, "building tabulated approximation for sin and cos")
    poly_degree_vector = [None] * 2**(frac_pi_index+1)


    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

    table_index_size = frac_pi_index+1
    cos_table_hi = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_hi"))
    cos_table_lo = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_lo"))
    sin_table = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("sin_table"))

    cos_hi_prec = self.precision.get_sollya_object() # int(self.precision.get_field_size() * 0.7)

    for i in xrange(2**(frac_pi_index+1)):
      local_x = i*pi/S2**frac_pi_index
      cos_local_hi = round(cos(local_x), cos_hi_prec, RN)
      cos_table_hi[i][0] = cos_local_hi 
      cos_table_lo[i][0] = round(cos(local_x) - cos_local_hi, self.precision.get_sollya_object(), RN)
      #cos_table_d[i][0] = round(cos(local_x), ML_Binary64.get_sollya_object(), RN)

      sin_table[i][0] = round(sin(local_x), self.precision.get_sollya_object(), RN)


    tabulated_cos_hi = TableLoad(cos_table_hi, modk, 0, tag = "tab_cos_hi", debug = debug_precision) 
    tabulated_cos_lo = TableLoad(cos_table_lo, modk, 0, tag = "tab_cos_lo", debug = debug_precision) 
    tabulated_sin  = TableLoad(sin_table, modk, 0, tag = "tab_sin", debug = debug_precision) 

    Log.report(Log.Info, "building mathematical polynomials for sin and cos")
    poly_degree_cos   = sup(guessdegree(cos(x), approx_interval, S2**-(self.precision.get_field_size()+1)) ) + 2
    poly_degree_sin   = sup(guessdegree(sin(x)/x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 2

    print poly_degree_cos, poly_degree_sin

    poly_degree_cos_list = range(0, poly_degree_cos + 1, 2)
    poly_degree_sin_list = range(0, poly_degree_sin + 1, 2)

    poly_degree_cos_list = [0, 2, 4, 6]
    poly_cos_prec_list = [1, 1] + [self.precision] * 2

    poly_degree_sin_list = [0, 2, 4, 6]
    poly_sin_prec_list = [1] + [self.precision] * 3

    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    poly_object_cos, poly_error_cos = Polynomial.build_from_approximation_with_error(cos(x), poly_degree_cos_list, poly_cos_prec_list, approx_interval, absolute, error_function = error_function)
    poly_object_sin, poly_error_sin = Polynomial.build_from_approximation_with_error(sin(x)/x, poly_degree_sin_list, poly_sin_prec_list, approx_interval, absolute, error_function = error_function)
  
    print poly_object_cos.get_sollya_object()
    print poly_object_sin.get_sollya_object()
    print "poly_error: ", poly_error_cos, poly_error_sin

    poly_cos = polynomial_scheme_builder(poly_object_cos.sub_poly(start_index = 4, offset = 1), red_vx, unified_precision = self.precision)
    poly_sin = polynomial_scheme_builder(poly_object_sin.sub_poly(start_index = 2), red_vx, unified_precision = self.precision)
    poly_cos.set_attributes(tag = "poly_cos", debug = debug_precision)
    poly_sin.set_attributes(tag = "poly_sin", debug = debug_precision)


    cos_eval_d = tabulated_cos_hi - red_vx * (tabulated_sin + tabulated_cos_hi * red_vx * 0.5 + tabulated_sin * poly_sin - tabulated_cos_hi * poly_cos)
    cos_eval_d.set_attributes(tag = "cos_eval_d", debug = debug_precision, precision = self.precision)

    cos_eval_2 = (tabulated_cos_hi - red_vx) + - red_vx * ((tabulated_sin - 1) + tabulated_cos_hi * red_vx * 0.5 + tabulated_sin * poly_sin - tabulated_cos_hi * poly_cos)
    cos_eval_2.set_attributes(tag = "cos_eval_2", precision = self.precision, debug = debug_precision)



    result_sel_c = LogicalOr(
                    LogicalOr(
                      Equal(modk, Constant(2**(frac_pi_index-1)-1), precision = ML_Int32),
                      Equal(modk, Constant(2**(frac_pi_index-1)), precision = ML_Int32)
                    ),
                    Equal(modk, Constant(2**(frac_pi_index-1)+1), precision = ML_Int32),
                    tag = "result_sel_c",
                    debug = debugd
                  )

    result = Statement(
              cos_eval_2,
            Return(
                cos_eval_d
              )
            )


    Log.report(Log.Info, "Construction of the initial MDL scheme")
    scheme = Statement(result)

    print "poly_error_cos: ", int(log2(sup(abs(poly_error_cos))))
    print "poly_error_sin: ", int(log2(sup(abs(poly_error_sin))))

    return scheme




if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_sincos", default_output_file = "new_sincos.c" )
  # argument extraction 
  cos_output = arg_template.test_flag_option("--cos", True, False, parse_arg = arg_template.parse_arg, help_str = "select cos output") 

  parse_arg_index_list = arg_template.sys_arg_extraction()
  arg_template.check_args(parse_arg_index_list)


  ml_sincos = ML_SinCos(arg_template.precision, 
                     libm_compliant            = arg_template.libm_compliant, 
                     debug_flag                = arg_template.debug_flag, 
                     target                    = arg_template.target, 
                     fuse_fma                  = arg_template.fuse_fma, 
                     fast_path_extract         = arg_template.fast_path,
                     function_name             = arg_template.function_name,
                     accuracy                  = arg_template.accuracy,
                     output_file               = arg_template.output_file,
                     cos_output                = cos_output)
  ml_sincos.gen_implementation()
