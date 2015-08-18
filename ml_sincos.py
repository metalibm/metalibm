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
    frac_pi_index = 3
    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, RN)
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, RN)
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = True)
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
    red_vx = Select(sel_c, -pre_red_vx, pre_red_vx)
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

    index_relative = []

    table_index_size = frac_pi_index+1
    cos_table = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table"))
    sin_table = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("sin_table"))

    for i in xrange(2**(frac_pi_index+1)):
      local_x = i*pi/S2**frac_pi_index
      cos_table[i][0] = round(cos(local_x), self.precision.get_sollya_object(), RN)
      sin_table[i][0] = round(sin(local_x), self.precision.get_sollya_object(), RN)


    tabulated_cos = TableLoad(cos_table, modk) 
    tabulated_sin = TableLoad(sin_table, modk) 

    Log.report(Log.Info, "building mathematical polynomials for sin and cos")
    poly_degree_cos   = sup(guessdegree(cos(x), approx_interval, S2**(self.precision.get_field_size()+1)))
    poly_degree_sin   = sup(guessdegree(sin(x)/x, approx_interval, S2**(self.precision.get_field_size()+1)))

    poly_object_cos = Polynomial.build_from_approximation(cos(x), poly_degree_cos, [1, 1] + [self.precision] * (poly_degree_cos - 1), approx_interval, absolute)
    poly_object_sin = Polynomial.build_from_approximation(sin(x)/x, poly_degree_sin, [1] + [self.precision] * (poly_degree_sin), approx_interval, absolute)

    poly_cos = polynomial_scheme_builder(poly_object_cos, red_vx, unified_precision = self.precision)
    poly_sin = polynomial_scheme_builder(poly_object_sin, red_vx, unified_precision = self.precision)


    result = poly_cos * tabulated_cos - poly_sin * red_vx * tabulated_sin 

    #######################################################################
    #                    LARGE ARGUMENT MANAGEMENT                        #
    #                 (lar: Large Argument Reduction)                     #
    #######################################################################

    # payne and hanek argument reduction for large arguments
    #red_func_name = "payne_hanek_cosfp32" # "payne_hanek_fp32_asm"
    red_func_name = "payne_hanek_fp32_asm"
    payne_hanek_func_op = FunctionOperator(red_func_name, arg_map = {0: FO_Arg(0)}, require_header = ["support_lib/ml_red_arg.h"]) 
    payne_hanek_func   = FunctionObject(red_func_name, [ML_Binary32], ML_Binary64, payne_hanek_func_op)
    payne_hanek_func_op.declare_prototype = payne_hanek_func
    #large_arg_red = FunctionCall(payne_hanek_func, vx)
    large_arg_red = payne_hanek_func(vx)
    red_bound     = S2**20
    
    cond = Abs(vx) >= red_bound
    cond.set_attributes(tag = "cond", likely = False)


    
    lar_neark = NearestInteger(large_arg_red, precision = ML_Int64)
    lar_modk = Modulo(lar_neark, Constant(16, precision = ML_Int64), tag = "lar_modk", debug = True) 
    # Modulo is supposed to be already performed (by payne_hanek_cosfp32)
    #lar_modk = NearestInteger(large_arg_red, precision = ML_Int64)
    pre_lar_red_vx = large_arg_red - Conversion(lar_neark, precision = ML_Binary64)
    pre_lar_red_vx.set_attributes(precision = ML_Binary64, debug = debug_lftolx, tag = "pre_lar_red_vx")
    lar_red_vx = Conversion(pre_lar_red_vx, precision = self.precision, debug = debug_precision, tag = "lar_red_vx")
    lar_red_vx_lo = Conversion(pre_lar_red_vx - Conversion(lar_red_vx, precision = ML_Binary64), precision = self.precision)
    lar_red_vx_lo.set_attributes(tag = "lar_red_vx_lo", precision = self.precision)

    lar_k = 3
    # large arg reduction Universal Power Map
    lar_upm = {}
    lar_switch_map = {}
    approx_interval = Interval(-0.5, 0.5)
    for i in xrange(2**(lar_k+1)):
      frac_pi = pi / S2**lar_k
      func = cos(frac_pi * i + frac_pi * x)
      
      degree = 6
      error_mode = absolute
      if i % 2**(lar_k) == 2**(lar_k-1):
        # close to sin(x) cases
        func = -sin(frac_pi * x) if i == 2**(lar_k-1) else sin(frac_pi * x)
        degree_list = range(0, degree+1, 2)
        precision_list = [binary32] * len(degree_list)
        poly_object, _ = Polynomial.build_from_approximation_with_error(func/x, degree_list, precision_list, approx_interval, error_mode)
        poly_object = poly_object.sub_poly(offset = -1)
      else:
        degree_list = range(degree+1)
        precision_list = [binary32] * len(degree_list)
        poly_object, _ = Polynomial.build_from_approximation_with_error(func, degree_list, precision_list, approx_interval, error_mode)

      if i == 3 or i == 5 or i == 7 or i == 9 or i == 11 or i == 13: 
          poly_precision = ML_Binary64
          c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = ML_Binary64)
          c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
          poly_hi = (c0 + c1 * lar_red_vx)
          poly_hi.set_precision(ML_Binary64)
          pre_poly_scheme = poly_hi + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
          pre_poly_scheme.set_attributes(precision = ML_Binary64)
          poly_scheme = Conversion(pre_poly_scheme, precision = self.precision)
      elif i == 4 or i == 12:
        c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
        c3 = Constant(coeff(poly_object.get_sollya_object(), 3), precision = self.precision)
        c5 = Constant(coeff(poly_object.get_sollya_object(), 5), precision = self.precision)
        poly_hi = polynomial_scheme_builder(poly_object.sub_poly(start_index = 3), lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
        poly_hi.set_attributes(tag = "poly_lar_%d_hi" % i, precision = ML_Binary64)
        poly_scheme = Conversion(FusedMultiplyAdd(c1, lar_red_vx, poly_hi, precision = ML_Binary64) + c1 * lar_red_vx_lo, precision = self.precision)
      else:
        poly_scheme = polynomial_scheme_builder(poly_object, lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
      # poly_scheme = polynomial_scheme_builder(poly_object, lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm) 
      poly_scheme.set_attributes(tag = "lar_poly_%d" % i, debug = debug_precision)
      lar_switch_map[(i,)] = Return(poly_scheme)
    
    lar_result = SwitchBlock(lar_modk, lar_switch_map)


    # main scheme
    #Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    # scheme = Statement(ConditionBlock(cond, lar_result, result))

    Log.report(Log.Info, "Construction of the initial MDL scheme")
    scheme = Statement(pre_red_vx_d, red_vx_lo_sub, ConditionBlock(cond, lar_result, result))

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
