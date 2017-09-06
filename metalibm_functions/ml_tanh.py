# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, tanh, guessdegree, dirtyinfnorm, RN, atanh, RD
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



class ML_HyperbolicTangent(ML_Function("ml_tanh")):
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = GenericProcessor(), 
             output_file = "my_tanh.c", 
             function_name = "my_tanh",
             language = C_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "tanh",
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
    
    def compute_reciprocal(vx):
        inv_seed = DivisionSeed(vx, precision = self.precision, tag = "inv_seed", debug = debug_multi)
        nr_1 = 2*inv_seed - vx*inv_seed*inv_seed
        nr_2 = 2*nr_1 - vx*nr_1*nr_1
        nr_3 =2*nr_2 - vx*nr_2*nr_2
        inv_vx = 2*nr_3 - vx*nr_3*nr_3
        
        return inv_vx
    
    
    # declaring target and instantiating optimization engine

    vx = self.implementation.add_input_variable("x", self.precision) 

    Log.set_dump_stdout(True)
    
    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)
        
    sollya_precision = self.precision.get_sollya_object()
      
    int_precision = {
        ML_Binary32 : ML_Int32,
        ML_Binary64 : ML_Int64
      }[self.precision]
      
    hi_precision = self.precision.get_field_size() - 12
    index_size = 3
    
    if self.precision is ML_Binary32:
      bound = 9
    else:
      bound = 22
    
    test_sign = Comparison(vx, 0, specifier = Comparison.Less, precision = ML_Bool, debug = debug_multi, tag = "Is_Negative")
    neg_vx = -vx
    
    sign = Variable("sign", precision = self.precision, var_type = Variable.Local)
    
    set_sign = Statement(
        ConditionBlock(test_sign,
          Statement(ReferenceAssign(vx, 2*neg_vx), ReferenceAssign(sign, -1)),
          Statement(ReferenceAssign(vx, 2*vx), ReferenceAssign(sign, 1))
      ))
  
    sollya_prec_map = {ML_Binary32: sollya.binary32, ML_Binary64: sollya.binary64}
    
    # Constants
    
    log_2 = round(log(2), sollya_prec_map[self.precision], sollya.RN)
    invlog2 = round(1/log(2), sollya_prec_map[self.precision], sollya.RN)
    
    interval_vx = Interval(0, bound)
    interval_fk = interval_vx * invlog2
    interval_k = Interval(floor(inf(interval_fk)), ceil(sup(interval_fk)))
    
    log2_hi_precision = self.precision.get_field_size() - 4
    log2_hi = round(log(2), log2_hi_precision, sollya.RN)
    log2_lo = round(log(2) - log2_hi, sollya_prec_map[self.precision], sollya.RN)


    # Reduction
    
    unround_k = vx * invlog2
    k = NearestInteger(unround_k, precision = self.precision, tag = "k")
    ik = NearestInteger(unround_k, precision = ML_Int32, debug = debug_multi, tag = "ik")
    exact_pre_mul = (k * log2_hi)
    exact_pre_mul.set_attributes(exact = True)
    exact_hi_part = vx - exact_pre_mul
    exact_hi_part.set_attributes(exact = True, prevent_optimization = True)
    exact_lo_part = - k * log2_lo
    exact_lo_part.set_attributes(prevent_optimization = True)
    
    r = exact_hi_part + exact_lo_part
    # z = s - exact_hi_part
    # t = exact_lo_part - z
    # r = s + t
    
    r.set_attributes(tag = "r", debug = debug_multi)
    
    r_interval = Interval(-log_2/S2, log_2/S2)
    r_interval_tanh = Interval(0, 0.48)
    r_interval_tanh2 = Interval(0.48, 0.52)
    r_interval_tanh3 = Interval(0.52, 0.9)
    r_interval_tanh4 = Interval(0.9, 1)
    
    local_ulp = sup(ulp(exp(r_interval), self.precision))
    
    print "ulp: ", local_ulp 
    error_goal = S2**-1*local_ulp
    print "error goal: ", error_goal 
    
    # Polynomial Approx
    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)
    Log.report(Log.Info, "\033[33;1m Building polynomial \033[0m\n")
    
    poly_degree = sup(guessdegree(expm1(sollya.x), r_interval, error_goal) + 1)
    poly_degree_tanh = sup(guessdegree(tanh(sollya.x), r_interval_tanh, error_goal) + 4)
    poly_degree_tanh2 = sup(guessdegree(tanh(sollya.x), r_interval_tanh2, error_goal)+ 2)
    poly_degree_tanh3 = sup(guessdegree(tanh(sollya.x), r_interval_tanh3, error_goal)+ 4)
    poly_degree_tanh4 = sup(guessdegree(tanh(sollya.x), r_interval_tanh4, error_goal)+ 2)
    
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme
    poly_degree_list = range(0, poly_degree)
    poly_degree_list_tanh = range(0, poly_degree_tanh)
    poly_degree_list_tanh2 = range(0, poly_degree_tanh2)
    poly_degree_list_tanh3 = range(0, poly_degree_tanh3)
    poly_degree_list_tanh4 = range(0, poly_degree_tanh4)
    
    precision_list = [self.precision] *(len(poly_degree_list) + 1)
    precision_list_tanh = [self.precision] *(len(poly_degree_list_tanh) + 1)
    precision_list_tanh2 = [self.precision] *(len(poly_degree_list_tanh2) + 1)
    precision_list_tanh3 = [self.precision] *(len(poly_degree_list_tanh3) + 1)
    precision_list_tanh4 = [self.precision] *(len(poly_degree_list_tanh4) + 1)
    
    poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(expm1(sollya.x), poly_degree, precision_list, r_interval, sollya.absolute, error_function = error_function)
    poly_object_tanh, poly_approx_error_tanh = Polynomial.build_from_approximation_with_error(tanh(sollya.x), poly_degree_tanh, precision_list_tanh, r_interval_tanh, sollya.absolute, error_function = error_function)
    poly_object_tanh2, poly_approx_error_tanh2 = Polynomial.build_from_approximation_with_error(tanh(sollya.x), poly_degree_tanh2, precision_list_tanh2, r_interval_tanh2, sollya.absolute, error_function = error_function)
    poly_object_tanh3, poly_approx_error_tanh3 = Polynomial.build_from_approximation_with_error(tanh(sollya.x), poly_degree_tanh3, precision_list_tanh3, r_interval_tanh3, sollya.absolute, error_function = error_function)
    poly_object_tanh4, poly_approx_error_tanh4 = Polynomial.build_from_approximation_with_error(tanh(sollya.x), poly_degree_tanh4, precision_list_tanh4, r_interval_tanh4, sollya.absolute, error_function = error_function)
    
    print "poly_approx_error: ", poly_approx_error, float(log2(poly_approx_error))
    print "poly_approx_error_tanh: ", poly_approx_error_tanh, float(log2(poly_approx_error_tanh))
    print "poly_approx_error_tanh2: ", poly_approx_error_tanh2, float(log2(poly_approx_error_tanh2))
    print "poly_approx_error_tanh3: ", poly_approx_error_tanh3, float(log2(poly_approx_error_tanh3))
    print "poly_approx_error_tanh4: ", poly_approx_error_tanh4, float(log2(poly_approx_error_tanh4))
    
    sub_poly = poly_object.sub_poly(start_index = 2)
    Log.report(Log.Info, "Poly : %s" % sub_poly)
    pre_sub_poly = polynomial_scheme_builder(sub_poly, r, unified_precision = self.precision)
    poly = r + pre_sub_poly
    poly.set_attributes(tag = "poly", debug = debug_multi)
    
    exp_k = ExponentInsertion(ik, tag = "exp_k", debug = debug_multi, precision = self.precision)
    exp_mk = ExponentInsertion(-ik, tag = "exp_mk", debug = debug_multi, precision = self.precision)
    
    diff = 1 - exp_mk
    diff.set_attributes(tag = "diff", debug = debug_multi) 

    std_result = exp_k * ( poly + diff )
    
    result1 = std_result + 2 
    result2 = 2 * compute_reciprocal(result1)
    result = 1 - result2
    result.set_attributes(tag = "result", debug = debug_multi)
    
    result_tanh = polynomial_scheme_builder(poly_object_tanh, 0.5*vx, unified_precision = self.precision)
    result_tanh.set_attributes(tag = "result_tanh", debug = debug_multi)
    
    result_tanh2 = polynomial_scheme_builder(poly_object_tanh2, 0.5*vx, unified_precision = self.precision)
    result_tanh2.set_attributes(tag = "result_tanh2", debug = debug_multi)
    
    result_tanh3 = polynomial_scheme_builder(poly_object_tanh3, 0.5*vx, unified_precision = self.precision)
    result_tanh3.set_attributes(tag = "result_tanh3", debug = debug_multi)
    
    result_tanh4 = polynomial_scheme_builder(poly_object_tanh4, 0.5*vx, unified_precision = self.precision)
    result_tanh4.set_attributes(tag = "result_tanh4", debug = debug_multi)
    # ov_value 
    ov_value = bound
    ov_flag = Comparison(vx*0.5, Constant(ov_value, precision = self.precision), specifier = Comparison.Greater, tag = "ov_flag", debug = debug_multi)
    
    test_interval = Comparison(vx*0.5, 1, specifier = Comparison.Greater, precision = ML_Bool)
    test_interval2 = Comparison(vx*0.5, 0.48, specifier = Comparison.Greater, precision = ML_Bool)
    test_interval3 = Comparison(vx*0.5, 0.52, specifier = Comparison.Greater, precision = ML_Bool)
    test_interval4 = Comparison(vx*0.5, 0.9, specifier = Comparison.Greater, precision = ML_Bool)
    # main scheme
    scheme = Statement(
                sign,
                set_sign,
                ConditionBlock(
                  ov_flag,
                  Return(sign*Constant(1.0, precision = self.precision)),
                  ConditionBlock(
                    test_interval,
                    Return(sign*result),
                    ConditionBlock(
                      test_interval4,
                      Return(result_tanh4),
                      ConditionBlock(
                        test_interval3,
                        Return(result_tanh3),
                        ConditionBlock(
                          test_interval2,
                          Return(result_tanh2),
                          Return(result_tanh)
                        )
                      )
                    )
                  )
                )
              )

      
    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_tanh"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call

  def numeric_emulate(self, input_value):
    return tanh(input_value)

  standard_test_cases =[[sollya_parse(x)] for x in  ["0x1.d7df94p-7", "0x1.efc2cp-6"]]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_tanh", default_output_file = "new_tanh.c" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_tanh          = ML_HyperbolicTangent(args)

    ml_tanh.gen_implementation()
