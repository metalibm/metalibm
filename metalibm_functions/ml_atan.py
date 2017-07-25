# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import (
        S2, Interval, ceil, floor, round, inf, sup, pi, log, atan,
        guessdegree, dirtyinfnorm
)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_Faithful, ML_CorrectlyRounded
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import ML_NewArgTemplate, ArgDefault
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

# Disabling Sollya's rounding warnings
sollya.roundingwarnings = sollya.off
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on

class ML_Atan(ML_Function("atan")):
  def __init__(self,
              arg_template = DefaultArgTemplate,
                precision = ML_Binary32,
                accuracy  = ML_CorrectlyRounded,
                libm_compliant = True,
                debug_flag = False,
                fuse_fma = True,
                fast_path_extract = True,
                target = GenericProcessor(),
                output_file = "ml_atan.c",
                function_name = "ml_atan"):
                
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])            
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "atan",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template
    )

    self.precision = precision

  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_atan"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    sollya_precision = self.precision.get_sollya_object()
    
    int_precision = {
        ML_Binary32 : ML_Int32,
        ML_Binary64 : ML_Int64
      }[self.precision]
    
    half_pi = round(pi*0.5, sollya_precision, sollya.RN)
    half_pi_cst = Constant(half_pi, precision = self.precision)
    abs_vx_std = Variable("abs_vx", precision = self.precision, var_type = Variable.Local)
    red_vx_std = Variable("red_vx", precision = self.precision, var_type = Variable.Local)
    
    if self.precision is ML_Binary32:
      bound = 24
    else:
      bound = 53
      
    test_zero = Test(vx, specifier = Test.IsZero, precision = ML_Bool, debug = debug_multi, tag = "Is_Zero")
    test_sign = Comparison(vx, 0, specifier = Comparison.Less, precision = ML_Bool, debug = debug_multi, tag = "Is_Negative")
    test_inv = Comparison(abs_vx_std, 1, specifier = Comparison.Greater, precision = ML_Bool, debug = debug_multi, tag = "Greater_Than_1")
    test_bound = Comparison(vx, S2**bound, specifier = Comparison.GreaterOrEqual, precision = ML_Bool, debug = debug_multi, tag ="bound")
    
    neg_vx = -vx
    sign = Variable("sign", precision = self.precision, var_type = Variable.Local)
    inv_vx = 1.0/abs_vx_std
  
    
    set_sign = Statement(
      ConditionBlock(test_sign,
        Statement(ReferenceAssign(abs_vx_std, neg_vx), ReferenceAssign(sign, -1)),
        Statement(ReferenceAssign(abs_vx_std, vx), ReferenceAssign(sign, 1))
    ))
    
    set_red_vx = Statement(
      ConditionBlock(test_inv,
        ReferenceAssign(red_vx_std, inv_vx),
        ReferenceAssign(red_vx_std, abs_vx_std))
    )
    
    r_interval = Interval(S2**-bound, 1)
    red_vx_std.set_interval(r_interval)
    
    inv_vx.set_attributes(tag = "inv_vx", debug = debug_multi)
    
    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    
    poly_degree = sup(guessdegree(atan(sollya.x), r_interval, S2**-self.precision.get_precision()) + 2)
    poly_degree_list = range(0, poly_degree + 2)
    # poly_degree_list = [0, 1, 3, 5, 7, 9]
    poly_prec_list = [self.precision] * len(poly_degree_list)
    
    Log.report(Log.Info, "Building mathematical polynomial for atan")
    
    poly_object, poly_error = Polynomial.build_from_approximation_with_error(atan(sollya.x), poly_degree_list, poly_prec_list, r_interval, sollya.absolute, error_function = error_function)
    
    Log.report(Log.Info, "poly error : {} / {:d}".format(poly_error, int(sollya.log2(poly_error))))
    Log.report(Log.Info, "poly : %s" % poly_object.sub_poly(start_index = 1))
    
    poly = polynomial_scheme_builder(poly_object.sub_poly(start_index = 1), red_vx_std, unified_precision = self.precision)
    poly.set_attributes(tag = "poly", debug = debug_multi)
    
    s = (half_pi_cst - poly)
    z = (s - half_pi_cst)
    t = (- poly - z)
    result = s + t
    
    s.set_attributes(tag = "s", debug = debug_multi)
    z.set_attributes(tag = "z", debug = debug_multi)
    t.set_attributes(tag = "t", debug = debug_multi)
    
    result.set_attributes(tag ="result", debug = debug_multi)
    std_return = ConditionBlock(
      test_inv,
      Return(sign*result),
      Return(sign*poly))
    
    scheme = Statement(
      abs_vx_std,
      red_vx_std,
      sign,
      set_sign,
      ConditionBlock(
        test_bound,
        ConditionBlock(
          test_sign,
          Return(-half_pi_cst),
          Return(half_pi_cst)
        ),
        set_red_vx
      ),
      std_return)
    
    return scheme

  def numeric_emulate(self, input_value):
    return atan(input_value)

  standard_test_cases =[[sollya.parse(x)] for x in  ["-0x1.3b150ap+8", "0x1.9e75a6p+0" ]]
if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_atan", default_output_file = "new_atan.c" )
    args = arg_template.arg_extraction()
    ml_atan = ML_Atan(args)
    ml_atan.gen_implementation()
