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

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  
from metalibm_core.utility.debug_utils import *

class ML_Acos(ML_Function("acos")):
  def __init__(self, 
             precision = ML_Binary32, 
             abs_accuracy = S2**-24, 
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = GenericProcessor(), 
             output_file = None, 
             function_name = None):

    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "acos",
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

  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_acos"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    sollya_precision = self.get_sollya_precision()

    # retrieving processor inverse approximation table
    #dummy_var = Variable("dummy", precision = self.precision)
    #dummy_div_seed = DivisionSeed(dummy_var, precision = self.precision)
    #inv_approx_table = self.processor.get_recursive_implementation(dummy_div_seed, language = None, table_getter = lambda self: self.approx_table_map)
    lo_bound_global = SollyaObject(0.0)
    hi_bound_global = SollyaObject(0.75)
    approx_interval = Interval(lo_bound_global, hi_bound_global)
    approx_interval_size = hi_bound_global - lo_bound_global

    # table creation
    table_index_size = 7
    field_index_size = 2
    exp_index_size = table_index_size - field_index_size

    table_size = 2**table_index_size
    table_index_range = range(table_size)

    local_degree = 9
    coeff_table = ML_Table(dimensions = [table_size, local_degree], storage_precision = self.precision)

    #local_interval_size = approx_interval_size / SollyaObject(table_size)
    #for i in table_index_range:
    #  degree = 6
    #  lo_bound = lo_bound_global + i * local_interval_size
    #  hi_bound = lo_bound_global + (i+1) * local_interval_size
    #  approx_interval = Interval(lo_bound, hi_bound)
    #  local_poly_object, local_error = Polynomial.build_from_approximation_with_error(acos(x), degree, [self.precision] * (degree+1), approx_interval, absolute)
    #  local_error = int(log2(sup(abs(local_error / acos(approx_interval)))))
    #  print approx_interval, local_error

    exp_lo = 2**exp_index_size
    for i in table_index_range:
      lo_bound = (1.0 + (i % 2**field_index_size) * S2**-field_index_size) * S2**(i / 2**field_index_size - exp_lo)
      hi_bound = (1.0 + ((i % 2**field_index_size) + 1) * S2**-field_index_size) * S2**(i / 2**field_index_size - exp_lo)
      local_approx_interval = Interval(lo_bound, hi_bound)
      local_poly_object, local_error = Polynomial.build_from_approximation_with_error(acos(1 - x), local_degree, [self.precision] * (local_degree+1), local_approx_interval, absolute)
      local_error = int(log2(sup(abs(local_error / acos(1 - local_approx_interval)))))
      coeff_table
      print local_approx_interval, local_error
      for d in xrange(local_degree):
        coeff_table[i][d] = coeff(local_poly_object.get_sollya_object(), d) 

    table_index = BitLogicRightShift(vx, vx.get_precision().get_field_size() - field_index_size) - (exp_lo << field_index_size)




    print "building mathematical polynomial"
    poly_degree = sup(guessdegree(acos(x), approx_interval, S2**-(self.precision.get_field_size()))) 
    print "guessed polynomial degree: ", int(poly_degree)
    #global_poly_object = Polynomial.build_from_approximation(log10(1+x)/x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, absolute)

    print "generating polynomial evaluation scheme"
    #_poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)

    # building eval error map
    #eval_error_map = {
    #  red_vx: Variable("red_vx", precision = self.precision, interval = red_vx.get_interval()),
    #  log_inv_hi: Variable("log_inv_hi", precision = self.precision, interval = table_high_interval),
    #  log_inv_lo: Variable("log_inv_lo", precision = self.precision, interval = table_low_interval),
    #}
    # computing gappa error
    #poly_eval_error = self.get_eval_error(result, eval_error_map)



    # main scheme
    print "MDL scheme"
    scheme = Statement(Return(vx))
    return scheme



if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_acos", default_output_file = "new_acos.c" )
  arg_template.sys_arg_extraction()


  ml_acos          = ML_Acos(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)
  ml_acos.gen_implementation()
