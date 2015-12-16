# -*- coding: utf-8 -*-

# Description: fast and low accuracy sine and cosine implementation
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)

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
from metalibm_core.code_generation.fixed_point_backend import FixedPointBackend

from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed




    


class ML_FastSinCos(ML_Function("ml_fast_cos")):
  """ Implementation of cosinus function """
  def __init__(self, 
               precision = ML_Binary32, 
               accuracy  = ML_Faithful,
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               fast_path_extract = True,
               processor = GenericProcessor(), 
               output_file = "cosf.c", 
               function_name = "cosf", 
               input_interval = Interval(0, 1),
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

      processor = processor,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag
    )
    self.precision  = precision
    self.cos_output = cos_output
    self.accuracy   = accuracy 
    self.input_interval = input_interval



  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_FastSinCos functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cos" if self.cos_output else "mpfr_sin"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self): 
    # declaring CodeFunction and retrieving input variable
    vx = self.implementation.add_input_variable("x", self.precision)


    Log.report(Log.Info, "target: %s " % self.processor.target_name)

    # display parameter information
    Log.report(Log.Info, "accuracy      : %s " % self.accuracy)
    Log.report(Log.Info, "input interval: %s " % self.input_interval)

    accuracy_goal = self.accuracy.get_goal()
    computation_precision = self.precision

    # polynomial approximation
    approx_interval = self.input_interval
    poly_degree = sup(guessdegree(cos(x), approx_interval, accuracy_goal))
    Log.report(Log.Info, "poly degree: %d" % poly_degree)

    poly_object = Polynomial.build_from_approximation(cos(x), poly_degree, [computation_precision] * (poly_degree+1), approx_interval, absolute)

    table_size_log = 8
    integer_size = 31
    integer_precision = ML_Int32

    max_bound = sup(abs(self.input_interval))
    max_bound_log = int(ceil(log2(max_bound)))
    scaling_power = integer_size - max_bound_log

    storage_precision = ML_Custom_FixedPoint_Format(3, 29)

    # cosine table
    cos_table = ML_Table(dimensions = [2**table_size_log, 1], storage_precision = ML_Int32, tag = self.uniquify_name("cos_table"))
    # filling table
    for i in xrange(2**table_size_log):
      local_x   = i / S2**table_size_log * S2**max_bound_log
      cos_local = nearestint(cos(local_x) * S2**storage_precision.get_frac_size())
      cos_table[i][0] = cos_local

    # argument reduction evaluation scheme
    scaling_factor = Constant(S2**scaling_power, precision = self.precision)

    red_vx_precision = ML_Custom_FixedPoint_Format(32 - scaling_power, scaling_power)
    # red_vx = NearestInteger(vx * scaling_factor, precision = integer_precision)
    red_vx = Conversion(vx, precision = red_vx_precision)

    hi_mask = 2**32 - 2**(32-table_size_log)
    red_vx_hi_int = BitLogicAnd(TypeCast(red_vx, precision = ML_Int32), hi_mask)
    red_vx_hi = TypeCast(red_vx_hi_int, precision = red_vx_precision)
    red_vx_lo = red_vx - red_vx_hi
    red_vx_lo.set_precision(red_vx_precision)
    table_index = BitLogicRightShift(TypeCast(red_vx, precision = ML_Int32), 32 - table_size_log)

    tabulated_cos = TableLoad(cos_table, table_index, 0, tag = "tab_cos")

    result = red_vx_lo # Conversion(red_vx_lo, precision = self.precision)

    # polynomial evaluation scheme
    Log.report(Log.Info, "generating implementation scheme")
    if self.debug_flag: 
        Log.report(Log.Info, "debug has been enabled")

    #poly_scheme = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, vx, unified_precision = computation_precision)



    scheme = Statement(
      Return(result)
    )


    return scheme





if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_fastsincos", default_output_file = "new_fastsincos.c" )
  # argument extraction 
  cos_output = arg_template.test_flag_option("--cos", True, False, parse_arg = arg_template.parse_arg, help_str = "select cos output") 
  enable_subexpr_sharing = arg_template.test_flag_option("--enable-subexpr-sharing", True, False, parse_arg = arg_template.parse_arg, help_str = "force subexpression sharing")

  parse_arg_index_list = arg_template.sys_arg_extraction()
  arg_template.check_args(parse_arg_index_list)


  ml_fastsincos = ML_FastSinCos(arg_template.precision, 
                     libm_compliant            = arg_template.libm_compliant, 
                     debug_flag                = arg_template.debug_flag, 
                     processor                 = arg_template.target, 
                     fuse_fma                  = arg_template.fuse_fma, 
                     fast_path_extract         = arg_template.fast_path,
                     function_name             = arg_template.function_name,
                     accuracy                  = arg_template.accuracy,
                     output_file               = arg_template.output_file,
                     input_interval            = arg_template.input_interval,
                     cos_output                = cos_output)
  ml_fastsincos.gen_implementation(display_after_opt = arg_template.display_after_opt, enable_subexpr_sharing = enable_subexpr_sharing)
