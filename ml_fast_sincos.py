# -*- coding: utf-8 -*-

# Description: fast and low accuracy sine and cosine implementation
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# Created:       December 16th, 2015
# Last-modified: December 17th, 2015

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

from metalibm_core.utility.ml_template import ML_ArgTemplate, precision_parser
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

# set sollya verbosity level to 0
verbosity(0)

## Fast implementation of trigonometric function sine and cosine
#  Focuses on speed rather than on accuracy. Accepts --accuracy
#  and --input-interval options
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
               result_precision = ML_Binary32,
               table_size_log = 8,
               cos_output = True):
    # initializing I/O precision
    io_precisions = [result_precision, precision] 

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
    self.table_size_log = table_size_log


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
    Log.report(Log.Info, "accuracy_goal=%f" % accuracy_goal)


    table_size_log = self.table_size_log
    integer_size = 31
    integer_precision = ML_Int32

    max_bound = sup(abs(self.input_interval))
    max_bound_log = int(ceil(log2(max_bound)))
    Log.report(Log.Info, "max_bound_log=%s " % max_bound_log)
    scaling_power = integer_size - max_bound_log
    Log.report(Log.Info, "scaling power: %s " % scaling_power)

    storage_precision = ML_Custom_FixedPoint_Format(1, 30, signed = True)

    Log.report(Log.Info, "tabulating cosine and sine")
    # cosine and sine fused table
    fused_table = ML_Table(dimensions = [2**table_size_log, 2], storage_precision = storage_precision, tag = "fast_lib_shared_table") # self.uniquify_name("cossin_table"))
    # filling table
    for i in xrange(2**table_size_log):
      local_x = i / S2**table_size_log * S2**max_bound_log

      cos_local = cos(local_x) # nearestint(cos(local_x) * S2**storage_precision.get_frac_size())

      sin_local = sin(local_x) # nearestint(sin(local_x) * S2**storage_precision.get_frac_size())

      fused_table[i][0] = cos_local
      fused_table[i][1] = sin_local

    # argument reduction evaluation scheme
    # scaling_factor = Constant(S2**scaling_power, precision = self.precision)

    red_vx_precision = ML_Custom_FixedPoint_Format(31 - scaling_power, scaling_power, signed = True)
    Log.report(Log.Verbose, "red_vx_precision.get_c_bit_size()=%d" % red_vx_precision.get_c_bit_size())
    # red_vx = NearestInteger(vx * scaling_factor, precision = integer_precision)
    red_vx = Conversion(vx, precision = red_vx_precision, tag = "red_vx", debug = debug_fixed32)

    computation_precision = red_vx_precision # self.precision
    output_precision      = self.io_precisions[0]
    Log.report(Log.Info, "computation_precision is %s" % computation_precision)
    Log.report(Log.Info, "storage_precision     is %s" % storage_precision)
    Log.report(Log.Info, "output_precision      is %s" % output_precision)

    hi_mask_value = 2**32 - 2**(32-table_size_log - 1)
    hi_mask = Constant(hi_mask_value, precision = ML_Int32)
    Log.report(Log.Info, "hi_mask=0x%x" % hi_mask_value)

    red_vx_hi_int = BitLogicAnd(TypeCast(red_vx, precision = ML_Int32), hi_mask, precision = ML_Int32, tag = "red_vx_hi_int", debug = debugd)
    red_vx_hi = TypeCast(red_vx_hi_int, precision = red_vx_precision, tag = "red_vx_hi", debug = debug_fixed32)
    red_vx_lo = red_vx - red_vx_hi
    red_vx_lo.set_attributes(precision = red_vx_precision, tag = "red_vx_lo", debug = debug_fixed32)
    table_index = BitLogicRightShift(TypeCast(red_vx, precision = ML_Int32), scaling_power - (table_size_log - max_bound_log), precision = ML_Int32, tag = "table_index", debug = debugd)

    tabulated_cos = TableLoad(fused_table, table_index, 0, tag = "tab_cos", precision = storage_precision, debug = debug_fixed32)
    tabulated_sin = TableLoad(fused_table, table_index, 1, tag = "tab_sin", precision = storage_precision, debug = debug_fixed32)

    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    Log.report(Log.Info, "building polynomial approximation for cosine")
    # cosine polynomial approximation
    poly_interval = Interval(0, S2**(max_bound_log - table_size_log))
    Log.report(Log.Info, "poly_interval=%s " % poly_interval)
    cos_poly_degree = 2 # int(sup(guessdegree(cos(x), poly_interval, accuracy_goal)))

    Log.report(Log.Verbose, "cosine polynomial approximation")
    cos_poly_object, cos_approx_error = Polynomial.build_from_approximation_with_error(cos(x), [0, 2] , [0] + [computation_precision.get_bit_size()], poly_interval, absolute, error_function = error_function)
    #cos_eval_scheme = PolynomialSchemeEvaluator.generate_horner_scheme(cos_poly_object, red_vx_lo, unified_precision = computation_precision)
    Log.report(Log.Info, "cos_approx_error=%e" % cos_approx_error)
    cos_coeff_list = cos_poly_object.get_ordered_coeff_list()
    coeff_C0 = cos_coeff_list[0][1]
    coeff_C2 = Constant(cos_coeff_list[1][1], precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True))

    Log.report(Log.Info, "building polynomial approximation for sine")

    # sine polynomial approximation
    sin_poly_degree = 2 # int(sup(guessdegree(sin(x)/x, poly_interval, accuracy_goal)))
    Log.report(Log.Info, "sine poly degree: %e" % sin_poly_degree)
    Log.report(Log.Verbose, "sine polynomial approximation")
    sin_poly_object, sin_approx_error = Polynomial.build_from_approximation_with_error(sin(x)/x, [0, 2], [0] + [computation_precision.get_bit_size()] * (sin_poly_degree+1), poly_interval, absolute, error_function = error_function)
    sin_coeff_list = sin_poly_object.get_ordered_coeff_list()
    coeff_S0 = sin_coeff_list[0][1]
    coeff_S2 = Constant(sin_coeff_list[1][1], precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True))

    # scheme selection between sine and cosine
    if self.cos_output:
      scheme = self.generate_cos_scheme(computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo)
    else:
      scheme = self.generate_sin_scheme(computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo)

    result = Conversion(scheme, precision = self.io_precisions[0])

    Log.report(Log.Verbose, "result operation tree :\n %s " % result.get_str(display_precision = True, depth = None, memoization_map = {}))
    scheme = Statement(
      Return(result)
    )

    return scheme


  ## generate scheme for cosine approximation of cos(X = x + u)
  #  @param computation_precision ML_Format used as default precision for scheme evaluation
  #  @param tabulated_cos tabulated value of cosine(high part of vx)
  #  @param tabulated_sin tabulated value of   sine(high part of vx)
  #  @param sin_C2 polynomial coefficient of sine approximation for u^3 
  #  @param cos_C2 polynomial coefficient of cosine approximation for u^2
  #  @param red_vx_lo low part of the reduced input variable (i.e. u)
  def generate_cos_scheme(self, computation_precision, tabulated_cos, tabulated_sin, sin_C2, cos_C2, red_vx_lo):
    cos_C2 = Multiplication(
              tabulated_cos,
              cos_C2,
              precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
              tag = "cos_C2"
            )
    u2 = Multiplication(
          red_vx_lo,
          red_vx_lo,
          precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
          tag = "u2"
        )
    sin_u = Multiplication(
              tabulated_sin,
              red_vx_lo,
              precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30, signed = True)
              tag = "sin_u"
            )

    cos_C2_u2 = Multiplication(
                  cos_C2,
                  u2,
                  precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30,signed = True)
                  tag = "cos_C2_u2"
                )

    S2_u2 = Multiplication(
              sin_C2, 
              u2,
              precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
              tag = "S2_u2"
            )

    S2_u3_sin = Multiplication(
                  S2_u2,
                  sin_u,
                  precision = computation_precision, # ML_Custom_FixedPoint_Format(5,26, signed = True)
                  tag = "S2_u3_sin"
                )

    cos_C2_u2_P_cos = Addition(
                        tabulated_cos,
                        cos_C2_u2,
                        precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                        tag = "cos_C2_u2_P_cos"
                      )

    cos_C2_u2_P_cos_M_sin_u = Subtraction(
                                cos_C2_u2_P_cos,
                                sin_u,
                                precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                              )

    scheme = Subtraction(
                cos_C2_u2_P_cos_M_sin_u,
                S2_u3_sin,
                precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
              )

    return scheme


  ## generate scheme for sine approximation of sin(X = x + u)
  #  @param computation_precision ML_Format used as default precision for scheme evaluation
  #  @param tabulated_cos tabulated value of cosine(high part of vx)
  #  @param tabulated_sin tabulated value of   sine(high part of vx)
  #  @param sin_C2 polynomial coefficient of sine approximation for u^3 
  #  @param cos_C2 polynomial coefficient of cosine approximation for u^2
  #  @param red_vx_lo low part of the reduced input variable (i.e. u)
  def generate_sin_scheme(self, computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo):
    sin_C2 = Multiplication(
              tabulated_sin,
              coeff_C2,
              precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
              tag = "sin_C2"
            )
    u2 = Multiplication(
          red_vx_lo,
          red_vx_lo,
          precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
          tag = "u2"
        )
    cos_u = Multiplication(
              tabulated_cos,
              red_vx_lo,
              precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30, signed = True)
              tag = "cos_u"
            )

    S2_u2 = Multiplication(
                  coeff_S2,
                  u2,
                  precision = ML_Custom_FixedPoint_Format(-1, 32,signed = True),
                  tag = "S2_u2"
                )

    sin_C2_u2 = Multiplication(
              sin_C2, 
              u2,
              precision = computation_precision,
              tag = "sin_C2_u2"
            )

    S2_u3_cos = Multiplication(
                  S2_u2,
                  cos_u,
                  precision = computation_precision, # ML_Custom_FixedPoint_Format(5,26, signed = True)
                  tag = "S2_u3_cos"
                )

    sin_P_cos_u = Addition(
                        tabulated_sin,
                        cos_u,
                        precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                        tag = "sin_P_cos_u"
                      )

    sin_P_cos_u_P_C2_u2_sin = Addition(
                                sin_P_cos_u,
                                sin_C2_u2,
                                precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                                tag = "sin_P_cos_u_P_C2_u2_sin"
                              )

    scheme = Addition(
                sin_P_cos_u_P_C2_u2_sin,
                S2_u3_cos,
                precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
              )

    return scheme


if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_fastsincos", default_output_file = "new_fastsincos.c" )
  # argument extraction 
  cos_output = arg_template.test_flag_option("--sin", False, True, parse_arg = arg_template.parse_arg, help_str = "select cos output") 
  enable_subexpr_sharing = arg_template.test_flag_option("--enable-subexpr-sharing", True, False, parse_arg = arg_template.parse_arg, help_str = "force subexpression sharing")
  table_size_log = arg_template.extract_option_value("--table-size-log", 8, parse_arg = arg_template.parse_arg, help_str = "logarithm of the table size to be used", processing = lambda x: int(x))

  result_precision = arg_template.extract_option_value("--result-precision", ML_Binary32, parse_arg = arg_template.parse_arg, help_str = "result precision", processing = precision_parser)
  Log.report(Log.Info, "result_precision is %s " % result_precision)

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
                     table_size_log            = table_size_log,
                     result_precision          = result_precision,
                     cos_output                = cos_output)
  ml_fastsincos.gen_implementation(display_after_opt = arg_template.display_after_opt, enable_subexpr_sharing = enable_subexpr_sharing)
