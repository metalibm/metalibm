# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 

class ML_UT_GappaCode(ML_Function("ml_ut_gappa_code")):
  def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = MPFRProcessor(), 
                 output_file = "ut_gappa_code.c", 
                 function_name = "ut_gappa_code"):
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_gappa_code",
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


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", ML_Binary32)
    vx.set_interval(Interval(-1, 1))

    vy = Variable("y", precision = ML_Exact)

    expr = vx * vx - vx * 2

    opt_expr = self.optimise_scheme(expr)
    
    gappa_goal = opt_expr 

    annotation = self.opt_engine.exactify(vy * (1 / vy))

    print annotation.get_str(depth = True, display_precision = True)

    gappa_code = self.gappa_engine.get_interval_code(opt_expr, {vx.get_handle().get_node(): Variable("x", precision = ML_Binary32, interval = vx.get_interval())})
    self.gappa_engine.add_hint(gappa_code, annotation, Constant(1, precision = ML_Exact), Comparison(vy, Constant(0, precision = ML_Integer), specifier = Comparison.NotEqual, precision = ML_Bool))

    eval_error = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))["goal"]

    print "eval error: ", eval_error

    scheme = Statement(Return(vx))

    return scheme

if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_ut_loop_operation", default_output_file = "new_ut_loop_operation.c" )
  arg_template.sys_arg_extraction()


  ml_ut_gappa_code = ML_UT_GappaCode(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)

  ml_ut_gappa_code.gen_implementation(display_after_gen = True, display_after_opt = True)

