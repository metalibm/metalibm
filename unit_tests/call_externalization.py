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
from metalibm_core.code_generation.fixed_point_backend import FixedPointBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 




class ML_UT_CallExternaliation(ML_Function("ml_ut_call_externalization")):
  def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = FixedPointBackend(), 
                 output_file = "ut_call_externalization.c", 
                 function_name = "ut_call_externalization"):
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_call_externalization",
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


  def generate_function_list(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    sub_computation = Multiplication(vx, vx, precision = self.precision)
    sub_computation, ext_function = self.externalize_call(sub_computation, [vx])

    sub_computation2 = Addition(sub_computation, vx, precision = self.precision)
    sub_computation2, ext_function2 = self.externalize_call(sub_computation2, [vx, sub_computation])


    result = Return(sub_computation2)


    # dummy scheme to make functionnal code generation
    scheme = Statement(result)
    self.implementation.set_scheme(scheme)

    return [ext_function, ext_function2, self.implementation]

if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_ut_call_externalization", default_output_file = "new_ut_call_externalization.c" )
  arg_template.sys_arg_extraction()


  ml_ut_call_externalization = ML_UT_CallExternaliation(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)

  ml_ut_call_externalization.gen_implementation(display_after_gen = False, display_after_opt = False)


