# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor


from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 

class ML_UT_PayneHanek(ML_Function("ml_ut_payne_hanek")):
  def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = MPFRProcessor(), 
                 output_file = "ut_payne_hanek.c", 
                 function_name = "ut_payne_hanek"):
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_payne_hanek",
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
    int_precision = {
      ML_Binary32: ML_Int32,
      ML_Binary64: ML_Int64
    }[self.precision]
    vx = self.implementation.add_input_variable("x", ML_Binary64)
    k = 4
    frac_pi = S2**k/pi 

    red_stat, red_vx, red_int = generate_payne_hanek(vx, frac_pi, self.precision, k = k, n= 100) 
    C32 = Constant(32, precision = int_precision)
    red_int_f = Conversion(Select(red_int < Constant(0, precision = int_precision), red_int + C32, red_int), precision = self.precision)

    scheme = Statement(
      red_stat,
      Return(red_vx + red_int_f)
    )

    return scheme

if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_ut_payne_hanek", default_output_file = "new_ut_payne_hanek.c" )
  arg_template.sys_arg_extraction()


  ml_ut_payne_hanek = ML_UT_PayneHanek(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)

  ml_ut_payne_hanek.gen_implementation(display_after_gen = False, display_after_opt = False)
