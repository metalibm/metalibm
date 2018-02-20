# -*- coding: utf-8 -*-

import sys

from sollya import S2

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 

class ML_UT_FunctionFormat(ML_Function("ml_ut_function_format")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_function_format.c",
        "function_name": "ut_function_format",
        "precision": ML_Binary32,
        "target": MPFRProcessor(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    mpfr_x = Conversion(vx, precision = ML_Mpfr_t)
    result = mpfr_x + mpfr_x
    result.set_precision(ML_Mpfr_t)

    scheme = Statement(Return(Conversion(result, precision = self.precision)))

    return scheme

def run_test(args):
  ml_ut_function_format = ML_UT_FunctionFormat(args)
  ml_ut_function_format.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_FunctionFormat.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)

