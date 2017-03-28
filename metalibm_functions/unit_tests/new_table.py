# -*- coding: utf-8 -*-

import sys
import random

from sollya import S2

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table, ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 

class ML_UT_NewTable(ML_Function("ml_ut_new_table")):
  def __init__(self, 
                 arg_template,
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = MPFRProcessor(), 
                 output_file = "ut_new_table.c", 
                 function_name = "ut_new_table"):
    # precision argument extraction
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_new_table",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template,
    )

    self.precision = precision


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.precision)

    table_size = 16
    row_size   = 2

    new_table = ML_NewTable(dimensions = [table_size, row_size], storage_precision = self.precision)
    for i in xrange(table_size):
      new_table[i][0]= i 
      new_table[i][1]= i + 1

    index = Modulo(vx, Constant(table_size, precision = ML_Int32), precision = ML_Int32)
    load_value_lo = TableLoad(new_table, index, Constant(0, precision = ML_Int32), precision = self.precision)
    load_value_hi = TableLoad(new_table, index, Constant(1, precision = ML_Int32), precision = self.precision)

    scheme = Statement(
      Return(
        Addition(
          load_value_lo,
          load_value_hi,
          precision = self.precision
        ),
        precision = self.precision,
      )
    )
    return scheme

  def numeric_emulate(self, input_value):
    table = [[i, i+1] for i in xrange(16)]
    print input_value
    index = int(input_value) % 16
    lo = table[index][0]
    hi = table[index][1]
    return lo + hi


def run_test(args):
  ml_ut_new_table = ML_UT_NewTable(args)
  ml_ut_new_table.gen_implementation(display_after_gen = True, display_after_opt = True)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate("new_ut_new_table", default_output_file = "new_ut_new_table.c" )
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
