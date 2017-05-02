# -*- coding: utf-8 -*-

import sys
import random

from sollya import S2

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.core.ml_table import ML_NewTable


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

class ML_UT_MultiAryFunction(ML_Function("ml_ut_multi_ary_function")):
  def __init__(self, 
                 arg_template,
                 output_file = "ut_multi_ary_function.c", 
                 function_name = "ut_multi_ary_function"):
    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_multi_ary_function",
      function_name = function_name,
      output_file = output_file,

      arity = 3,

      arg_template = arg_template,
    )



  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.get_input_precision(0))
    vy = self.implementation.add_input_variable("y", self.get_input_precision(1))
    vz = self.implementation.add_input_variable("z", self.get_input_precision(2))

    scheme = Statement(
      Return(
        Addition(
          Multiplication(vx, vy, precision = self.precision), 
          vz,
          precision = self.precision
        )
      )
    )
    return scheme

  def numeric_emulate(self, x, y, z):
    return x * y + z


def run_test(args):
  ml_ut_multi_ary_function = ML_UT_MultiAryFunction(args)
  ml_ut_multi_ary_function.gen_implementation(display_after_gen = True, display_after_opt = True)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate("new_ut_multi_ary_function", default_output_file = "new_ut_multi_ary_function.c" )
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
