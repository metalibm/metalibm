# -*- coding: utf-8 -*-
# vim: sts=2 sw=2

import sys

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
    )

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Result, FO_Arg
    )

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import *

from metalibm_core.opt.ml_blocks import generate_count_leading_zeros

class ML_Lzcnt(ML_Function("ml_lzcnt")):
  def __init__(self,
               arg_template = DefaultArgTemplate,
               precision = ML_Int32,
               abs_accuracy = None,
               libm_compliant = True,
               debug_flag = False,
               fuse_fma = True,
               fast_path_extract = True,
               target = GenericProcessor(),
               output_file = "my_lznct.c",
               function_name = "my_lzcnt"):
    # initializing I/O precision
    io_precisions = [precision] * 2
    self.precision = precision

    # initializing base class
    ML_FunctionBasis.__init__(self,
      base_name = "lzcnt",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = abs_accuracy,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template
    )


  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.precision)

    return Return(generate_count_leading_zeros(vx))

  def numeric_emulate(self, input_value):
    input_value = int(input_value)
    if input_value == 0:
      return float(self.precision.get_c_bit_size())
    n = 0
    while (input_value & 0x80000000) == 0:
      n += 1
      input_value <<= 1
    return float(n);

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_function_name = "new_lzcnt", default_output_file = "new_lzcnt.c" )
  args = arg_template.arg_extraction()

  ml_lzcnt          = ML_Lzcnt(args)

  ml_lzcnt.gen_implementation()
