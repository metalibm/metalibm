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
from metalibm_functions.unit_tests.utils import TestRunner 

class ML_UT_Lzcnt(ML_Function("ml_lzcnt"), TestRunner):
  def __init__(self,
      arg_template = DefaultArgTemplate(
        output_file = "ut_block_lzcnt.c",
        function_name = "ut_lzcnt",
        precision = ML_Int32,
        auto_test_range = Interval(0, 2**31),
        auto_test_execute = 1000)
      ):
    # initializing I/O precision
    self.precision = arg_template.precision

    # initializing base class
    ML_FunctionBasis.__init__(self, arg_template = arg_template)


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

  @staticmethod
  def get_default_args(**kw):
    default_arg = {
      "function_name": "new_lzcnt",
      "output_file": "ut_lzcnt.c",
      "precision": ML_Int32
    }
    default_arg.update(**kw)
    return DefaultArgTemplate(**default_arg)

  @staticmethod
  def __call__(args):
    # just ignore args here and trust default constructor? seems like a bad idea.
    ml_ut_block_lzcnt = ML_UT_Lzcnt(args)
    ml_ut_block_lzcnt.gen_implementation()
                                         
    return True

run_test = ML_UT_Lzcnt


if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
      default_function_name = "new_lzcnt",
      default_output_file = "new_lzcnt.c",
      default_arg = DefaultArgTemplate(precision = ML_Int32)
      )
  # Overwrite default args by command line args if any
  args = arg_template.arg_extraction()

  run_test(args)

