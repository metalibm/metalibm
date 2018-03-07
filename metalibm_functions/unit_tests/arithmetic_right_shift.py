# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *

from metalibm_core.utility.ml_template import *


class ML_UT_ArithmeticRightShift(ML_Function("ml_ut_arithmetic_right_shift")):
  def __init__(self,
               arg_template,
               precision = ML_Int32,
               libm_compliant = True,
               debug_flag = False,
               output_file = "ut_arithmetic_right_shift.c",
               function_name = "ut_arithmetic_right_shift"):
    # precision argument extraction
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self,
      base_name = "ut_arithmetic_right_shift",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      debug_flag = debug_flag,
      arg_template = arg_template
    )

    self.precision = precision


  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)

    result = BitArithmeticRightShift(vx, 8)

    return Return(result)

  def numeric_emulate(self, *args):
    return SollyaObject(int(args[0]) >> 8)


def run_test(args):
  ml_ut_arithmetic_right_shift = ML_UT_ArithmeticRightShift(args)
  ml_ut_arithmetic_right_shift.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
      "new_ut_arithmetic_right_shift",
      default_output_file = "new_ut_arithmetic_right_shift.c",
      )
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)
