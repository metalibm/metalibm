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
from metalibm_functions.unit_tests.utils import TestRunner
from sollya import SollyaObject

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.ml_formats import ML_Int32
from metalibm_core.core.ml_operations import BitArithmeticRightShift, Return

from metalibm_core.utility.ml_template import DefaultFunctionArgTemplate


class ML_UT_ArithmeticRightShift(ML_FunctionBasis, TestRunner):
  function_name = "ml_ut_arithmetic_right_shift"
  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_arithmetic_right_shift.c",
        "function_name": "ut_arithmetic_right_shift",
        "precision": ML_Int32,
    }
    default_args.update(kw)
    return DefaultFunctionArgTemplate(**default_args)


  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)

    result = BitArithmeticRightShift(vx, 8)

    return Return(result)

  def numeric_emulate(self, *args):
    return SollyaObject(int(args[0]) >> 8)

  @staticmethod
  def __call__(args):
    ml_ut_arithmetic_right_shift = ML_UT_ArithmeticRightShift(args)
    ml_ut_arithmetic_right_shift.gen_implementation()
    return True

run_test = ML_UT_ArithmeticRightShift


if __name__ == "__main__":
  # auto-test
  arg_template = DefaultFunctionArgTemplate(default_args=ML_UT_ArithmeticRightShift.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
