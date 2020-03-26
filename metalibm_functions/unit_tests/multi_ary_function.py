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
import sys
import random


from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.core.ml_table import ML_NewTable


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

class ML_UT_MultiAryFunction(ML_Function("ml_ut_multi_ary_function")):
  arity = 3
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_multi_ary_function.c",
        "function_name": "ut_multi_ary_function",
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


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

  @staticmethod
  def __call__(args):
    # just ignore args here and trust default constructor? seems like a bad idea.
    ml_ut_multiary_function = ML_UT_MultiAryFunction(args)
    ml_ut_multiary_function.gen_implementation()

    return True


run_test = ML_UT_MultiAryFunction


if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_MultiAryFunction.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
