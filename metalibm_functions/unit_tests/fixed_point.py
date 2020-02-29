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

from sollya import Interval

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import *

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.utility.ml_template import *


class ML_UT_FixedPoint(ML_Function("ml_ut_fixed_point")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_fixed_point.c",
        "function_name": "ut_fixed_point",
        "target": FixedPointBackend.get_target_instance(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True,
        "precision": ML_Int32,
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", ML_Int32)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    fixed_format = ML_Custom_FixedPoint_Format(3, 29, False)


    acc_format = ML_Custom_FixedPoint_Format(6, 58, False)

    c = Constant(2, precision = acc_format)

    ivx = TypeCast(vx, precision = fixed_format)
    add_ivx = Addition(
                c, 
                Multiplication(ivx, ivx, precision = acc_format),
                precision = acc_format
              )
    result = add_ivx # Conversion(add_ivx, precision = self.precision)

    # dummy scheme to make functionnal code generation
    scheme = Statement(Return(result))

    return scheme


## Test execution function
def run_test(args):
  ml_ut_fixed_point = ML_UT_FixedPoint(args)
  ml_ut_fixed_point.gen_implementation(display_after_gen = False, display_after_opt = False)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_FixedPoint.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)

