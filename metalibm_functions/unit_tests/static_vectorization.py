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
# created:          Feb  3rd, 2016
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for ML static vectorization
###############################################################################


import sys

from sollya import Interval

from metalibm_core.core.ml_function import  ML_FunctionBasis

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import *

from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.core.ml_vectorizer import StaticVectorizer

from metalibm_core.utility.ml_template import *

from metalibm_functions.unit_tests.utils import TestRunner


class ML_UT_StaticVectorization(ML_FunctionBasis, TestRunner):
  function_name = "ml_ut_static_vectorization"
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_static_vectorization.c",
        "function_name": "ut_static_vectorization",
        "precision": ML_Binary32,
        "target": VectorBackend(),
        "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"],
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_function_list(self):
    vector_size = 2

    # declaring function input variable
    #vx = self.implementation.add_input_variable("x", self.precision)
    vx = Variable("x", precision = self.precision) 
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    cond0 = Test(vx, specifier = Test.IsInfOrNaN, likely = False)
    cond1 = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, likely = True)

    exp0 = vx
    exp1 = vx + vx * vx + Constant(1, precision = self.precision)
    exp2 = vx * vx * vx 
    scheme = Statement(
      ConditionBlock(cond0,
        Return(exp0),
        ConditionBlock(cond1,
          Return(exp1),
          Return(exp2)
        )
      )
    )

    return self.generate_vector_implementation(scheme, [vx], vector_size)

  @staticmethod
  def __call__(args):
    ml_ut_static_vectorization = ML_UT_StaticVectorization(args)
    ml_ut_static_vectorization.gen_implementation()
    return True

run_test = ML_UT_StaticVectorization

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_StaticVectorization.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)


