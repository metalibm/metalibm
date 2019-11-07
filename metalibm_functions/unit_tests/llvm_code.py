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
# created:          Apr  5th, 2018
# last-modified:    Sep 20th, 2018
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for LLVM-IR code Generation
###############################################################################


import sys

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import (
    Constant, Comparison, ConditionBlock, Return, Statement
)
from metalibm_core.core.ml_formats import ML_Int32, ML_Bool

from metalibm_core.targets.common.llvm_ir import LLVMBackend

from metalibm_core.code_generation.code_constant import LLVM_IR_Code

from metalibm_functions.unit_tests.utils import TestRunner


from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)


class ML_UT_LLVMCode(ML_FunctionBasis, TestRunner):
  function_name = "ml_ut_llvm_code"
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_llvm_code.ll",
        "function_name": "ut_llvm_code",
        "precision": ML_Int32,
        "target": LLVMBackend(),
        "language": LLVM_IR_Code,
        "fast_path_extract": True,
        "fuse_fma": False,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)
    vy = self.implementation.add_input_variable("y", self.precision)

    Cst0 = Constant(5, precision=self.precision)
    Cst1 = Constant(7, precision=self.precision)
    comp = Comparison(vx, vy, specifier=Comparison.Greater, precision=ML_Bool, tag="comp")
    comp_eq = Comparison(vx, vy, specifier=Comparison.Equal, precision=ML_Bool, tag="comp_eq")

    scheme = Statement(
        ConditionBlock(
            comp,
            Return(
                vy,
                precision=self.precision
            ),
            ConditionBlock(
                comp_eq,
                Return(
                    vx + vy * Cst0 - Cst1,
                    precision=self.precision
                )
            )
        ),
        ConditionBlock(
            comp_eq,
            Return(Cst1 * vy, precision=self.precision)
        ),
        Return(vx * vy, precision=self.precision)
    )

    return scheme

  @staticmethod
  def __call__(args):
    ml_ut_llvm_code = ML_UT_LLVMCode(args)
    ml_ut_llvm_code.gen_implementation()
    return True


run_test = ML_UT_LLVMCode

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_LLVMCode.get_default_args())
  args = arg_template.arg_extraction()

  if ML_UT_LLVMCode.__call__(args):
    exit(0)
  else:
    exit(1)



