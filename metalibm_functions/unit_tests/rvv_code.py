# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:          Oct  3rd, 2021
# last-modified:    Oct  3rd, 2021
# Author(s): Nicolas Brunie
###############################################################################
import sys

from sollya import Interval

from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.ml_operations import Addition, TableStore, TableLoad, Statement
from metalibm_core.core.ml_formats import ML_Int32, ML_Binary32, ML_Void
from metalibm_core.core.ml_complex_formats import ML_Binary32_p
from metalibm_core.core.vla_common import (VLAGetLength, VLAOperation)

from metalibm_core.code_generation.code_function import FunctionGroup

from metalibm_core.utility.ml_template import DefaultFunctionArgTemplate, MetaFunctionArgTemplate

from metalibm_core.targets.riscv.riscv_vector import RVV_vBinary32_m1, RISCV_RVV64

from metalibm_functions.unit_tests.utils import TestRunner


class ML_UT_RVVCode(ML_FunctionBasis, TestRunner):
  function_name = "ml_ut_vector_code"
  def __init__(self, args=DefaultFunctionArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_rvv_code.c",
        "function_name": "ut_rvv_code",
        "precision": ML_Void,
        "target": RISCV_RVV64.get_target_instance(),
        "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"],
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultFunctionArgTemplate(**default_args)

  def generate_function_list(self):
    # declaring function input variable
    px = self.implementation.add_input_variable("px", ML_Binary32_p)
    py = self.implementation.add_input_variable("py", ML_Binary32_p)
    l = self.implementation.add_input_variable("l", ML_Int32)

    l = VLAGetLength(l, precision=ML_Int32)
    vx = VLAOperation(px, l, specifier=TableLoad, precision=RVV_vBinary32_m1)
    vadd = VLAOperation(vx, vx, l, specifier=Addition, precision=RVV_vBinary32_m1)
    scheme = Statement(
      VLAOperation(py, vadd, l, specifier=TableStore, precision=ML_Void)
    )
    # dummy scheme to make functionnal code generation
    self.implementation.set_scheme(scheme)

    return FunctionGroup([self.implementation])

  @staticmethod
  def __call__(args):
    ml_ut_vector_code = ML_UT_RVVCode(args)
    ml_ut_vector_code.gen_implementation()
    return True

run_test = ML_UT_RVVCode

if __name__ == "__main__":
  # auto-test
  arg_template = MetaFunctionArgTemplate(default_arg=ML_UT_RVVCode.get_default_args())
  args = arg_template.arg_extraction()

  if run_test.__call__(args):
    exit(0)
  else:
    exit(1)


