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
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction, FunctionGroup
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  


class ML_UT_VectorCode(ML_Function("ml_ut_vector_code")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_vector_code.c",
        "function_name": "ut_vector_code",
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
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)
    vy = self.implementation.add_input_variable("y", self.precision)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))


    vec = Variable("vec", precision = v2float32, var_type = Variable.Local)

    vec2 = Multiplication(vec, vec, precision = v2float32)
    vec3 = Addition(vec, vec2, precision = v2float32)

    large_vector = VectorAssembling(vec2, vec3, precision=v4float32)
    sub_vector = SubVectorExtract(large_vector, 0, 3, precision=v2float32)

    result = Addition(sub_vector[0], sub_vector[1], precision = ML_Binary32)

    scheme = Statement(
      ReferenceAssign(vec[0], vx),
      ReferenceAssign(vec[1], vy),
      Return(result)
    )

    # dummy scheme to make functionnal code generation
    self.implementation.set_scheme(scheme)

    return FunctionGroup([self.implementation])


def run_test(args):
  ml_ut_vector_code = ML_UT_VectorCode(args)
  ml_ut_vector_code.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_VectorCode.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


