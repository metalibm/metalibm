# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Feb  3rd, 2016
# last-modified:    Feb  5th, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for ML static vectorization 
###############################################################################


import sys

from sollya import S2

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine

from metalibm_core.core.ml_vectorizer import StaticVectorizer

from metalibm_core.utility.ml_template import *


class ML_UT_OpenCLCode(ML_Function("ml_ut_opencl_code")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_opencl_code.c",
        "function_name": "ut_opencl_code",
        "precision": ML_Binary32,
        "target": FixedPointBackend(),
        "vector_size": 2,
        "language": C_Code,
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


  def generate_scheme(self):
    vector_size = 2

    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    cond0 = Test(vx, specifier = Test.IsInfOrNaN, likely = False)
    cond1 = Comparison(vx, Constant(0, precision = self.precision), specifier = Comparison.GreaterOrEqual, likely = True)

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

    return scheme


def run_test(args):
  ml_ut_opencl_code = ML_UT_OpenCLCode(args)
  ml_ut_opencl_code.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_OpenCLCode.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)



