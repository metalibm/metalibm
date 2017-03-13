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

from sollya import S2, Interval

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


from metalibm_core.utility.debug_utils import * 




class ML_UT_StaticVectorization(ML_Function("ml_ut_static_vectorization")):
  def __init__(self, 
                 arg_template,
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = FixedPointBackend(), 
                 output_file = "ut_static_vectorization.c", 
                 function_name = "ut_static_vectorization"):
    # precision argument extraction
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_static_vectorization",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template
    )

    self.precision = precision


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

def run_test(args):
  ml_ut_static_vectorization = ML_UT_StaticVectorization(args)
  ml_ut_static_vectorization.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_function_name = "new_ut_static_vectorization", default_output_file = "new_ut_static_vectorization.c" )
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)


