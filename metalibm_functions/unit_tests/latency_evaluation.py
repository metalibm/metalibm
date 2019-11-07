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

from sollya import SollyaObject

S2 = SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine


from metalibm_core.utility.ml_template import *


class LatencyEvaluator:
  def __init__(self, target, language = C_Code):
    self.target = target
    ## dictionnary storing pair (optree, latency) of already 
    #  evaluated nodes
    self.latency_map = {}
    self.language = language

  ## Evaluate the critical path latency of 
  #  the evaluation of @p optree
  def evaluate(self, optree):
    if optree in self.latency_map:
      return self.latency_map[optree]
    else:
      if isinstance(optree, ML_ArithmeticOperation):
        optree_impl = self.target.get_implementation(optree, language = self.language)
        latency = optree_impl.get_speed_measure() + max([self.evaluate(inp) for inp in optree.get_inputs()])
        self.latency_map[optree] = latency
        return latency
      if isinstance(optree, ML_LeafNode):
        return 1.0
      else:
        latency = max([self.evaluate(inp) for inp in optree.get_inputs()])
        self.latency_map[optree] = latency
        return latency

class ML_UT_LatencyEvaluation(ML_Function("ml_ut_latency_evaluation")):
  def __init__(self, 
                 arg_template,
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = MPFRProcessor(), 
                 output_file = "ut_latency_evaluation.c", 
                 function_name = "ut_latency_evaluation"):
    # precision argument extraction
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_latency_evaluation",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template,
    )

    self.precision = precision


  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.precision)

    operations = Multiplication(
      Addition(vx, vx, precision = self.precision),
      Constant(3, precision = self.precision),
      precision = self.precision
    )

    latency_pass  = LatencyEvaluator(self.processor)
    latency_value = latency_pass.evaluate(operations)
    print "latency evaluation result: {}".format(latency_value)
    
    scheme = Statement(
              Return(operations, precision = self.precision)
            )
    return scheme

def run_test(args):
  ml_ut_latency_evaluation = ML_UT_LatencyEvaluation(args)
  ml_ut_latency_evaluation.gen_implementation(display_after_gen = True, display_after_opt = True)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate("new_ut_latency_eval", default_output_file = "new_ut_latency_eval.c" )
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
