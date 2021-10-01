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
from sollya import SollyaObject

from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.core.ml_operations import (ML_ArithmeticOperation,
  ML_LeafNode, Multiplication, Addition, Constant,
  Statement, Return)

from metalibm_core.core.ml_formats import ML_Int32

from metalibm_core.code_generation.code_constant import C_Code 

from metalibm_core.utility.ml_template import (
  MetaFunctionArgTemplate, DefaultFunctionArgTemplate)


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

class ML_UT_LatencyEvaluation(ML_FunctionBasis):
  function_name = "ml_ut_latency_evaluation"

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_latency_evaluation.c",
        "function_name": "ut_latency_evaluation",
        "precision": ML_Int32,
    }
    default_args.update(kw)
    return DefaultFunctionArgTemplate(**default_args)

  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.precision)

    operations = Multiplication(
      Addition(vx, vx, precision = self.precision),
      Constant(3, precision = self.precision),
      precision = self.precision
    )

    latency_pass  = LatencyEvaluator(self.processor)
    latency_value = latency_pass.evaluate(operations)
    print("latency evaluation result: {}".format(latency_value))
    
    scheme = Statement(Return(operations, precision = self.precision))
    return scheme

  @staticmethod
  def __call__(args):
    ml_ut_latency_evaluation = ML_UT_LatencyEvaluation(args)
    ml_ut_latency_evaluation.gen_implementation(display_after_gen = True, display_after_opt = True)
    return True

run_test = ML_UT_LatencyEvaluation

if __name__ == "__main__":
  # auto-test
  arg_template = MetaFunctionArgTemplate(default_arg=ML_UT_LatencyEvaluation.get_default_args())
  args = arg_template.arg_extraction()

  if run_test.__call__(args):
    exit(0)
  else:
    exit(1)
