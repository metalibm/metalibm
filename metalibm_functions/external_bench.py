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

import sollya

from sollya import (
        Interval, ceil, floor, round, inf, sup, pi, log, exp, cos, sin,
        guessdegree, dirtyinfnorm
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.code_generation.code_function import CodeFunction, FunctionGroup

from metalibm_core.core.ml_vectorizer import vectorize_format

from metalibm_core.utility.ml_template import (
    MultiAryArgTemplate, DefaultMultiAryArgTemplate, precision_parser)
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

def atan2_emulate(vy, vx):
    if vx > 0:
        return sollya.atan(vy / vx)
    elif vy < 0:
        # vy / vx > 0
        return -sollya.pi + sollya.atan(vy / vx)
    else:
        # vy > 0, vy / vx < 0
        return sollya.pi + sollya.atan(vy / vx)


class ML_ExternalBench(ML_Function("ml_external_bench")):
  """ Implementation of external bench function wrapper """
  def __init__(self, args=DefaultMultiAryArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)
    # initializing specific properties
    self.headers = args.headers
    self.libraries = args.libraries
    self.extra_src_files = args.extra_src_files
    self.bench_function_name = args.bench_function_name
    self.emulate = args.emulate
    self.arity = args.arity
    self.function_input_vector_size = args.function_input_vector_size
    if len(self.auto_test_range) != self.arity:
        self.auto_test_range = [self.auto_test_range[0]] * self.arity
    if len(self.bench_test_range) != self.arity:
        self.bench_test_range = [self.bench_test_range[0]] * self.arity

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_ExternalBench
        builtin from a default argument mapping overloaded with @p kw """
    default_args_exp = {
        "output_file": "bench.c",
        "function_name": "external_bench_wrapper",
        "extra_src_files": [],
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor.get_target_instance()
    }
    default_args_exp.update(kw)
    return DefaultMultiAryArgTemplate(**default_args_exp)


  def generate_function_list(self):
    Log.report(Log.Verbose, "generating external bench for function {} with vector-size {}/{}".format(self.bench_function_name, self.vector_size, self.function_input_vector_size))
    if self.function_input_vector_size > 1:
        output_format = vectorize_format(self.precision, self.function_input_vector_size)
    else:
        output_format = self.precision
    benched_function = CodeFunction(self.bench_function_name, output_format=output_format, external=True, vector_size=self.function_input_vector_size)
    # need to overwrite self.implementation as it is used to determine
    # if vectorization is required in ml_function
    self.implementation = benched_function
    for arg_format in self.get_input_precisions():
        arg_format = vectorize_format(arg_format, self.function_input_vector_size)
        benched_function.register_new_input_variable(Variable("", precision=arg_format))
    return FunctionGroup([benched_function])

  def get_extra_build_opts(self):
    return self.extra_src_files

  def numeric_emulate(self, *args):
    return self.emulate(*args)

if __name__ == "__main__":
  # auto-test
  arg_template = MultiAryArgTemplate(default_arg=ML_ExternalBench.get_default_args())
  def precision_list_parser(s):
    return [precision_parser(p) for p in s.split(",")]

  # argument extraction
  arg_template.get_parser().add_argument(
    "--function", dest="bench_function_name", default="expf",
    action="store", type=str, help="name of the function to be benched")
  arg_template.get_parser().add_argument(
    "--headers", dest="headers", default=[], action="store",
    type=lambda s: s.split(","),
    help="comma separated list of required headers")
  arg_template.get_parser().add_argument(
    "--extra-src-files", dest="extra_src_files", default=[], action="store",
    type=lambda s: s.split(","),
    help="comma separated list of required libraries")
  arg_template.get_parser().add_argument(
    "--function-input-vector-size", default=1, action="store",
    type=int,
    help="number of elements in function input vector (1 for scalar)")
  arg_template.get_parser().add_argument(
    "--libraries", dest="libraries", default=[], action="store",
    type=lambda s: s.split(","),
    help="comma separated list of required libraries")

  def local_eval(s):
    return eval(s)
  arg_template.get_parser().add_argument(
    "--emulate", dest="emulate", default=lambda x: x, action="store",
    type=local_eval, help="function numeric emulation")

  args = arg_template.arg_extraction()
  ml_sincos = ML_ExternalBench(args)
  ml_sincos.gen_implementation()
