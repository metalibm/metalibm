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
import random


from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import *

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table, ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value

from metalibm_core.utility.debug_utils import *

class ML_UT_ImplicitIntervalEval(ML_Function("ml_ut_implicit_interval_eval")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Exponential,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_implicit_interval_eval.c",
        "function_name": "ut_implicit_interval_eval",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": MPFRProcessor()
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    # map of expected interval values
    expected_interval = {}

    vx_interval = Interval(-1, 1)
    vx = self.implementation.add_input_variable("x", self.precision, interval = vx_interval)
    expected_interval[vx] = vx_interval

    cst = Constant(7, tag = "cst")
    cst_interval = Interval(7)
    expected_interval[cst] = cst_interval

    shl = BitLogicLeftShift(
        NearestInteger(vx)
        , 2, interval = 2 *  vx_interval,
        tag = "shl"
    )
    shl_interval = 2 * vx_interval
    expected_interval[shl] = shl_interval

    r = vx + cst * vx + shl - cst
    r.set_attributes(tag = "r")

    r_interval = vx_interval + cst_interval * vx_interval  + shl_interval - cst_interval
    expected_interval[r] = r_interval

    for var in [vx, cst, r, shl]:
        if var.get_interval() != expected_interval[var]:
            Log.report(
                Log.Error,
                "unexpected interval for {}: got {}, expected {}".format(
                    var.get_str(display_precision = True),
                    var.get_interval(),
                    expected_interval[var]
                )
            )
        else:
            Log.report(Log.Info, "node {}: {} vs {}".format(var.get_tag(), var.get_interval(), expected_interval[var]))

    return Statement()

  def numeric_emulate(self, input_value):
    raise NotImplementedError


def run_test(args):
  ml_ut_implicit_interval_eval = ML_UT_ImplicitIntervalEval(args)
  ml_ut_implicit_interval_eval.gen_implementation(
    display_after_gen = True, display_after_opt = True
  )
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_ImplicitIntervalEval.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
