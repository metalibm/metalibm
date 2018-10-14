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
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import (
    execute_gappa_script_extract,
    is_gappa_installed
)
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.debug_utils import * 

class ML_UT_GappaCode(ML_Function("ml_ut_gappa_code")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_gappa_code.c",
        "function_name": "ut_gappa_code",
        "precision": ML_Binary32,
        "target": MPFRProcessor(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", ML_Binary32)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    # declaring free Variable y
    vy = Variable("y", precision = ML_Exact)

    # declaring expression with vx variable
    expr = vx * vx - vx * 2
    # declaring second expression with vx variable
    expr2 = vx * vx - vx

    # optimizing expressions (defining every unknown precision as the
    # default one + some optimization as FMA merging if enabled)
    opt_expr = self.optimise_scheme(expr)
    opt_expr2 = self.optimise_scheme(expr2)

    # setting specific tag name for optimized expression (to be extracted 
    # from gappa script )
    opt_expr.set_tag("goal")
    opt_expr2.set_tag("new_goal")
    
    # defining default goal to gappa execution
    gappa_goal = opt_expr 

    # declaring EXACT expression to be used as hint in Gappa's script
    annotation = self.opt_engine.exactify(vy * (1 / vy))

    # the dict var_bound is used to limit the DAG part to be explored when
    # generating the gappa script, each pair (key, value), indicate a node to stop at <key>
    # and a node to replace it with during the generation: <node>,
    # <node> must be a Variable instance with defined interval
    # vx.get_handle().get_node() is used to retrieve the node instanciating the abstract node <vx>
    # after the call to self.optimise_scheme
    var_bound = {
      vx.get_handle().get_node(): Variable("x", precision = ML_Binary32, interval = vx.get_interval())
    } 
    # generating gappa code to determine interval for <opt_expr>
    gappa_code = self.gappa_engine.get_interval_code(opt_expr, var_bound)

    # add a manual hint to the gappa code
    # which state thtat vy * (1 / vy) -> 1 { vy <> 0 };
    self.gappa_engine.add_hint(gappa_code, annotation, Constant(1, precision = ML_Exact), Comparison(vy, Constant(0, precision = ML_Integer), specifier = Comparison.NotEqual, precision = ML_Bool))
    
    # adding the expression <opt_expr2> as an extra goal in the gappa script
    self.gappa_engine.add_goal(gappa_code, opt_expr2)

    # executing gappa on the script generated from <gappa_code>
    # extract the result and store them into <gappa_result>
    # which is a dict indexed by the goals' tag
    if is_gappa_installed():
        gappa_result = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))
        Log.report(Log.Info, "eval error: ", gappa_result["new_goal"])
    else:
        Log.report(Log.Warning, "gappa was not installed: unable to check execute_gappa_script_extract")

    # dummy scheme to make functionnal code generation
    scheme = Statement(Return(vx))

    return scheme

def run_test(args):
  ml_ut_gappa_code = ML_UT_GappaCode(args)
  ml_ut_gappa_code.gen_implementation(display_after_gen=False, display_after_opt=False)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_GappaCode.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)

