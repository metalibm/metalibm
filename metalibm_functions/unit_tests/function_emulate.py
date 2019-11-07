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

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Arg, FO_Result
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.utility.debug_utils import * 

class ML_UT_FunctionFormat(ML_Function("ml_ut_function_format")):
  def __init__(self, args=DefaultArgTemplate):
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_function_emulate.c",
        "function_name": "ut_function_emulate",
        "precision": ML_Binary32,
        "libm_compliant": True,
        "debug_flag": False,
        "fuse_fma": True,
        "fast_path_extract": True,
        "target": MPFRProcessor(),
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    mpfr_x = Conversion(vx, precision = ML_Mpfr_t)


    emulate_func_name = "mpfr_exp"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t], ML_Mpfr_t, emulate_func_op)
    emulate_func_op.declare_prototype = emulate_func
    mpfr_call = Conversion(emulate_func(mpfr_x), precision = self.precision)

    scheme = Statement(Return(mpfr_call))

    return scheme

def run_test(args):
  ml_ut_function_format = ML_UT_FunctionFormat(args)
  ml_ut_function_format.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_FunctionFormat.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)

