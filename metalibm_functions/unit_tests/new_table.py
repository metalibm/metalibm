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


class ML_UT_NewTable(ML_Function("ml_ut_new_table")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_new_table.c",
        "function_name": "ut_new_table",
        "precision": ML_Binary32,
        "target": MPFRProcessor(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.precision)

    table_size = 16
    row_size   = 2

    new_table = ML_NewTable(dimensions = [table_size, row_size], storage_precision = self.precision)
    for i in range(table_size):
      new_table[i][0]= i 
      new_table[i][1]= i + 1

    index = Modulo(vx, Constant(table_size, precision = ML_Int32), precision = ML_Int32)
    load_value_lo = TableLoad(new_table, index, Constant(0, precision = ML_Int32), precision = self.precision)
    load_value_hi = TableLoad(new_table, index, Constant(1, precision = ML_Int32), precision = self.precision)

    Log.report(Log.Info, "table interval: {}".format(new_table.get_interval()))

    out_table = ML_NewTable(dimensions = [table_size], storage_precision = self.precision, empty = True)

    result = Addition(
      load_value_lo,
      load_value_hi,
      precision = self.precision
    )

    scheme = Statement(
      TableStore(
        result, 
        out_table,
        Constant(13, precision = ML_Int32),
        precision = ML_Void,
      ),
      Return(
        TableLoad(
          out_table,
          Constant(13, precision = ML_Int32),
          precision = self.precision
        ),
        precision = self.precision,
      )
    )
    return scheme

  def numeric_emulate(self, input_value):
    table = [[i, i+1] for i in range(16)]
    index = int(input_value) % 16
    lo = table[index][0]
    hi = table[index][1]
    return lo + hi


def run_test(args):
  ml_ut_new_table = ML_UT_NewTable(args)
  ml_ut_new_table.gen_implementation(display_after_gen = True, display_after_opt = True)
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_NewTable.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)
