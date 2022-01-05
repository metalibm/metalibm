# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2022 Nicolas Brunie
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
# created:          Jan  5th, 2022
# last-modified:    Jan  5th, 2022
# Author(s): Nicolas Brunie
###############################################################################

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.ml_operations import Addition, Constant, Return, TableStore, TableLoad, Statement
from metalibm_core.core.ml_formats import ML_Int32, ML_Binary32

from metalibm_core.utility.ml_template import DefaultFunctionArgTemplate, MetaFunctionArgTemplate

from metalibm_functions.unit_tests.utils import TestRunner


class ML_UT_LinearizeTable(ML_FunctionBasis, TestRunner):
  function_name = "ml_ut_linearize_table"
  def __init__(self, args=DefaultFunctionArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_table_linearized.c",
        "function_name": "ut_table_linearized",
        "precision": ML_Binary32,
        "target": GenericProcessor.get_target_instance(),
        "passes": ["optimization:table_linearization"],
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultFunctionArgTemplate(**default_args)

  def generate_scheme(self):
    # declaring function input variable
    vi = self.implementation.add_input_variable("vi", ML_Int32)

    table2d = ML_NewTable(dimensions = [17, 3], storage_precision=self.precision)
    for i in range(17):
      table2d[i][0] = i
      table2d[i][1] = i + 7
      table2d[i][2] = i + 11

    firstElt = TableLoad(table2d, vi, Constant(0, precision=ML_Int32), precision=self.precision)    
    secondElt = TableLoad(table2d, vi, Constant(1, precision=ML_Int32), precision=self.precision)    
    thirdElt = TableLoad(table2d, vi, Constant(2, precision=ML_Int32), precision=self.precision)    
    dynElt = TableLoad(table2d, vi, vi, precision=self.precision)

    statement = Statement(
      Return(Addition(firstElt, 
                      Addition(secondElt, 
                              Addition(thirdElt, dynElt, precision=self.precision), precision=self.precision)))
    )

    return statement

  @staticmethod
  def __call__(args):
    ml_ut_linearize_table = ML_UT_LinearizeTable(args)
    ml_ut_linearize_table.gen_implementation()
    return True

run_test = ML_UT_LinearizeTable

if __name__ == "__main__":
  # auto-test
  arg_template = MetaFunctionArgTemplate(default_arg=ML_UT_LinearizeTable.get_default_args())
  args = arg_template.arg_extraction()

  if run_test.__call__(args):
    exit(0)
  else:
    exit(1)


