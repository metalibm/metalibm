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
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *


class ML_UT_PointerManipulation(ML_Function("ml_ut_pointer_manipulation")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_pointer_manipulation.c",
        "function_name": "ut_pointer_manipulation",
        "precision": ML_Binary32,
        "target": MPFRProcessor(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", ML_Binary32)
    px = self.implementation.add_input_variable("px", ML_Binary32_p)

    result = vx * vx
    # pointer dereferencing and value assignment
    px_assign = ReferenceAssign(Dereference(px, precision = ML_Binary32), result)

    # pointer to pointer cast
    py = Variable("py", precision=ML_Binary64_p, vartype=Variable.Local)
    py_assign = ReferenceAssign(py, TypeCast(px, precision=ML_Binary64_p))


    table_size = 16
    row_size   = 2

    new_table = ML_NewTable(dimensions = [table_size, row_size], storage_precision = self.precision)
    for i in range(table_size):
      new_table[i][0]= i 
      new_table[i][1]= i + 1
    # cast between table and pointer
    pz = Variable("pz", precision=ML_Pointer_Format(self.precision), vartype=Variable.Local)
    pz_assign = ReferenceAssign(pz, TypeCast(new_table, precision=ML_Binary64_p))

    scheme = Statement(px_assign, py_assign, pz_assign)

    return scheme
    

def run_test(args):
  ml_ut_pointer_manipulation = ML_UT_PointerManipulation(args)
  ml_ut_pointer_manipulation.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_PointerManipulation.get_default_args())
  args = arg_template.arg_extraction()

  if run_test(args):
    exit(0)
  else:
    exit(1)

