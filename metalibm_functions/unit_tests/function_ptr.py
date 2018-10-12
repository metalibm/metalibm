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
# created:              Oct   13th, 2018
# last-modified:        Oct   13th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

from metalibm_core.core.ml_function import ML_FunctionBasis

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

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_functions.unit_tests.utils import TestRunner


class ML_UT_FunctionPointer(ML_FunctionBasis, TestRunner):
    function_name = "ml_ut_function_ptr"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
                builtin from a default argument mapping overloaded with @p kw """
        default_args = {
                "output_file": "ut_function_ptr.c",
                "function_name": "ut_function_ptr",
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
        vx = self.implementation.add_input_variable("x", ML_Binary32)
        px = self.implementation.add_input_variable("px", ML_Binary32_p)
        fct_ptr_format = FunctionType([ML_Binary32, ML_Int32], ML_Binary32)
        fct_ptr = self.implementation.add_input_variable("fct", fct_ptr_format)

        result = vx * vx + fct_ptr(vx, Constant(3, precision=ML_Int32))
        #result.set_precision(ML_Binary32)
        #vx_assign = ReferenceAssign(px, result)
        px_assign = ReferenceAssign(Dereference(px, precision = ML_Binary32), result)
        scheme = Statement(px_assign)

        return scheme

    @staticmethod
    def __call__(args):
        ml_ut_function_ptr = ML_UT_FunctionPointer(args)
        ml_ut_function_ptr.gen_implementation()
        return True

run_test = ML_UT_FunctionPointer

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_FunctionPointer.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_FunctionPointer.__call__(args):
        exit(0)
    else:
        exit(1)

