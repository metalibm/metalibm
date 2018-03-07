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
###############################################################################
import sys

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.ml_function import ML_Function

class ML_SinCospi(ML_Function):
    def __init__(self, 
                 base_name = "sincospi",
                 name=None,
                 output_file = None,
                 io_precision = ML_Binary32, 
                 libm_compliant = True, 
                 processor = GenericProcessor(), 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 debug_flag = False
             ):
        # declaring CodeFunction and retrieving input variable
        ML_Function.__init__(self,
                             base_name = base_name,
                             name=name,
                             output_file = name,
                             io_precision = io_precision,
                             libm_compliant = libm_compliant,
                             processor=processor,
                             fuse_fma = fuse_fma,
                             fast_path_extract = fast_path_extract,
                             debug_flag = debug_flag)

        # main scheme
        Log.report(Log.Info, "Construction of the initial evaluation scheme")

        vx = self.implementation.add_input_variable("x", self.io_precision) 
        self.evalScheme = Statement(Return(vx))

        self.opt_engine.optimization_process(self.evalScheme, self.io_precision)
        self.generate_C()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "sincospi", default_output_file = "sincospi.c" )
    arg_template.sys_arg_extraction()
    ml_sincospi     = ML_SinCospi(base_name                 = arg_template.function_name,
                                  io_precision              = arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  processor                 = arg_template.target,  # TODO rename target into processor 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  output_file               = arg_template.output_file
    )
