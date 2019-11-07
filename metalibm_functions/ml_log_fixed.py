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

import sollya

from sollya import (
        S2, Interval, ceil, floor, round, inf, sup, log, exp,
        log10, guessdegree
)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.debug_utils import debugd
from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value

class ML_Logarithm(object):
    def __init__(self,
                 precision = ML_Binary64,
                 abs_accuracy = S2**-24,
                 libm_compliant = True,
                 debug_flag = False,
                 fuse_fma = True,
                 fast_path_extract = True,
                 target = GenericProcessor(),
                 output_file = "log_fixed.c",
                 function_name = "log_fixed"):
        # declaring CodeFunction and retrieving input variable
        self.function_name = function_name
        self.precision = precision
        self.processor = target
        func_implementation = CodeFunction(self.function_name, output_format = self.precision)
        vx = func_implementation.add_input_variable("x", self.precision) 

        sollya_precision = self.precision.sollya_object

        vx_exp = RawSignExpExtraction(vx, tag = "vx_exp", precision = ML_Int32, debug = debugd)
        vx_exp_u = Conversion(vx_exp, precision = ML_UInt32)
        vx_exp_u.set_precision(ML_UInt32)
        tt = CountLeadingZeros(vx_exp_u)
        tt_u = Conversion(tt, precision = ML_UInt32)
        t = tt_u + vx_exp_u;
        scheme = Statement(Return(t))

        #print scheme.get_str(depth = None, display_precision = True)

        opt_eng = OptimizationEngine(self.processor)

        # fusing FMA
        if fuse_fma:
            print "MDL fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        #print scheme.get_str(depth = None, display_precision = True)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)

        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        print "silencing operation"
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        func_implementation.set_scheme(scheme)

        # check processor support
        opt_eng.check_processor_support(scheme)

        #print scheme.get_str(depth = None, display_precision = True)

        # factorizing fast path
        opt_eng.factorize_fast_path(scheme)
        #print scheme.get_str(depth = None, display_precision = True)

        cg = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = func_implementation.get_definition(cg, C_Code, static_cst = True)
        self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        #print self.result.get(cg)
        output_stream = open("%s.c" % func_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "log_fixed", default_output_file = "log_fixed.c")
    arg_template.sys_arg_extraction()

    ml_log = ML_Logarithm(precision          = arg_template.precision,
                          libm_compliant     = arg_template.libm_compliant,
                          debug_flag         = arg_template.debug_flag,
                          target             = arg_template.target,
                          fuse_fma           = arg_template.fuse_fma,
                          fast_path_extract  = arg_template.fast_path,
                          function_name      = arg_template.function_name,
                          output_file        = arg_template.output_file)
