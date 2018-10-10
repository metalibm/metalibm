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
# created:                    Oct    8th, 2018
# last-modified:        Oct    8th, 2018
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for multi-precision format expansion
###############################################################################


import sys

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import (
        Return, Addition, Statement, Conversion
)
from metalibm_core.core.ml_formats import (
        ML_Binary32, ML_Binary64,
        ML_SingleSingle, ML_DoubleDouble
)

from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_functions.unit_tests.utils import TestRunner


from metalibm_core.utility.ml_template import (
        DefaultArgTemplate, ML_NewArgTemplate
)


class ML_UT_MultiPrecision(ML_FunctionBasis, TestRunner):
    function_name = "ml_ut_multi_precision"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
                builtin from a default argument mapping overloaded with @p kw """
        default_args = {
                "output_file": "ut_multi_precision.c",
                "function_name": "ut_multi_precision",
                "precision": ML_Binary32,
                "target": GenericProcessor(),
                "language": C_Code,
                "arity": 2,
                "input_precisions": [ML_Binary32, ML_Binary32],
                "fast_path_extract": True,
                "fuse_fma": False,
                "libm_compliant": True
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)


    def generate_scheme(self):
        # declaring function input variable
        v_x = [self.implementation.add_input_variable("x%d" % index, self.get_input_precision(index)) for index in range(self.arity)]

        double_format = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble
        }[self.precision]

        exact_add = Addition(v_x[0], v_x[1], precision=double_format, tag="exact_add")
        result = Conversion(exact_add, precision=self.precision)

        scheme = Statement(
            Return(result)
        )

        return scheme

    def numeric_emulate(self, *args):
        acc = 0.0
        for i in range(self.arity):
            acc += args[i]
        return acc


    @staticmethod
    def __call__(args):
        ml_ut_llvm_code = ML_UT_MultiPrecision(args)
        ml_ut_llvm_code.gen_implementation()
        return True


run_test = ML_UT_MultiPrecision

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MultiPrecision.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_MultiPrecision.__call__(args):
        exit(0)
    else:
        exit(1)



