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
# created:                    Apr    5th, 2018
# last-modified:        Sep 20th, 2018
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for non-embedded binary execution
###############################################################################


import sys

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import (
        Constant, Comparison, ConditionBlock, Return, Statement
)
from metalibm_core.core.ml_formats import ML_Int32, ML_Bool

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_functions.unit_tests.utils import TestRunner

from metalibm_core.utility.ml_template import (
        DefaultMultiAryArgTemplate, MultiAryArgTemplate
)


class ML_UT_EmbeddedBin(ML_FunctionBasis, TestRunner):
    function_name = "ml_ut_embedded_bin"
    def __init__(self, args=DefaultMultiAryArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)
        self.arity = args.arity


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
                builtin from a default argument mapping overloaded with @p kw """
        default_args = {
                "precision": ML_Int32,
                "target": GenericProcessor.get_target_instance(),
                "bench_test_number": 10,
                "execute_trigger": True,
                "embedded_bin": False,
                "function_name": "ml_ut_embedded_bin",
                "bench_test_range": DefaultMultiAryArgTemplate.bench_test_range * 2,
                "auto_test_range": DefaultMultiAryArgTemplate.auto_test_range * 2,
                "arity": 2
        }
        default_args.update(kw)
        return DefaultMultiAryArgTemplate(**default_args)


    def generate_scheme(self):
        # declaring function input variable
        vx = self.implementation.add_input_variable("x", self.precision)
        vy = self.implementation.add_input_variable("y", self.precision)

        scheme = Return(vx + vy)

        return scheme

    @staticmethod
    def __call__(args):
        ml_ut_embedded_bin = ML_UT_EmbeddedBin(args)
        ml_ut_embedded_bin.gen_implementation()
        return True


run_test = ML_UT_EmbeddedBin

if __name__ == "__main__":
    # auto-test
    arg_template = MultiAryArgTemplate(default_arg=ML_UT_EmbeddedBin.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_EmbeddedBin.__call__(args):
        exit(0)
    else:
        exit(1)



