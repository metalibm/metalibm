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
# created:              Nov  9th, 2018
# last-modified:        Nov  9th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for ML static vectorization
###############################################################################


import sys

from sollya import Interval

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import (
    ML_DoubleDouble
)

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.targets.common.vector_backend import VectorBackend


from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)
from metalibm_core.utility.debug_utils import debug_multi
from metalibm_functions.unit_tests.utils import TestRunner


class ML_UT_MultiPrecisionVectorization(ML_FunctionBasis, TestRunner):
    function_name = "ml_ut_mp_vectorization"
    arity = 2
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw
        """
        default_args = {
                "output_file": "ut_mp_vectorization.c",
                "function_name": "ut_mp_vectorization",
                "precision": ML_DoubleDouble,
                "target": GenericProcessor.get_target_instance(),
                "fast_path_extract": True,
                "fuse_fma": True,
                "passes": ["start:basic_legalization", "start:expand_multi_precision"],
                "libm_compliant": True
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):

        # declaring function input variable
        vx = self.implementation.add_input_variable("x", self.precision)
        vy = self.implementation.add_input_variable("y", self.precision)

        add_vx = Addition(
            vx, vy, precision=self.precision,
            tag="add_vx", debug=debug_multi)
        mult = Multiplication(
            add_vx, vx, precision=self.precision,
            tag="result", debug=debug_multi)

        result = FMA(vx, vy, mult, precision=self.precision, tag="result")

        scheme = Statement(
           Return(result),
           tag="scheme"
        )

        return scheme

    def numeric_emulate(self, vx, vy):
        return vx * vy + ((vx + vy) * vx)

    @staticmethod
    def __call__(args):
        """ TestRunner call function """
        ml_ut_mp_vectorization = ML_UT_MultiPrecisionVectorization(args)
        ml_ut_mp_vectorization.gen_implementation()
        return True

    standard_test_cases = [
        (1.0, 2.0),
        (1.5, 2.0),
        (2.0, 2.0),
        (3.0, 2.0),
        (
            sollya.parse("0x1.4c0ff9a97083c804f4db2002c8p-1"),
            sollya.parse("0x1.7fd01d4fe3196307a4e0008b48p-3")
        ),
        (
            sollya.parse("0x1.4c0ff9a97083c804f4db2002c8p-1"),
            sollya.parse("0x1.7fd01d4fe3196307a4e0008b48p-3")
        ),
        (
            sollya.parse("0x1.4c0ff9a97083c804f4db2002c8p-1"),
            sollya.parse("0x1.7fd01d4fe3196307a4e0008b48p-3")
        ),
        (
            sollya.parse("0x1.4c0ff9a97083c804f4db2002c8p-1"),
            sollya.parse("0x1.7fd01d4fe3196307a4e0008b48p-3")
        ),
    ]

run_test = ML_UT_MultiPrecisionVectorization


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MultiPrecisionVectorization.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_MultiPrecisionVectorization.__call__(args):
        exit(0)
    else:
        exit(1)


