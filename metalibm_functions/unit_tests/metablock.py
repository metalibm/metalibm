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
# Instances (see valid/unit_test.py
# 1.  --pre-gen-passes m128_promotion --target x86_avx2
#

import sys

import sollya
from sollya import Interval

S2 = sollya.SollyaObject(2)

from metalibm_functions.unit_tests.utils import TestRunner

from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.ml_operations import (
    Addition, Return
)
from metalibm_core.core.ml_formats import (
    ML_Binary64, ML_DoubleDouble)
from metalibm_core.core.precisions import ML_CorrectlyRounded


from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)

from metalibm_core.code_generation.generic_processor import GenericProcessor


class ML_UT_MetaBlock(TestRunner, ML_FunctionBasis):
    name = "ml_ut_metablock"
    arity = 2
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_copysign.c",
            "function_name": "ut_copysign",
            "precision": ML_DoubleDouble,
            "input_precisions": [ML_Binary64, ML_Binary64],
            "target": GenericProcessor.get_target_instance(),
            "auto_test_range": [Interval(S2**-8, S2**8), Interval(S2**-8, S2**8)],
            "accuracy": ML_CorrectlyRounded,
            "auto_test": 1000,
            "execute_trigger": True,
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        # declaring function input variable
        vx = self.implementation.add_input_variable("x", self.get_input_precision(0))
        vy = self.implementation.add_input_variable("y", self.get_input_precision(1))

        result = Addition(vx, vy, precision=ML_DoubleDouble)

        scheme = Return(result, precision=self.precision,)

        return scheme

    def numeric_emulate(self, x, y):
        """ numeric emulation """
        result = x + y
        return result

    # execution function for the TestRunner class
    @staticmethod
    def __call__(args):
        meta_instance = ML_UT_MetaBlock(args)
        meta_instance.gen_implementation()
        return True

run_test = ML_UT_MetaBlock


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MetaBlock.get_default_args())
    args = arg_template.arg_extraction()

    if run_test.__call__(args):
        exit(0)
    else:
        exit(1)


