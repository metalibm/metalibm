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

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.ml_operations import (
    Constant, Return, Addition, Multiplication, TypeCast, Statement,
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Int32, ML_Custom_FixedPoint_Format
)
from metalibm_core.core.precisions import dar

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_functions.unit_tests.utils import TestRunner

import metalibm_core.opt.runtime_error_eval as runtime_error_eval

FIXED_FORMAT = ML_Custom_FixedPoint_Format(3, 29, False)

class ML_UT_ErrorEval(ML_FunctionBasis, TestRunner):
    function_name = "ML_UT_ErrorEval"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_eval_error.c",
            "function_name": "ut_eval_error",
            "precision": FIXED_FORMAT,
            "target": GenericProcessor.get_target_instance(),
            "fast_path_extract": True,
            "fuse_fma": True,
            "debug": True,
            "libm_compliant": True,
            "test_range": Interval(S2**-8, S2**8),
            "accuracy": dar(S2**-6),
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("x", FIXED_FORMAT)
        # declaring specific interval for input variable <x>
        vx.set_interval(Interval(-1, 1))

        acc_format = ML_Custom_FixedPoint_Format(6, 58, False)

        c = Constant(2, precision=acc_format, tag="C2")

        ivx = vx
        add_ivx = Addition(
                    c,
                    Multiplication(ivx, ivx, precision=acc_format, tag="mul"),
                    precision=acc_format,
                    tag="add"
                  )
        result = add_ivx

        input_mapping = {ivx: ivx.get_precision().round_sollya_object(0.125)}
        error_eval_map = runtime_error_eval.generate_error_eval_graph(result, input_mapping)

        # dummy scheme to make functionnal code generation
        scheme = Statement()
        for node in error_eval_map:
            scheme.add(error_eval_map[node])
        scheme.add(Return(result))
        return scheme

    def numeric_emulate(self, x):
        """ numeric emulation """
        # extracting mantissa from x
        # abs_x = abs(x)
        # mantissa = abs_x / S2**sollya.floor(sollya.log2(abs_x))
        # index = sollya.floor((mantissa - 1.0) * 2**8)
        # result = sollya.round(1/sollya.sqrt(1.0 + index * S2**-8), 9, sollya.RN)
        result = sollya.round(1.0/x, 9, sollya.RN)
        return result

    def __call__(args):
        meta_instance = ML_UT_ErrorEval(args)
        meta_instance.gen_implementation()
        return True

run_test = ML_UT_ErrorEval


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_ErrorEval.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_ErrorEval.__call__(args):
        exit(0)
    else:
        exit(1)


