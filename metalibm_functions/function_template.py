# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          Mar  9th, 2020
# last-modified:    Mar  9th, 2020
###############################################################################
import sollya

from metalibm_core.core.ml_operations import (
    Return, Statement,
)
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_formats import ML_Binary32
from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate


class FunctionTemplate(ScalarUnaryFunction):
    function_name = "func_template"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for FunctionTemplate,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "func_template.c",
            "function_name": "func_template",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scalar_scheme(self, vx):
        scheme = Statement(
            Return(vx),
        )
        return scheme

    def numeric_emulate(self, input_value):
        """ Numeric emaluation of exponential """
        return input_value

    standard_test_cases = [
        (sollya.parse("0x1.1p0"),),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=FunctionTemplate.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    func_template = FunctionTemplate(args)

    func_template.gen_implementation()
