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
from sollya import Interval
import bigfloat

from metalibm_core.core.ml_operations import (
    Return, Statement, Division, Trunc, Conversion,
    FusedMultiplyAdd,
)
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_formats import ML_Binary32
from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.simple_scalar_function import ScalarBinaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate


class MetaFMOD(ScalarBinaryFunction):
    function_name = "ml_fmod"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetaFMOD,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_fmod.c",
            "function_name": "ml_fmod",
            "input_intervals": (Interval(-100, 100), Interval(-100, 100)),
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance(),
            "arity": MetaFMOD.arity,
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scalar_scheme(self, vx, vy):
        div = Division(vx, vy, precision=self.precision)
        div_if = Trunc(div, precision=self.precision)
        rem = FusedMultiplyAdd(-div_if, vy, vx)
        scheme = Statement(
            Return(rem),
        )
        return scheme

    def numeric_emulate(self, vx, vy):
        """ Numeric emaluation of exponential """
        return float(bigfloat.fmod(float(vx), float(vy)))

    standard_test_cases = [
        (10,0),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=MetaFMOD.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    ml_fmod = MetaFMOD(args)

    ml_fmod.gen_implementation()
