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
# created:          Dec  5th, 2018
# last-modified:    Dec  5th, 2018
###############################################################################
import sollya


from metalibm_core.core.ml_operations import (
    Comparison, ConditionBlock, Return,
)
from metalibm_core.core.ml_formats import ML_Binary32

from metalibm_core.core.precisions import (
    ML_Faithful, ML_CorrectlyRounded, ML_DegradedAccuracyAbsolute,
    ML_DegradedAccuracyRelative
)
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)

from metalibm_core.core.special_values import (
    FP_QNaN, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import (
    debug_multi
)

from metalibm_core.utility.gappa_utils import is_gappa_installed


class ML_Demo(ML_FunctionBasis):
    function_name = "ml_demo"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Demo,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_demo.c",
            "function_name": "ml_demo",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scheme(self):
        # declaring input variable
        vx = self.implementation.add_input_variable("x", self.precision)

        vx2 = vx * vx

        scheme = ConditionBlock(
            vx > 0,
            Return(vx - 0.33 * vx2 * vx + (2 / 15.0) * vx * vx2 * vx2),
            Return(FP_QNaN(self.precision))
        )

        return scheme

    def numeric_emulate(self, input_value):
        """ Numeric emaluation of exponential """
        return sollya.tanh(input_value)


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Demo.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    ml_exp = ML_Demo(args)

    ml_exp.gen_implementation()
