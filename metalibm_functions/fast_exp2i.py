# -*- coding: utf-8 -*-
""" Implementation of fast exponentation of integers to floating-point values
"""
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
# last-modified:    Mar  21st, 2020
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
from metalibm_core.core.ml_operations import (Max, Min, ExponentInsertion, Return)
from metalibm_core.core.ml_formats import (ML_Int32, ML_Binary32)
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate)
from metalibm_core.utility.debug_utils import debug_multi


class FastExp2i(ScalarUnaryFunction):
    """ Meta-implementation of fast-exponentation of integers to
        floating-point values """
    function_name = "fast_exp2i"

    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super(ScalarUnaryFunction, self).__init__(args)

    def generate_scalar_scheme(self, vx):
        output_precision = self.precision
        input_precision = vx.get_precision()

        bias = -output_precision.get_bias()
        bound_exp = Max(
            Min(vx, output_precision.get_emax(), precision=input_precision),
            output_precision.get_emin_normal(), precision=input_precision) + bias
        scheme = Return(
            ExponentInsertion(bound_exp,
                              specifier=ExponentInsertion.NoOffset,
                              precision=self.precision), tag="result", debug=debug_multi)
        return scheme

    def numeric_emulate(self, input_value):
        return 2**input_value


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetalibmSqrt,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_fast_exp2i = {
            "output_file": "fast_expi.c",
            "function_name": "fast_expi",
            "input_precisions": [ML_Int32],
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_fast_exp2i.update(kw)
        return DefaultArgTemplate(**default_args_fast_exp2i)


if __name__ == "__main__":
    # auto-test
    ARG_TEMPLATE = ML_NewArgTemplate(default_arg=FastExp2i.get_default_args())

    ARGS = ARG_TEMPLATE.arg_extraction()

    ML_FAST_EXP_I = FastExp2i(ARGS)
    ML_FAST_EXP_I.gen_implementation()
