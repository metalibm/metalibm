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
import sys

import sollya

from sollya import SollyaObject, Interval, log2, acos, sup
S2 = SollyaObject(2)

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_NewArgTemplate

from metalibm_core.utility.debug_utils import *

class ML_Acos(ML_FunctionBasis):
    function_name = "ml_acos"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Acos,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_acos = {
            "output_file": "ml_acos.c",
            "function_name": "ml_acos",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_acos.update(kw)
        return DefaultArgTemplate(**default_args_acos)

    def generate_scheme(self):
        """ generate scheme """
        vx = self.implementation.add_input_variable("x", self.get_input_precision())

        approx_interval = Interval(-0.5, 0.5)
        target_epsilon = S2**-(self.precision.get_field_size())
        poly_degree = sup(sollya.guessdegree(acos(sollya.x), approx_interval, target_epsilon))

        Log.report(Log.Info, "poly_degree={}", poly_degree)

        poly_object, poly_error = Polynomial.build_from_approximation_with_error(
            acos(sollya.x),
            poly_degree,
            [self.precision] * (poly_degree+1),
            approx_interval,
            sollya.absolute)

        Log.report(Log.Info, "poly_error={}", poly_error)

        poly_scheme = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, vx, unified_precision=self.precision)

        main_block = ConditionBlock(
            LogicalOr(vx > 1.0, vx < -1.0, likely=False),
            Return(FP_QNaN(self.precision)),
            Return(poly_scheme)
        )

        #scheme = Statement(main_block)
        scheme = Statement(Return(FP_QNaN(self.precision)))
        return scheme

    def numeric_emulate(self, input_value):
        """ Numeric emaluation of arc cosine """
        return sollya.acos(input_value)


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate( default_arg=ML_Acos.get_default_args())
    args = arg_template.arg_extraction()

    ml_acos          = ML_Acos(args)
    ml_acos.gen_implementation()
