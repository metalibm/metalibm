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
#
# created:            Mar   11th, 2016
# last-modified:      Mar    3rd, 2019
#
# description: meta-implementation of error-function erf
#              erf(x) = 2 / pi * integral(0, x, e^(-t^2), dt)
###############################################################################

import sollya

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, log1p,
    dirtyinfnorm,
    guessdegree
)

from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.ml_table import ML_NewTable, generic_mantissa_msb_index_fct
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.special_values import (
        FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)
from metalibm_core.core.approximation import (
    search_bound_threshold, generic_poly_split, SubFPIndexing
)
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.debug_utils import debug_multi


# static constant for numerical value 2
S2 = sollya.SollyaObject(2)


class ML_Erf(ScalarUnaryFunction):
    """ Meta implementation of the error-function """
    function_name = "ml_erf"
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Erf,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_erf = {
                "output_file": "my_erf.c",
                "function_name": "my_erf",
                "precision": ML_Binary32,
                "accuracy": ML_Faithful,
                "target": GenericProcessor.get_target_instance(),
                "passes": [
                    ("start:instantiate_abstract_prec"),
                    ("start:instantiate_prec"),
                    ("start:basic_legalization"),
                    ("start:expand_multi_precision")],
        }
        default_args_erf.update(kw)
        return DefaultArgTemplate(**default_args_erf)

    def generate_scalar_scheme(self, vx):
        abs_vx = Abs(vx, precision=self.precision)

        FCT_LIMIT = 1.0

        one_limit = search_bound_threshold(sollya.erf, FCT_LIMIT, 1.0, 10.0, self.precision)
        one_limit_exp = int(sollya.floor(sollya.log2(one_limit)))
        Log.report(Log.Debug, "erf(x) = 1.0 limit is {}, with exp={}", one_limit, one_limit_exp)

        upper_approx_bound = 10

        # empiral numbers
        eps_exp = {ML_Binary32: -3, ML_Binary64: -5}[self.precision]
        eps = S2**eps_exp

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(0, eps)
        # fonction to approximate is erf(x) / x
        # it is an even function erf(x) / x = erf(-x) / (-x)
        approx_fct = sollya.erf(sollya.x) - (sollya.x)
        poly_degree = int(sup(guessdegree(approx_fct, approx_interval, S2**-(self.precision.get_field_size()+5)))) + 1

        poly_degree_list = list(range(1, poly_degree, 2))
        Log.report(Log.Debug, "poly_degree is {} and list {}", poly_degree, poly_degree_list)
        global_poly_object = Polynomial.build_from_approximation(approx_fct, poly_degree_list, [self.precision]*len(poly_degree_list), approx_interval, sollya.relative)
        Log.report(Log.Debug, "inform is {}", dirtyinfnorm(approx_fct - global_poly_object.get_sollya_object(), approx_interval))
        poly_object = global_poly_object.sub_poly(start_index=1, offset=1)

        ext_precision = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble,
        }[self.precision]

        pre_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, abs_vx, unified_precision=self.precision)

        result = FMA(pre_poly, abs_vx, abs_vx)
        result.set_attributes(tag="result", debug=debug_multi)


        eps_target = S2**-(self.precision.get_field_size() +5)
        def offset_div_function(fct):
            return lambda offset: fct(sollya.x + offset)

        # empiral numbers
        field_size = {
            ML_Binary32: 6,
            ML_Binary64: 8
        }[self.precision]

        near_indexing = SubFPIndexing(eps_exp, 0, 6, self.precision)
        near_approx = generic_poly_split(offset_div_function(sollya.erf), near_indexing, eps_target, self.precision, abs_vx)
        near_approx.set_attributes(tag="near_approx", debug=debug_multi)

        def offset_function(fct):
            return lambda offset: fct(sollya.x + offset)
        medium_indexing = SubFPIndexing(1, one_limit_exp, 7, self.precision)

        medium_approx = generic_poly_split(offset_function(sollya.erf), medium_indexing, eps_target, self.precision, abs_vx)
        medium_approx.set_attributes(tag="medium_approx", debug=debug_multi)

        # approximation for positive values
        scheme = ConditionBlock(
            abs_vx < eps,
            Return(result),
            ConditionBlock(
                abs_vx < near_indexing.get_max_bound(),
                Return(near_approx),
                ConditionBlock(
                    abs_vx < medium_indexing.get_max_bound(),
                    Return(medium_approx),
                    Return(Constant(1.0, precision=self.precision))
                )
            )
        )
        return scheme

    def numeric_emulate(self, input_value):
        return sollya.erf(input_value)

    standard_test_cases = [
        (sollya.parse("0x1.4c0d4e9f58p-8"),),
        (1.0, None),
        (4.0, None),
        (0.5, None),
        (1.5, None),
        (1024.0, None),
        (sollya.parse("0x1.13b2c6p-2"), None),
        (sollya.parse("0x1.2cb10ap-5"), None),
        (0.0, None),
        (sollya.parse("0x1.07e08ep+1"), None),
    ]



if __name__ == "__main__":
        # auto-test
        arg_template = ML_NewArgTemplate(default_arg=ML_Erf.get_default_args())
        args = arg_template.arg_extraction()

        ml_erf = ML_Erf(args)
        ml_erf.gen_implementation()
