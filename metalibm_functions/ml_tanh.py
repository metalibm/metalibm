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
###############################################################################
import sys

import sollya

from sollya import (
     Interval, tanh
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.core.ml_formats import ML_Binary32, ML_Int32
from metalibm_core.core.precisions import ML_Faithful

from metalibm_core.core.polynomials import (
    Polynomial, PolynomialSchemeEvaluator, SollyaError
)
from metalibm_core.core.special_values import FP_PlusInfty
from metalibm_core.core.ml_operations import (
    Return, Subtraction, TableLoad, Constant, NearestInteger, Multiplication,
    Division, Addition, Conversion, Max, Min,
    Abs, Negation, Select
)
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.core.approximation import piecewise_approximation

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate, ArgDefault
from metalibm_core.utility.log_report  import Log

# disabling sollya's rounding warning
sollya.roundingwarnings = sollya.off
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on


class ML_HyperbolicTangent(ML_FunctionBasis):
    """ Implementation of hyperbolic tangent function """
    function_name = "ml_tanh"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self,
          args
        )

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_HyperbolicTangent,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_tanh = {
            "output_file": "my_tanh.c",
            "function_name": "my_tanh",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor()
        }
        default_args_tanh.update(kw)
        return DefaultArgTemplate(**default_args_tanh)

    def generate_approx_poly_near_zero(self, function, high_bound, error_bound, variable):
        """ Generate polynomial approximation scheme """
        error_function = lambda p, f, ai, mod, t: sollya.dirtyinfnorm(p - f, ai)
        # Some issues encountered when 0 is one of the interval bound
        # so we use a symetric interval around it
        approx_interval = Interval(2**-100, high_bound)
        local_function = function / sollya.x

        degree = sollya.sup(sollya.guessdegree(local_function, approx_interval, error_bound))
        degree_list = range(0, int(degree)+4, 2)


        poly_object, approx_error = Polynomial.build_from_approximation_with_error(
            function / sollya.x,
            degree_list,
            [1] + [self.precision] * (len(degree_list) - 1),
            approx_interval, sollya.absolute,
            error_function = error_function
        )
        Log.report(Log.Info, "approximation poly: {}\n  with error {}".format(
                poly_object, approx_error
            )
        )

        poly_scheme = Multiplication(
            variable,
            PolynomialSchemeEvaluator.generate_horner_scheme(
                poly_object,
                variable,
                self.precision
            )
        )
        return poly_scheme, approx_error

    def generate_scheme(self):
        """ Generating implementation script for hyperic tangent
            meta-function """
        # registering the single input variable to the function
        vx = self.implementation.add_input_variable("x", self.precision)

        #Log.set_dump_stdout(True)
        # tanh(x) = sinh(x) / cosh(x)
        #         = (e^x - e^-x) / (e^x + e^-x)
        #         = (e^(2x) - 1) / (e^(2x) + 1)
        #   when x -> +inf, tanh(x) -> 1
        #   when x -> -inf, tanh(x) -> -1
        #   ~0 e^x    ~ 1 + x - x^2 / 2 + x^3 / 6 + ...
        #      e^(-x) ~ 1 - x - x^2 / 2- x^3/6 + ...
        #   when x -> 0, tanh(x) ~ (2 (x + x^3/6 + ...)) / (2 - x^2 + ...) ~ x
        # We can divide the input interval into 3 parts
        # positive, around 0, and finally negative

        # Possible argument reduction
        # x = m.2^E = k * log(2) + r
        # (k != 0) => tanh(x) = (2k * e^(2r) - 1) / (2k * e^(2r) + 1)
        #                     = (1 - 1 * e^(-2r) / 2k) / (1 + e^(-2r) / 2k)
        #
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        #         = (e^(2x) + 1 - 1- 1) / (e^(2x) + 1)
        #         = 1 - 2 / (e^(2x) + 1)

        # tanh is odd so we reduce the computation to the absolute value of
        # vx
        abs_vx = Abs(vx,precision=self.precision)

        # if p is the expected output precision
        # x > (p+2) * log(2) / 2 => tanh(x) = 1 - eps
        #   where eps < 1/2 * 2^-p
        p = self.precision.get_mantissa_size()
        high_bound = (p+2) * sollya.log(2) / 2
        near_zero_bound = 0.125
        interval_num = 1024

        interval_size = (high_bound - near_zero_bound) / (1024)
        new_interval_size = S2**int(sollya.log2(interval_size))
        interval_num *= 2
        high_bound = new_interval_size * interval_num + near_zero_bound

        # Near 0 approximation
        near_zero_scheme, near_zero_error = self.generate_approx_poly_near_zero(
            sollya.tanh(sollya.x),
            near_zero_bound,
            S2**-p,
            abs_vx
        )

        # approximation parameters
        poly_degree = 5
        approx_interval = Interval(near_zero_bound, high_bound)

        sollya.settings.points = 117

        approx_scheme, approx_error = piecewise_approximation(
            sollya.tanh,
            abs_vx,
            self.precision,
            bound_low=near_zero_bound,
            bound_high=high_bound,
            num_intervals=interval_num,
            max_degree=5,
            error_threshold=S2**-p
        )
        Log.report(Log.Warning, "approx_error={}".format(approx_error))

        complete_scheme = Select(
            abs_vx < near_zero_bound,
            near_zero_scheme,
            Select(
                abs_vx < high_bound,
                approx_scheme,
                Constant(1.0,precision=self.precision)
            )
        )

        Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
        scheme = Return(
            Select(
                vx<0,Negation(complete_scheme),complete_scheme
            ), precision=self.precision)
        return scheme

    def numeric_emulate(self, input_value):
        return tanh(input_value)

    standard_test_cases =[
        [sollya.parse(x)] for x in  [
        "-0x1.572306p+0",
        "0x1.af0bf2p+1",
        "-0x1.af0bf2p+1",
        "-0x1.51b618p-13",
        "0x1.ffb99ep-1"
    ]]



if __name__ == "__main__":
    # building argument template for main generation
    arg_template = ML_NewArgTemplate(
        default_arg=ML_HyperbolicTangent.get_default_args()
    )

    # argument extraction
    args = arg_template.arg_extraction()
    ml_tanh = ML_HyperbolicTangent(args)
    ml_tanh.gen_implementation()
