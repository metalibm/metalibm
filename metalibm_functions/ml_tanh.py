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
    DefaultArgTemplate
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

from metalibm_core.core.indexing import SubUniformIntervalIndexing
from metalibm_core.core.approximation import (
    search_bound_threshold, generate_piecewise_poly_approx,
    load_piecewese_poly_params_from_axf,
    generate_piecewise_poly_approx_from_params
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

# disabling sollya's rounding warning
sollya.roundingwarnings = sollya.off
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on

from metalibm_core.utility.axf_utils import (
    AXF_JSON_Importer, AXF_SimplePolyApprox, AbsoluteApproxError,
    AXF_JSON_Exporter, AXF_ApproxError, AXF_Polynomial,
)


class ML_HyperbolicTangent(ScalarUnaryFunction):
    """ Implementation of hyperbolic tangent function """
    function_name = "ml_tanh"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)
        self.load_axf_approx = args.load_axf_approx
        self.dump_axf_approx = args.dump_axf_approx
        self.interval_num = args.interval_num
        self.near_zero_bound = args.near_zero_bound
        self.max_poly_degree = args.max_poly_degree

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_HyperbolicTangent,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_tanh = {
            "output_file": "my_tanh.c",
            "function_name": "my_tanh",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "load_axf_approx": None,
            "dump_axf_approx": False,
            "interval_num": 1024,
            "near_zero_bound": 0.125,
            "max_poly_degree": 7,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_tanh.update(kw)
        return DefaultArgTemplate(**default_args_tanh)

    def generate_approx_poly_near_zero(self, function, high_bound, error_bound):
        """ Generate polynomial approximation scheme for <function>
            in [0;<high_bound>] """
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

        axf_approx = AXF_SimplePolyApprox(
            AXF_Polynomial.from_poly(poly_object),
            function / sollya.x, degree_list,
            [1] + [self.precision] * (len(degree_list) - 1),
            approx_interval, approx_error=AXF_ApproxError.from_AE(AbsoluteApproxError(approx_error))
        )

        Log.report(Log.Info, "approximation poly: {}\n  with error {}".format(
                poly_object, approx_error))

        return poly_object, approx_error, axf_approx

    def generate_scalar_scheme(self, vx):
        """ Generating implementation script for hyperic tangent
            meta-function """
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
        near_zero_bound = self.near_zero_bound
        interval_num = self.interval_num
        Log.report(Log.Verbose, "high_bound={}, near_zero_bound={}, interval_num={}", float(high_bound), near_zero_bound, interval_num)

        interval_size = (high_bound - near_zero_bound) / (interval_num)
        new_interval_size = S2**int(sollya.log2(interval_size))
        interval_num *= 2
        high_bound = new_interval_size * interval_num + near_zero_bound
        Log.report(Log.Verbose, "high_bound={}, near_zero_bound={}, interval_num={}", float(high_bound), near_zero_bound, interval_num)

        ERROR_THRESHOLD = S2**-p
        Log.report(Log.Info, "ERROR_THRESHOLD={}", ERROR_THRESHOLD)


        # approximation parameters
        max_poly_degree = self.max_poly_degree
        approx_interval = Interval(near_zero_bound, high_bound)
        uniform_indexing = SubUniformIntervalIndexing(approx_interval, interval_num)  

        sollya.settings.points = 117

        # loading/generating approximation
        if self.load_axf_approx:
            Log.report(Log.Debug, "loading approximation from file")
            [near_zero_axf_approx, axf_approx] = AXF_JSON_Importer.from_file(self.load_axf_approx)

            # near 0 approximation
            near_zero_poly = near_zero_axf_approx.poly
            near_zero_error = near_zero_axf_approx.approx_error.value

            # far from 0 approximation
            offset_table, max_degree, coeff_table, approx_error = load_piecewese_poly_params_from_axf(axf_approx, uniform_indexing)
            approx_scheme = generate_piecewise_poly_approx_from_params(offset_table, max_degree, coeff_table, uniform_indexing, self.precision, abs_vx)

        else:
            # Near 0 approximation
            near_zero_poly, near_zero_error, axf_approx_near_zero = self.generate_approx_poly_near_zero(
                sollya.tanh(sollya.x),
                near_zero_bound,
                S2**-p
            )
            axf_approx_near_zero.tag = "tanh_near_zero"

            # far from 0 approximation
            def offset_function(fct):
                return lambda offset: fct(sollya.x + offset)

            approx_scheme, axf_approx = generate_piecewise_poly_approx(offset_function(sollya.tanh),
                                                                uniform_indexing,
                                                                ERROR_THRESHOLD * 2**-3,
                                                                self.precision,
                                                                abs_vx,
                                                                max_degree=max_poly_degree, # forcing max_degree
                                                                error_target_type=sollya.absolute,
                                                                axf_export=not self.dump_axf_approx is False)
            approx_error = axf_approx.approx_error.export_to_ml_error()


            if self.dump_axf_approx:
                axf_approx.tag = "tanh"
                AXF_JSON_Exporter.to_file(self.dump_axf_approx, [axf_approx_near_zero.serialize_to_dict(), axf_approx.serialize_to_dict()])

        # generate poly evaluation scheme for near 0 approximation
        near_zero_scheme = Multiplication(
            abs_vx,
            PolynomialSchemeEvaluator.generate_horner_scheme(
                near_zero_poly,
                abs_vx,
                self.precision
            )
        )

        Log.report(Log.Warning, "approx_error={}".format(approx_error))

        comp_near_zero_bound = abs_vx < near_zero_bound
        comp_near_zero_bound.set_attributes(tag="comp_near_zero_bound", debug=debug_multi)
        comp_high_bound = abs_vx < high_bound
        comp_high_bound.set_attributes(tag="comp_high_bound", debug=debug_multi)

        complete_scheme = Select(
            comp_near_zero_bound,
            near_zero_scheme,
            Select(
                comp_high_bound,
                approx_scheme,
                Constant(1.0, precision=self.precision)
            )
        )

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
        "0x1.ffb99ep-1",
        "0x1.f68b2cp-4"
    ]]



if __name__ == "__main__":
    # building argument template for main generation
    arg_template = ML_NewArgTemplate(
        default_arg=ML_HyperbolicTangent.get_default_args())
    arg_template.get_parser().add_argument(
         "--load-axf-approx", default=None,
        action="store", help="load tanh approx from an axf file rathen than computing it")
    arg_template.get_parser().add_argument(
         "--dump-axf-approx", default=False,
        action="store", help="export approximation used in AXF format")
    arg_template.get_parser().add_argument(
         "--interval-num", default=1024, type=int,
        action="store", help="number of approximation sub-divisions")
    arg_template.get_parser().add_argument(
         "--near-zero-bound", default=0.125, type=sollya.parse,
        action="store", help="bound to switch from near-zero to generic approximation")
    arg_template.get_parser().add_argument(
         "--max-poly-degree", default=7, type=int,
        action="store", help="maximal polynomial degree in table approximation")


    # argument extraction
    args = arg_template.arg_extraction()
    ml_tanh = ML_HyperbolicTangent(args)
    ml_tanh.gen_implementation()
