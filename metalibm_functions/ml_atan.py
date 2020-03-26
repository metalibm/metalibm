# -*- coding: utf-8 -*-
""" meta-implementation of arc-tangent (atan) function """

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
# created:          Mar  7th, 2018
# last-modified:    Mar 18th, 2020
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

import sollya

from sollya import Interval

from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.simple_scalar_function import (
    ScalarBinaryFunction, ScalarUnaryFunction)

from metalibm_core.core.ml_operations import (
    Abs,
    Select, Statement, Return,
    LogicalOr, LogicalAnd, LogicalNot,
)
from metalibm_core.core.ml_formats import ML_Binary32
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.approximation import piecewise_approximation

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import debug_multi

S2 = sollya.SollyaObject(2)

# Disabling Sollya's rounding warnings
sollya.roundingwarnings = sollya.off
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on


class MetaAtan(ScalarUnaryFunction):
    """ Meta implementation of arctangent function """
    function_name = "ml_atan"
    def __init__(self, args):
        super().__init__(args)
        self.method = args.method

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetaAtan,
                builtin from a default argument mapping overloaded with @p kw
        """
        default_args_exp = {
            "output_file": "my_atan.c",
            "function_name": "my_atan",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "method": "piecewise",
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scalar_scheme(self, vx):
        """ Evaluation scheme generation """
        # if abs_vx < 1.0 then atan(abx_vx) is directly approximated
        # if abs_vx >= 1.0 then atan(abs_vx) = pi/2 - atan(1 / abs_vx)
        return self.generic_atan2_generate(vx)

    def generic_atan2_generate(self, _vx, vy=None):
        # computing absolute value of vx
        vx = _vx if vy is None else _vx / vy

        if vy is None:
            abs_vx = Select(vx < 0, -vx, vx, tag="abs_vx", debug=debug_multi)
            bound_cond = abs_vx > 1
            inv_abs_vx = 1 / abs_vx

            cond = LogicalOr(
                LogicalAnd(vx < 0, LogicalNot(bound_cond)),
                vx > 1,
                tag="cond", debug=debug_multi
            )

            # reduced argument
            red_vx = Select(bound_cond, inv_abs_vx, abs_vx, tag="red_vx", debug=debug_multi)
        else:
            bound_cond = Abs(_vx) > Abs(vy)
            sign_cond = (_vx * vy) < 0
            # atan input is negative
            cond = LogicalOr(
                LogicalAnd(sign_cond, LogicalNot(bound_cond)),
                vx > 1,
                tag="cond", debug=debug_multi
            )

            numerator = Select(cond, _vx, vy, tag="numerator", debug=debug_multi)
            denominator = Select(cond, vy, _vx, tag="denominator", debug=debug_multi)
            # reduced argument
            red_vx = numerator / denominator
            red_vx.set_attributes(tag="red_vx", debug=debug_multi)


        approx_fct = sollya.atan(sollya.x)

        if self.method == "piecewise":
            sign_vx = Select(cond, -1, 1, precision=self.precision, tag="sign_vx", debug=debug_multi)

            cst_sign = Select(vx < 0, -1, 1, precision=self.precision)
            cst = cst_sign * Select(bound_cond, sollya.pi / 2, 0, precision=self.precision)


            bound_low = 0.0
            bound_high = 1.0
            num_intervals = 8
            error_threshold = S2**-(self.precision.get_mantissa_size() + 8)



            approx, eval_error = piecewise_approximation(approx_fct,
                                    red_vx,
                                    self.precision,
                                    bound_low=bound_low,
                                    bound_high=bound_high,
                                    max_degree=None,
                                    num_intervals=num_intervals,
                                    error_threshold=error_threshold,
                                    odd=True)

            result = cst + sign_vx * approx

        elif self.method == "single":
            approx_interval = Interval(0, 1.0)
            # determining the degree of the polynomial approximation
            poly_degree_range = sollya.guessdegree(approx_fct / sollya.x,
                                                   approx_interval,
                                                   S2**-(self.precision.get_field_size() + 2))
            poly_degree = int(sollya.sup(poly_degree_range)) + 4
            Log.report(Log.Info, "poly_degree={}".format(poly_degree))

            # arctan is an odd function, so only odd coefficient must be non-zero
            poly_degree_list = list(range(1, poly_degree+1, 2))
            poly_object, poly_error = Polynomial.build_from_approximation_with_error(
                approx_fct, poly_degree_list,
                [1] + [self.precision.get_sollya_object()] * (len(poly_degree_list)-1),
                approx_interval)

            odd_predicate = lambda index, _: ((index-1) % 4 != 0)
            even_predicate = lambda index, _: (index != 1 and (index-1) % 4 == 0)

            poly_odd_object = poly_object.sub_poly_cond(odd_predicate, offset=1)
            poly_even_object = poly_object.sub_poly_cond(even_predicate, offset=1)

            sollya.settings.display = sollya.hexadecimal
            Log.report(Log.Info, "poly_error: {}".format(poly_error))
            Log.report(Log.Info, "poly_odd: {}".format(poly_odd_object))
            Log.report(Log.Info, "poly_even: {}".format(poly_even_object))

            poly_odd = PolynomialSchemeEvaluator.generate_horner_scheme(poly_odd_object, abs_vx)
            poly_odd.set_attributes(tag="poly_odd", debug=debug_multi)
            poly_even = PolynomialSchemeEvaluator.generate_horner_scheme(poly_even_object, abs_vx)
            poly_even.set_attributes(tag="poly_even", debug=debug_multi)
            exact_sum = poly_odd + poly_even

            # poly_even should be (1 + poly_even)
            result = vx + vx * exact_sum
            result.set_attributes(tag="result", precision=self.precision)

        else:
            raise NotImplementedError

        std_scheme = Statement(
            Return(result)
        )
        scheme = std_scheme

        return scheme

    def numeric_emulate(self, input_value):
        return sollya.atan(input_value)

    standard_test_cases = [[sollya.parse(x)] for x in  ["0x1.107a78p+0", "0x1.9e75a6p+0"]]



class MetaAtan2(ScalarBinaryFunction, MetaAtan):
    """ Meta-function for 2-argument arc tangent (atan2) """

    def generate_scalar_scheme(self, vy, vx):
        # as in standard library atan2(y, x), take y as first
        # parameter and x as second, we inverse vy and vx in method
        # argument list
        # extract of atan2 specification from man page
        # If y is +0 (-0) and x is less than 0, +pi (-pi) is returned.
        # If y is +0 (-0) and x is greater than 0, +0 (-0) is returned.
        # If y is less than 0 and x is +0 or -0, -pi/2 is returned.
        # If y is greater than 0 and x is +0 or -0, pi/2 is returned.
        # If either x or y is NaN, a NaN is returned.
        # If y is +0 (-0) and x is -0, +pi (-pi) is returned.
        # If y is +0 (-0) and x is +0, +0 (-0) is returned.
        # If  y  is  a  finite  value  greater  (less)  than 0, and x is negative infinity, +pi (-pi) is
        # returned.
        # If y is a finite value greater (less) than 0, and x is positive infinity, +0 (-0) is returned.
        # If y is positive infinity (negative infinity), and x is finite, pi/2 (-pi/2) is returned.
        # If y is positive infinity (negative infinity) and x is negative infinity, +3*pi/4 (-3*pi/4) is
        # returned.
        # If  y  is  positive  infinity (negative infinity) and x is positive infinity, +pi/4 (-pi/4) is
        # returned.
        vy.set_attributes(tag="y")
        vx.set_attributes(tag="x")
        return self.generic_atan2_generate(vy, vx)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetaAtan,
                builtin from a default argument mapping overloaded with @p kw
        """
        default_args_exp = {
            "output_file": "my_atan2.c",
            "function_name": "my_atan2",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "input_intervals": [DefaultArgTemplate.input_intervals[0]] * 2,
            "method": "piecewise",
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)


    def numeric_emulate(self, vy, vx):
        return sollya.atan(vy / vx)

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=MetaAtan.get_default_args())
    # extra options
    arg_template.get_parser().add_argument(
        "--method", dest="method", default="piecewise", choices=["piecewise", "single"],
        action="store", help="select approximation method")

    args = arg_template.arg_extraction()
    ml_atan = MetaAtan(args)
    ml_atan.gen_implementation()
