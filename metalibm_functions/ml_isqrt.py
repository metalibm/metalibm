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
# last-modified:        Mar    7th, 2018
#
# Description: Implementation of reciprocal square root
###############################################################################

import sollya


from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_operations import *
from metalibm_core.core.special_values import (
    FP_PlusZero, FP_PlusInfty, FP_MinusInfty, FP_QNaN
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi

def generate_NR_iteration_recp_sqrt(value, approx, half_value, precision, c_half):
    """ Generate the optree graph for one Newton-Raphson iteration of reciprocal
        square root
        @param value input value (goal is to refine rsqrt(value)
        @param approx the previous approximation
        @param half_value 0.5 * value (factorized node)
        @param precision format for the iteration
        @param c_half shared constant 0.5
        @return new approximation """
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    Attributes.set_default_silent(True)

    square = approx * approx
    mult = FMSN(half_value, square, c_half)
    new_approx = FMA(approx, mult, approx)

    Attributes.unset_default_rounding_mode()
    Attributes.unset_default_silent()

    return new_approx


def compute_isqrt(vx, init_approx, num_iter, precision, debug_lftolx=None):
    """ Compute an approximation of reciprocal square root on @p vx """
    C_half = Constant(0.5, precision = precision)
    h = C_half * vx
    h.set_attributes(tag = "h", debug = debug_multi, silent = True, rounding_mode = ML_RoundToNearest)

    current_approx = init_approx
    # correctly-rounded inverse computation
    for i in range(num_iter):
        current_approx = generate_NR_iteration_recp_sqrt(vx, current_approx, h,
                                                        precision, C_half)
        current_approx.set_attributes(tag="iter_%d" % i, debug=debug_multi)

    final_approx = current_approx
    final_approx.set_attributes(tag="final_approx", debug=debug_multi)

    return final_approx



class ML_Isqrt(ML_FunctionBasis):
    """ Reciprocal Square Root """
    function_name = "ml_rsqrt"
    def __init__(self, args):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)
        self.num_iter = args.num_iter


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Isqrt,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_isqrt = {
            "output_file": "my_isqrt.c",
            "function_name": "my_isqrt",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "num_iter": 4,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_isqrt.update(kw)
        return DefaultArgTemplate(**default_args_isqrt)

    def generate_scheme(self):
        # declaring target and instantiating optimization engine

        vx = self.implementation.add_input_variable("x", self.precision)
        vx.set_attributes(precision = self.precision, tag = "vx", debug =debug_multi)
        Log.set_dump_stdout(True)

        Log.report(Log.Info, "\033[33;1m Generating implementation scheme \033[0m")
        if self.debug_flag:
            Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

        # local overloading of RaiseReturn operation
        def SqrtRaiseReturn(*args, **kwords):
            kwords["arg_value"] = vx
            kwords["function_name"] = self.function_name
            return RaiseReturn(*args, **kwords)

        C0 = Constant(0, precision = self.precision)
        C0_plus = Constant(FP_PlusZero(self.precision))

        test_NaN = Test(vx, specifier = Test.IsNaN, likely = False, debug = debug_multi, tag = "is_NaN", precision = ML_Bool)
        test_negative = Comparison(vx, C0, specifier = Comparison.Less, debug = debug_multi, tag = "is_Negative", precision = ML_Bool, likely = False)

        test_zero = Comparison(vx, C0_plus, specifier = Comparison.Equal, likely = False, debug = debug_multi, tag = "Is_Zero", precision = ML_Bool)
        test_inf = Test(vx, specifier = Test.IsInfty, likely = False, debug = debug_multi, tag = "is_Inf", precision = ML_Bool)
        test_NaN_or_Neg = LogicalOr(test_NaN, test_negative, precision = ML_Bool, likely = False)

        test_NaN_or_Inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debug_multi, tag = "is_nan_or_inf", precision = ML_Bool)
        test_negative_or_zero = Comparison(vx, C0, specifier = Comparison.LessOrEqual, debug = debug_multi, tag = "is_Negative_or_zero", precision = ML_Bool, likely = False)

        test_std = LogicalNot(LogicalOr(test_NaN_or_Inf, test_negative_or_zero, precision = ML_Bool, likely = False), precision = ML_Bool, likely = True)

        return_PosZero = Statement(Return(FP_PlusInfty(self.precision)))
        return_NegZero = Statement(Return(FP_MinusInfty(self.precision)))
        return_NaN_or_neg = Statement(Return(FP_QNaN(self.precision)))
        return_inf = Statement(Return(C0))

        NR_init = ReciprocalSquareRootSeed(vx, precision = self.precision, tag = "sqrt_seed", debug = debug_multi)
        result = compute_isqrt(vx, NR_init, self.num_iter, self.precision)

        return_non_std = ConditionBlock(
            test_NaN_or_Neg,
            return_NaN_or_neg,
            ConditionBlock(
                test_inf,
                return_inf,
                ConditionBlock(
                    test_zero,
                    return_PosZero,
                    return_NegZero
                )
            )
        )

        scheme = Statement(ConditionBlock(
            test_std,
            Statement(Return(result)),
            Statement(return_non_std)))

        return scheme


    def numeric_emulate(self, input):
        return 1.0 / sollya.sqrt(input)

    standard_test_cases = [
        (1.651028399744791652636877188342623412609100341796875,)
    ]

if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(default_arg=ML_Isqrt.get_default_args())
    arg_template.parser.add_argument(
        "--num-iter", dest="num_iter", action="store", default=3, type=int,
        help="number of Newton-Raphson iterations")
    args = arg_template.arg_extraction()

    ml_isqrt = ML_Isqrt(args)
    ml_isqrt.gen_implementation()

