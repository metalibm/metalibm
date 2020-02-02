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
# last-modified:    Oct 5th, 2018
#
# Description:      Meta-implementation of floating-point division
###############################################################################
import sys
import sollya

from sollya import Interval
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_NewArgTemplate, DefaultArgTemplate
from metalibm_core.utility.debug_utils import debug_multi

class NR_Iteration(object):
    """ Newton-Raphson iteration generator """
    def __init__(self, approx, divisor, force_fma=False):
        """ 
            @param approx initial approximation of 1.0 / @p divisor
            @param divisor reciprocal input
            @param force_fma force the use of Fused Multiply and Add """
        self.approx = approx
        self.divisor = divisor
        self.force_fma = force_fma
        if force_fma:
            self.error = FusedMultiplyAdd(divisor, approx, 1.0, specifier=FusedMultiplyAdd.SubtractNegate)
            self.new_approx = FusedMultiplyAdd(self.error, self.approx, self.approx, specifier=FusedMultiplyAdd.Standard)
        else:
            self.error = 1 - divisor * approx
            self.new_approx = self.approx + self.error * self.approx

    def get_hint_rules(self, gcg, gappa_code, exact):
        divisor = self.divisor.get_handle().get_node()
        approx = self.approx.get_handle().get_node()
        new_approx = self.new_approx.get_handle().get_node()

        Attributes.set_default_precision(ML_Exact)

        if self.force_fma:
            rule0 = FusedMultiplyAdd(divisor, approx, 1.0, specifier = FusedMultiplyAdd.SubtractNegate)
        else:
            rule0 = 1.0 - divisor * approx
        rule1 = 1.0 - divisor * (approx - exact) - 1.0
        
        rule2 = new_approx - exact
        subrule = approx * (2 - divisor * approx)
        rule3 = (new_approx - subrule) - (approx - exact) * (approx - exact) * divisor

        if self.force_fma:
            new_error = FusedMultiplyAdd(divisor, approx, 1.0, specifier = FusedMultiplyAdd.SubtractNegate)
            rule4 = FusedMultiplyAdd(new_error, approx, approx)
        else:
            rule4 = approx + (1 - divisor * approx) * approx

        Attributes.unset_default_precision()

        # registering hints
        gcg.add_hint(gappa_code, rule0, rule1)
        gcg.add_hint(gappa_code, rule2, rule3)
        gcg.add_hint(gappa_code, subrule, rule4)

def dividend_mult(div_approx, inv_approx, dividend, divisor, index):
    """ Second part of iteration to converge to dividend / divisor
        from inv_approx ~ 1 / divisor 
        and  div_approx ~ dividend / divisor """
    # yerr = dividend - div_approx * divisor
    yerr = FMSN(div_approx, divisor, dividend)
    yerr.set_attributes(tag="yerr%d" % index, debug=debug_multi)
    # new_div = div_approx + yerr * inv_approx
    new_div = FMA(yerr, inv_approx, div_approx)
    new_div.set_attributes(tag="new_div%d" % index, debug=debug_multi)
    return new_div

def compute_reduced_reciprocal(init_approx, vy, num_iteration):
    """ Compute the correctly rounded approximation of 1.0 / vy
        using @p init_approx as starting point and execution
        @p num_iteration Newton-Raphson iteration(s) """
    current_approx = init_approx
    inv_iteration_list = []

    # compute precision (up to accuracy) approximation of 1 / _vy
    for i in range(num_iteration):
        new_iteration = NR_Iteration(current_approx, vy, force_fma=True) #False if (i != num_iteration - 1) else True)
        inv_iteration_list.append(new_iteration)
        current_approx = new_iteration.new_approx
        current_approx.set_attributes(tag="iter_%d" % i, debug=debug_multi)

    # multiplication correction iteration
    # to get correctly rounded full division _vx / _vy
    current_approx.set_attributes(tag = "final_recp_approx", debug=debug_multi)
    return inv_iteration_list, current_approx


def compute_reduced_division(vx, vy, recp_approx):
    """ From an initial accurate approximation @p recp_approx of 1.0 / vy, computes
        an approximation to accuracy @p accuracy of vx / vy """
    # vx and vy are assumed to be in [1, 2[
    # which means vx / vy is in [0.5, 2]

    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    Attributes.set_default_silent(True)

    # multiplication correction iteration
    # to get correctly rounded full division _vx / _vy
    current_div_approx = vx * recp_approx
    num_dividend_mult_iteration = 1
    for i in range(num_dividend_mult_iteration):
        current_div_approx = dividend_mult(current_div_approx, recp_approx, vx, vy, i)

    # last iteration
    yerr_last = FMSN(current_div_approx, vy, vx) #, clearprevious = True)
    Attributes.unset_default_rounding_mode()
    Attributes.unset_default_silent()
    last_div_approx = FMA(
        yerr_last, recp_approx, current_div_approx, rounding_mode=ML_GlobalRoundMode)

    yerr_last.set_attributes(tag = "yerr_last", debug=debug_multi)

    result = last_div_approx
    return yerr_last, result


def scaling_div_result(div_approx, scaling_ex, scaling_factor_y, precision):
    """ Reconstruct division result from approximation of scaled inputs 
        vx was scaled by scaling_factor_x = 2**-ex
        vy was scaled by scaling_factor_y = 2**-ey 
        so real result is
            = div_approx * scaling_factor_y / scaling_factor_x 
            = div_approx * 2**(-ey + ex) """
    # To avoid overflow / underflow when computing 2**(-ey + ex)
    # the scaling could be performed in 2 steps
    #      1. multiplying by 2**-ey
    #      2. multiplying by 2**ex
    unscaling_ex = ExponentInsertion(scaling_ex, precision=precision)

    unscaled_result = div_approx * unscaling_ex * scaling_factor_y
    unscaled_result.set_attributes(debug=debug_multi, tag="unscaled_result")
    return unscaled_result


def subnormalize_result(recp_approx, div_approx, ex, ey, yerr_last, precision):
    """ If the result of the division is subnormal,
        an extended approximation of division must first be obtained
        and then subnormalize to ensure correct rounding """
    # TODO: fix extended precision determination
    extended_precision = {
        ML_Binary64: ML_DoubleDouble,
        ML_Binary32: ML_SingleSingle,
    }[precision]

    # we make an extra step in extended precision
    ext_pre_result = FMA(yerr_last, recp_approx, div_approx, precision=extended_precision, tag="ext_pre_result")
    # subnormalize the result according to final result exponent
    subnormal_pre_result_ext = SpecificOperation(
        ext_pre_result,
        ex - ey,
        precision=extended_precision,
        specifier=SpecificOperation.Subnormalize,
        tag="subnormal_pre_result",
        debug=debug_multi)
    subnormal_pre_result = subnormal_pre_result_ext.hi
    sub_scale_factor = ex - ey
    subnormal_result = subnormal_pre_result * ExponentInsertion(sub_scale_factor, precision=precision)

    return subnormal_result 

def bit_match(fp_optree, bit_id, likely = False, **kwords):
    return NotEqual(BitLogicAnd(TypeCast(fp_optree, precision = ML_Int64), 1 << bit_id), 0, likely = likely, **kwords)


def extract_and_inject_sign(sign_source, sign_dest, int_precision = ML_Int64, fp_precision = ML_Binary64, **kwords):
    int_sign_dest = sign_dest if isinstance(sign_dest.get_precision(), ML_Fixed_Format) else TypeCast(sign_dest, precision = int_precision)
    return TypeCast(BitLogicOr(BitLogicAnd(TypeCast(sign_source, precision = int_precision), 1 << (sign_source.precision.bit_size - 1)), int_sign_dest), precision = fp_precision)


class ML_Division(ML_FunctionBasis):
    function_name = "ml_div"
    arity = 2

    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args=args)
        self.num_iter = args.num_iter
    @staticmethod
    def get_default_args(**args):
        """ Generate a default argument structure set specifically for
            the Hyperbolic Cosine """
        default_div_args = {
            "precision": ML_Binary32,
            "accuracy": ML_CorrectlyRounded,
            "target": GenericProcessor(),
            "output_file": "my_div.c",
            "function_name": "my_div",
            "language": C_Code,
            "num_iter": 3,
            "passes": ["typing:basic_legalization", "beforecodegen:expand_multi_precision"],
            "vector_size": 1,
            "arity": ML_Division.arity,
        }
        default_div_args.update(args)
        return DefaultArgTemplate(**default_div_args)

    def generate_scheme(self):
        # We wish to compute vx / vy
        vx = self.implementation.add_input_variable("x", self.precision)
        vy = self.implementation.add_input_variable("y", self.precision)

        # maximum exponent magnitude (to avoid overflow/ underflow during
        # intermediary computations
        int_prec = self.precision.get_integer_format()
        max_exp_mag = Constant(self.precision.get_emax() - 3, precision=int_prec)

        exact_ex = ExponentExtraction(vx, tag = "exact_ex", precision=int_prec, debug=debug_multi)
        exact_ey = ExponentExtraction(vy, tag = "exact_ey", precision=int_prec, debug=debug_multi)

        ex = Max(Min(exact_ex, max_exp_mag, precision=int_prec), -max_exp_mag, tag="ex", precision=int_prec)
        ey = Max(Min(exact_ey, max_exp_mag, precision=int_prec), -max_exp_mag, tag="ey", precision=int_prec)


        Attributes.set_default_rounding_mode(ML_RoundToNearest)
        Attributes.set_default_silent(True)

        # computing the inverse square root
        init_approx = None

        scaling_factor_x = ExponentInsertion(-ex, tag="sfx_ei", precision=self.precision, debug=debug_multi) 
        scaling_factor_y = ExponentInsertion(-ey, tag="sfy_ei", precision=self.precision, debug=debug_multi) 

        # scaled version of vx and vy
        scaled_vx = vx * scaling_factor_x
        scaled_vy = vy * scaling_factor_y

        scaled_vx.set_attributes(tag="scaled_vx", debug=debug_multi)
        scaled_vy.set_attributes(tag="scaled_vy", debug=debug_multi)


        # We need a first approximation to 1 / scaled_vy
        dummy_seed = ReciprocalSeed(EmptyOperand(precision=self.precision), precision=self.precision)

        if self.processor.is_supported_operation(dummy_seed):
            init_approx = ReciprocalSeed(scaled_vy, precision=self.precision, tag="init_approx", debug=debug_multi)

        else:
            # generate tabulated version of seed
            raise NotImplementedError


        current_approx_std = init_approx
        # correctly-rounded inverse computation
        num_iteration = self.num_iter

        Attributes.unset_default_rounding_mode()
        Attributes.unset_default_silent()


        # check if inputs are zeros
        x_zero = Test(vx, specifier=Test.IsZero, likely=False, precision=ML_Bool)
        y_zero = Test(vy, specifier=Test.IsZero, likely=False, precision=ML_Bool)

        comp_sign = Test(vx, vy, specifier=Test.CompSign, tag = "comp_sign", debug = debug_multi )

        # check if divisor is NaN
        y_nan = Test(vy, specifier=Test.IsNaN, likely=False, precision=ML_Bool)

        # check if inputs are signaling NaNs
        x_snan = Test(vx, specifier=Test.IsSignalingNaN, likely=False, precision=ML_Bool)
        y_snan = Test(vy, specifier=Test.IsSignalingNaN, likely=False, precision=ML_Bool)

        # check if inputs are infinities
        x_inf = Test(vx, specifier=Test.IsInfty, likely=False, tag="x_inf", precision=ML_Bool)
        y_inf = Test(vy, specifier=Test.IsInfty, likely=False, tag="y_inf", debug=debug_multi, precision=ML_Bool)


        scheme = None
        gappa_vx, gappa_vy = None, None
        # gappa_init_approx = None
        # gappa_current_approx = None


        # initial reciprocal approximation of 1.0 / scaled_vy
        inv_iteration_list, recp_approx = compute_reduced_reciprocal(init_approx, scaled_vy, self.num_iter)

        recp_approx.set_attributes(tag="recp_approx", debug=debug_multi)

        # approximation of scaled_vx / scaled_vy
        yerr_last, reduced_div_approx = compute_reduced_division(scaled_vx, scaled_vy, recp_approx)

        unscaled_result = scaling_div_result(reduced_div_approx, ex, scaling_factor_y, self.precision)

        subnormal_result = subnormalize_result(recp_approx, reduced_div_approx, exact_ex, exact_ey, yerr_last, self.precision)

        x_inf_or_nan = Test(vx, specifier = Test.IsInfOrNaN, likely=False)
        y_inf_or_nan = Test(vy, specifier = Test.IsInfOrNaN, likely=False, tag="y_inf_or_nan", debug = debug_multi)

        # gappa_vx = scaled_vx
        # gappa_vy = scaled_vy
        # gappa_init_approx = init_approx

        # managing special cases
        # x inf and y inf
        pre_scheme = ConditionBlock(x_inf_or_nan,
            ConditionBlock(x_inf,
                ConditionBlock(y_inf_or_nan,
                    Statement(
                        ConditionBlock(y_snan, Raise(ML_FPE_Invalid)),
                        Return(FP_QNaN(self.precision)),
                    ),
                    ConditionBlock(
                        comp_sign,
                        Return(FP_MinusInfty(self.precision)),
                        Return(FP_PlusInfty(self.precision)))
                ),
                Statement(
                    ConditionBlock(x_snan, Raise(ML_FPE_Invalid)),
                    Return(FP_QNaN(self.precision))
                )
            ),
            ConditionBlock(x_zero,
                ConditionBlock(LogicalOr(y_zero, y_nan, precision=ML_Bool),
                    Statement(
                        ConditionBlock(y_snan, Raise(ML_FPE_Invalid)),
                        Return(FP_QNaN(self.precision))
                    ),
                    Return(vx)
                ),
                ConditionBlock(y_inf_or_nan,
                    ConditionBlock(y_inf,
                        Return(
                            Select(
                                comp_sign,
                                FP_MinusZero(self.precision),
                                FP_PlusZero(self.precision))),
                        Statement(
                            ConditionBlock(y_snan, Raise(ML_FPE_Invalid)),
                            Return(FP_QNaN(self.precision))
                        )
                    ),
                    ConditionBlock(y_zero,
                        Statement(
                            Raise(ML_FPE_DivideByZero),
                            ConditionBlock(comp_sign,
                                Return(FP_MinusInfty(self.precision)),
                                Return(FP_PlusInfty(self.precision))
                            )
                        ),
                        # managing numerical value result cases
                        ConditionBlock(
                            Test(unscaled_result, specifier=Test.IsSubnormal, likely=False),
                            # result is subnormal
                            Statement(
                                ConditionBlock(
                                    Comparison(
                                        yerr_last, 0,
                                        specifier=Comparison.NotEqual, likely=True),
                                    Statement(Raise(ML_FPE_Inexact, ML_FPE_Underflow))
                                ),
                                Return(subnormal_result),
                            ),
                            # result is normal
                            Statement(
                                ConditionBlock(
                                    Comparison(
                                        yerr_last, 0,
                                        specifier=Comparison.NotEqual, likely=True),
                                    Raise(ML_FPE_Inexact)
                                ),
                                Return(unscaled_result)
                            )
                        )
                    )
                )
            )
        )
        # managing rounding mode save and restore
        # to ensure intermediary computations are performed in round-to-nearest
        # clearing exception before final computation

        #rnd_mode = GetRndMode()
        #scheme = Statement(
        #    rnd_mode,
        #    SetRndMode(ML_RoundToNearest),
        #    yerr_last,
        #    SetRndMode(rnd_mode),
        #    unscaled_result,
        #    ClearException(),
        #    pre_scheme
        #)

        scheme = pre_scheme

        return scheme

    def numeric_emulate(self, x, y):
        return x / y

    def misc(self):
        print("Gappa script generation")
        seed_var = Variable("seed", precision = self.precision, interval = Interval(0.5, 1))
        cg_eval_error_copy_map = {
            gappa_init_approx.get_handle().get_node(): seed_var,
            gappa_vx.get_handle().get_node(): Variable("x", precision = self.precision, interval = Interval(1, 2)),
            gappa_vy.get_handle().get_node(): Variable("y", precision = self.precision, interval = Interval(1, 2)),
        }
        G1 = Constant(1, precision = ML_Exact)
        exact = G1 / gappa_vy
        exact.set_precision(ML_Exact)
        exact.set_tag("div_exact")
        gappa_goal = gappa_current_approx.get_handle().get_node() - exact
        gappa_goal.set_precision(ML_Exact)
        gappacg = GappaCodeGenerator(target, declare_cst = False, disable_debug = True)
        gappa_code = gappacg.get_interval_code(gappa_goal, cg_eval_error_copy_map)

        new_exact_node = exact.get_handle().get_node()

        for nr in inv_iteration_list:
            nr.get_hint_rules(gappacg, gappa_code, new_exact_node)

        seed_wrt_exact = seed_var - new_exact_node
        seed_wrt_exact.set_precision(ML_Exact)
        gappacg.add_hypothesis(gappa_code, seed_wrt_exact, Interval(-S2**-7, S2**-7))

        try:
            eval_error = execute_gappa_script_extract(gappa_code.get(gappacg))["goal"]
            print("eval_error: "), eval_error
        except:
            print("error during gappa run")

    standard_test_cases = [
        (sollya.parse("-0x1.34a246p-2"), sollya.parse("-0x1.26e2e2p-1")),
    ]



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_arg=ML_Division.get_default_args()
    )
    arg_template.get_parser().add_argument(
         "--num-iter", dest="num_iter", default=3, type=int,
        action="store", help="number of newton-raphson iterations")

    args = arg_template.arg_extraction()

    ml_div = ML_Division(args)
    ml_div.gen_implementation()
