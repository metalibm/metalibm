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
import sollya

from sollya import (
    Interval, round, inf, sup, log, expm1, log2,
    guessdegree, dirtyinfnorm, floor,
    SollyaObject
)

S2 = SollyaObject(2)

from metalibm_core.core.ml_operations import (
    Test, RaiseReturn, Comparison, Statement, NearestInteger,
    ConditionBlock, Return, ClearException, ExponentInsertion,
    Constant, Variable, Addition, Subtraction,
    LogicalNot, LogicalOr,
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Int32,
    ML_FPE_Invalid, ML_FPE_Overflow, ML_FPE_Underflow,
    ML_FPE_Inexact
)
from metalibm_core.core.precisions import (
    ML_Faithful, ML_CorrectlyRounded, ML_DegradedAccuracyAbsolute,
    ML_DegradedAccuracyRelative
)
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.core.polynomials import (
    PolynomialSchemeEvaluator, Polynomial
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Arg
)
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.core.special_values import (
    FP_QNaN, FP_PlusInfty, FP_PlusZero
)
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import (
    debug_multi
)
from metalibm_core.utility.num_utils   import ulp
import metalibm_core.utility.gappa_utils as gappa_utils


class ML_Exponential(ScalarUnaryFunction):
    function_name = "ml_exp"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Exponential,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_exp.c",
            "function_name": "ml_exp",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scalar_scheme(self, vx):
        Log.set_dump_stdout(True)

        Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
        if self.debug_flag:
            Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
            kwords["arg_value"] = vx
            kwords["function_name"] = self.function_name
            if self.libm_compliant:
                return RaiseReturn(*args, precision=self.precision, **kwords)
            else:
                return Return(kwords["return_value"], precision=self.precision)

        test_nan_or_inf = Test(
            vx, specifier=Test.IsInfOrNaN, likely=False,
            debug=debug_multi, tag="nan_or_inf")
        test_nan = Test(
            vx, specifier=Test.IsNaN, debug=debug_multi, tag="is_nan_test")
        test_positive = Comparison(
            vx, 0, specifier=Comparison.GreaterOrEqual, debug=debug_multi,
            tag="inf_sign")

        test_signaling_nan = Test(
            vx, specifier=Test.IsSignalingNaN, debug=debug_multi,
            tag="is_signaling_nan")
        return_snan = Statement(
            ExpRaiseReturn(ML_FPE_Invalid, return_value=FP_QNaN(self.precision))
        )

        # return in case of infinity input
        infty_return = Statement(
            ConditionBlock(
                test_positive,
                Return(FP_PlusInfty(self.precision), precision=self.precision),
                Return(FP_PlusZero(self.precision), precision=self.precision)
            )
        )
        # return in case of specific value input (NaN or inf)
        specific_return = ConditionBlock(
            test_nan,
            ConditionBlock(
                test_signaling_nan,
                return_snan,
                Return(FP_QNaN(self.precision), precision=self.precision)
            ),
            infty_return)
        # return in case of standard (non-special) input

        # exclusion of early overflow and underflow cases
        precision_emax      = self.precision.get_emax()
        precision_max_value = S2 * S2**precision_emax
        exp_overflow_bound  = sollya.ceil(log(precision_max_value))
        early_overflow_test = Comparison(
            vx, exp_overflow_bound,
            likely=False, specifier=Comparison.Greater)
        early_overflow_return = Statement(
            ClearException() if self.libm_compliant else Statement(),
            ExpRaiseReturn(
                ML_FPE_Inexact, ML_FPE_Overflow,
                return_value=FP_PlusInfty(self.precision)
            )
        )

        precision_emin = self.precision.get_emin_subnormal()
        precision_min_value = S2 ** precision_emin
        exp_underflow_bound = floor(log(precision_min_value))

        early_underflow_test = Comparison(
            vx, exp_underflow_bound,
            likely=False, specifier=Comparison.Less)
        early_underflow_return = Statement(
            ClearException() if self.libm_compliant else Statement(),
            ExpRaiseReturn(
                ML_FPE_Inexact, ML_FPE_Underflow,
                return_value=FP_PlusZero(self.precision)))

        # constant computation
        invlog2 = self.precision.round_sollya_object(1/log(2), sollya.RN)

        interval_vx = Interval(exp_underflow_bound, exp_overflow_bound)
        interval_fk = interval_vx * invlog2
        interval_k = Interval(floor(inf(interval_fk)), sollya.ceil(sup(interval_fk)))


        log2_hi_precision = self.precision.get_field_size() - (sollya.ceil(log2(sup(abs(interval_k)))) + 2)
        Log.report(Log.Info, "log2_hi_precision: %d" % log2_hi_precision)
        invlog2_cst = Constant(invlog2, precision = self.precision)
        log2_hi = round(log(2), log2_hi_precision, sollya.RN)
        log2_lo = self.precision.round_sollya_object(log(2) - log2_hi, sollya.RN)

        # argument reduction
        unround_k = vx * invlog2
        unround_k.set_attributes(tag = "unround_k", debug = debug_multi)
        k = NearestInteger(unround_k, precision = self.precision, debug = debug_multi, tag="k")
        ik = NearestInteger(unround_k, precision = self.precision.get_integer_format(), debug = debug_multi, tag="ik")
        exact_pre_mul = (k * log2_hi)
        exact_pre_mul.set_attributes(exact= True)
        exact_hi_part = vx - exact_pre_mul
        exact_hi_part.set_attributes(exact = True, tag = "exact_hi", debug = debug_multi, prevent_optimization = True)
        exact_lo_part = - k * log2_lo
        exact_lo_part.set_attributes(tag = "exact_lo", debug = debug_multi, prevent_optimization = True)
        r =  exact_hi_part + exact_lo_part
        r.set_tag("r")
        r.set_attributes(debug = debug_multi)

        approx_interval = Interval(-log(2)/2, log(2)/2)

        approx_interval_half = approx_interval / 2
        approx_interval_split = [Interval(-log(2)/2, inf(approx_interval_half)), approx_interval_half, Interval(sup(approx_interval_half), log(2)/2)]

        # TODO: should be computed automatically
        exact_hi_interval = approx_interval
        exact_lo_interval = - interval_k * log2_lo

        opt_r = self.optimise_scheme(r, copy = {})

        tag_map = {}
        self.opt_engine.register_nodes_by_tag(opt_r, tag_map)

        cg_eval_error_copy_map = {
            vx: Variable("x", precision=self.precision, interval=interval_vx),
            tag_map["k"]: Variable("k", interval=interval_k, precision=self.precision)
        }

        #try:
        if gappa_utils.is_gappa_installed():
            eval_error = self.gappa_engine.get_eval_error_v2(
                self.opt_engine, opt_r, cg_eval_error_copy_map,
                gappa_filename=gappa_utils.generate_gappa_filename("red_arg.g"))
        else:
            eval_error = 0.0
            Log.report(Log.Warning, "gappa is not installed in this environnement")
        Log.report(Log.Info, "eval error: %s" % eval_error)


        local_ulp = sup(ulp(sollya.exp(approx_interval), self.precision))
        # FIXME refactor error_goal from accuracy
        Log.report(Log.Info, "accuracy: %s" % self.accuracy)
        if isinstance(self.accuracy, ML_Faithful):
            error_goal = local_ulp
        elif isinstance(self.accuracy, ML_CorrectlyRounded):
            error_goal = S2**-1 * local_ulp
        elif isinstance(self.accuracy, ML_DegradedAccuracyAbsolute):
            error_goal = self.accuracy.goal
        elif isinstance(self.accuracy, ML_DegradedAccuracyRelative):
            error_goal = self.accuracy.goal
        else:
            Log.report(Log.Error, "unknown accuracy: %s" % self.accuracy)


        # error_goal = local_ulp #S2**-(self.precision.get_field_size()+1)
        error_goal_approx = S2**-1 * error_goal

        Log.report(Log.Info, "\033[33;1m building mathematical polynomial \033[0m\n")
        poly_degree = max(sup(guessdegree(expm1(sollya.x)/sollya.x, approx_interval, error_goal_approx)) - 1, 2)
        init_poly_degree = poly_degree

        error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

        polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
        #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

        MAX_NUM_ITERATION = 20

        for _ in range(MAX_NUM_ITERATION):
            Log.report(Log.Info, "attempting poly degree: %d" % poly_degree)
            precision_list = [1] + [self.precision] * (poly_degree)
            poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(expm1(sollya.x), poly_degree, precision_list, approx_interval, sollya.absolute, error_function = error_function)
            Log.report(Log.Info, "polynomial: %s " % poly_object)
            sub_poly = poly_object.sub_poly(start_index = 2)
            Log.report(Log.Info, "polynomial: %s " % sub_poly)

            Log.report(Log.Info, "poly approx error: %s" % poly_approx_error)

            Log.report(Log.Info, "\033[33;1m generating polynomial evaluation scheme \033[0m")
            pre_poly = polynomial_scheme_builder(poly_object, r, unified_precision = self.precision)
            pre_poly.set_attributes(tag = "pre_poly", debug = debug_multi)

            pre_sub_poly = polynomial_scheme_builder(sub_poly, r, unified_precision = self.precision)
            pre_sub_poly.set_attributes(tag = "pre_sub_poly", debug = debug_multi)

            poly = 1 + (exact_hi_part + (exact_lo_part + pre_sub_poly))
            poly.set_tag("poly")

            # optimizing poly before evaluation error computation
            #opt_poly = self.opt_engine.optimization_process(poly, self.precision, fuse_fma = fuse_fma)
            #opt_sub_poly = self.opt_engine.optimization_process(pre_sub_poly, self.precision, fuse_fma = fuse_fma)
            opt_poly = self.optimise_scheme(poly)
            opt_sub_poly = self.optimise_scheme(pre_sub_poly)

            # evaluating error of the polynomial approximation
            r_gappa_var        = Variable("r", precision = self.precision, interval = approx_interval)
            exact_hi_gappa_var = Variable("exact_hi", precision = self.precision, interval = exact_hi_interval)
            exact_lo_gappa_var = Variable("exact_lo", precision = self.precision, interval = exact_lo_interval)
            vx_gappa_var       = Variable("x", precision = self.precision, interval = interval_vx)
            k_gappa_var        = Variable("k", interval = interval_k, precision = self.precision)


            #print "exact_hi interval: ", exact_hi_interval

            sub_poly_error_copy_map = {
                #r.get_handle().get_node(): r_gappa_var,
                #vx.get_handle().get_node():  vx_gappa_var,
                exact_hi_part.get_handle().get_node(): exact_hi_gappa_var,
                exact_lo_part.get_handle().get_node(): exact_lo_gappa_var,
                #k.get_handle().get_node(): k_gappa_var,
            }

            poly_error_copy_map = {
                exact_hi_part.get_handle().get_node(): exact_hi_gappa_var,
                exact_lo_part.get_handle().get_node(): exact_lo_gappa_var,
            }


            if gappa_utils.is_gappa_installed():
                sub_poly_eval_error = -1.0
                gappa_sub_poly_filename = gappa_utils.generate_gappa_filename("{}_gappa_sub_poly.g".format(self.function_name))
                sub_poly_eval_error = self.gappa_engine.get_eval_error_v2(self.opt_engine, opt_sub_poly, sub_poly_error_copy_map, gappa_filename =gappa_sub_poly_filename)

                dichotomy_map = [
                    {
                        exact_hi_part.get_handle().get_node(): approx_interval_split[0],
                    },
                    {
                        exact_hi_part.get_handle().get_node(): approx_interval_split[1],
                    },
                    {
                        exact_hi_part.get_handle().get_node(): approx_interval_split[2],
                    },
                ]
                gappa_poly_filename = gappa_utils.generate_gappa_filename("gappa_poly.g")
                poly_eval_error_dico = self.gappa_engine.get_eval_error_v3(self.opt_engine, opt_poly, poly_error_copy_map, gappa_filename=gappa_poly_filename, dichotomy = dichotomy_map)

                poly_eval_error = max([sup(abs(err)) for err in poly_eval_error_dico])
            else:
                poly_eval_error = 0.0
                sub_poly_eval_error = 0.0
                Log.report(Log.Warning, "gappa is not installed in this environnement")
                Log.report(Log.Info, "stopping autonomous degree research")
                # incrementing polynomial degree to counteract initial decrementation effect
                poly_degree += 1
                break
            Log.report(Log.Info, "poly evaluation error: %s" % poly_eval_error)
            Log.report(Log.Info, "sub poly evaluation error: %s" % sub_poly_eval_error)

            global_poly_error     = None
            global_rel_poly_error = None

            for case_index in range(3):
                poly_error = poly_approx_error + poly_eval_error_dico[case_index]
                rel_poly_error = sup(abs(poly_error / sollya.exp(approx_interval_split[case_index])))
                if global_rel_poly_error == None or rel_poly_error > global_rel_poly_error:
                    global_rel_poly_error = rel_poly_error
                    global_poly_error = poly_error
            flag = error_goal > global_rel_poly_error


            if flag:
                break
            else:
                poly_degree += 1

        late_overflow_test = Comparison(
            ik, self.precision.get_emax(),
            specifier=Comparison.Greater, likely=False,
            debug=debug_multi, tag="late_overflow_test")
        overflow_exp_offset = int(self.precision.get_emax() - self.precision.get_field_size() / 2)
        cst_overflow_exp_offset = Constant(overflow_exp_offset, precision=self.precision.get_integer_format())
        diff_k = Subtraction(
            ik,
            cst_overflow_exp_offset,
            precision=self.precision.get_integer_format(),
            debug=debug_multi,
            tag="diff_k",
        )
        late_overflow_result = (ExponentInsertion(diff_k, precision = self.precision) * poly) * ExponentInsertion(cst_overflow_exp_offset, precision = self.precision)
        late_overflow_result.set_attributes(silent = False, tag = "late_overflow_result", debug = debug_multi, precision = self.precision)
        late_overflow_return = ConditionBlock(Test(late_overflow_result, specifier = Test.IsInfty, likely = False), ExpRaiseReturn(ML_FPE_Overflow, return_value = FP_PlusInfty(self.precision)), Return(late_overflow_result, precision=self.precision))

        late_underflow_test = Comparison(k, self.precision.get_emin_normal(), specifier = Comparison.LessOrEqual, likely=False, tag="late_underflow_test")
        underflow_exp_offset = 2 * self.precision.get_field_size()
        corrected_exp = Addition(
          ik,
          Constant(
            underflow_exp_offset,
            precision=self.precision.get_integer_format()
          ),
          precision=self.precision.get_integer_format(),
          tag="corrected_exp"
        )
        late_underflow_result = (ExponentInsertion(corrected_exp, precision = self.precision) * poly) * ExponentInsertion(-underflow_exp_offset, precision = self.precision)
        late_underflow_result.set_attributes(debug = debug_multi, tag = "late_underflow_result", silent = False)
        test_subnormal = Test(late_underflow_result, specifier = Test.IsSubnormal)
        late_underflow_return = Statement(ConditionBlock(test_subnormal, ExpRaiseReturn(ML_FPE_Underflow, return_value = late_underflow_result)), Return(late_underflow_result, precision=self.precision))

        twok = ExponentInsertion(ik, tag = "exp_ik", debug = debug_multi, precision = self.precision)
        #std_result = twok * ((1 + exact_hi_part * pre_poly) + exact_lo_part * pre_poly) 
        std_result = twok * poly
        std_result.set_attributes(tag = "std_result", debug = debug_multi)
        std_cond = LogicalNot(LogicalOr(late_overflow_test, late_underflow_test), likely=True)

        result_scheme = ConditionBlock(
            std_cond,
            Return(std_result, precision=self.precision),
            ConditionBlock(
                late_overflow_test,
                late_overflow_return,
                late_underflow_return,
            )
        )
        std_return = ConditionBlock(early_overflow_test, early_overflow_return, ConditionBlock(early_underflow_test, early_underflow_return, result_scheme))

        # main scheme
        Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
        scheme = ConditionBlock(
            test_nan_or_inf,
            Statement(
                ClearException() if self.libm_compliant else Statement(),
                specific_return
            ),
            std_return
        )

        return scheme

    def numeric_emulate(self, input_value):
        """ Numeric emulation of exponential """
        return sollya.exp(input_value)

    @property
    def standard_test_cases(self):
        return [
            (sollya.parse("0xbf50bc3a"),),
            (sollya.parse("0x1.0p-126"),),
            (sollya.parse("0x1.0p-127"),),
            (sollya.parse("-0x1.fffffep126"),),
            (sollya.parse("-infty"),),
            (sollya.parse("infty"),),
            (FP_QNaN(self.precision),),
            # issue in generic newlib implementation
            (sollya.parse("0x1.62e302p+6"),),
        ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Exponential.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    ml_exp = ML_Exponential(args)

    ml_exp.gen_implementation()
