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

from sollya import S2, Interval

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.common import test_flag_option, extract_option_value  


class ML_Division(object):
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 num_iter = 3,
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "__divsf3.c", 
                 function_name = "__divsf3"):
        # declaring CodeFunction and retrieving input variable
        self.precision = precision
        self.function_name = function_name
        exp_implementation = CodeFunction(self.function_name, output_format = precision)
        vx = exp_implementation.add_input_variable("x", precision) 
        vy = exp_implementation.add_input_variable("y", precision) 
        processor = target

        class NR_Iteration(object):
            def __init__(self, approx, divisor, force_fma = False):
                self.approx = approx
                self.divisor = divisor
                self.force_fma = force_fma
                if force_fma:
                    self.error = FusedMultiplyAdd(divisor, approx, 1.0, specifier = FusedMultiplyAdd.SubtractNegate)
                    self.new_approx = FusedMultiplyAdd(self.error, self.approx, self.approx, specifier = FusedMultiplyAdd.Standard)
                else:
                    self.error = 1 - divisor * approx
                    self.new_approx = self.approx + self.error * self.approx

            def get_new_approx(self):
                return self.new_approx

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

        debugf        = ML_Debug(display_format = "%f")
        debuglf       = ML_Debug(display_format = "%lf")
        debugx        = ML_Debug(display_format = "%x")
        debuglx       = ML_Debug(display_format = "%lx")
        debugd        = ML_Debug(display_format = "%d")
        #debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s)" % v)
        debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\" ev=%x", pre_process = lambda v: "double_to_64b_encoding(%s), __k1_fpu_get_exceptions()" % v)
        debug_ddtolx  = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v))
        debug_dd      = ML_Debug(display_format = "{.hi=%lf, .lo=%lf}", pre_process = lambda v: "%s.hi, %s.lo" % (v, v))

        ex = Max(Min(ExponentExtraction(vx), 1020), -1020, tag = "ex", debug = debugd)
        ey = Max(Min(ExponentExtraction(vy), 1020), -1020, tag = "ey", debug = debugd)

        exact_ex = ExponentExtraction(vx, tag = "exact_ex")
        exact_ey = ExponentExtraction(vy, tag = "exact_ey")

        Attributes.set_default_rounding_mode(ML_RoundToNearest)
        Attributes.set_default_silent(True)

        # computing the inverse square root
        init_approx = None

        scaling_factor_x = ExponentInsertion(-ex, tag = "sfx_ei") 
        scaling_factor_y = ExponentInsertion(-ey, tag = "sfy_ei") 

        scaled_vx = vx * scaling_factor_x
        scaled_vy = vy * scaling_factor_y

        scaled_vx.set_attributes(debug = debug_lftolx, tag = "scaled_vx")
        scaled_vy.set_attributes(debug = debug_lftolx, tag = "scaled_vy")

        scaled_vx.set_precision(ML_Binary64)
        scaled_vy.set_precision(ML_Binary64)

        # forcing vx precision to make processor support test
        init_approx_precision = DivisionSeed(scaled_vx, scaled_vy, precision = self.precision, tag = "seed", debug = debug_lftolx)
        if not processor.is_supported_operation(init_approx_precision):
            if self.precision != ML_Binary32:
                px = Conversion(scaled_vx, precision = ML_Binary32, tag = "px", debug=debugf) if self.precision != ML_Binary32 else vx
                py = Conversion(scaled_vy, precision = ML_Binary32, tag = "py", debug=debugf) if self.precision != ML_Binary32 else vy

                init_approx_fp32 = Conversion(DivisionSeed(px, py, precision = ML_Binary32, tag = "seed", debug = debugf), precision = self.precision, tag = "seed_ext", debug = debug_lftolx)
                if not processor.is_supported_operation(init_approx_fp32):
                    Log.report(Log.Error, "The target %s does not implement inverse square root seed" % processor)
                else:
                    init_approx = init_approx_fp32
            else:
                Log.report(Log.Error, "The target %s does not implement inverse square root seed" % processor)
        else:
            init_approx = init_approx_precision

        current_approx_std = init_approx 
        # correctly-rounded inverse computation
        num_iteration = num_iter

        Attributes.unset_default_rounding_mode()
        Attributes.unset_default_silent()


        def compute_div(_init_approx, _vx = None, _vy = None, scale_result = None): 
            inv_iteration_list = []
            Attributes.set_default_rounding_mode(ML_RoundToNearest)
            Attributes.set_default_silent(True)
            _current_approx = _init_approx
            for i in range(num_iteration):
                new_iteration = NR_Iteration(_current_approx, _vy, force_fma = False if (i != num_iteration - 1) else True)
                inv_iteration_list.append(new_iteration)
                _current_approx = new_iteration.get_new_approx()
                _current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_lftolx)


            def dividend_mult(div_approx, inv_approx, dividend, divisor, index, force_fma = False):
                #yerr = dividend - div_approx * divisor
                yerr = FMSN(div_approx, divisor, dividend)
                yerr.set_attributes(tag = "yerr%d" % index, debug = debug_lftolx)
                #new_div = div_approx + yerr * inv_approx
                new_div = FMA(yerr, inv_approx, div_approx)
                new_div.set_attributes(tag = "new_div%d" % index, debug = debug_lftolx)
                return new_div

            # multiplication correction iteration
            # to get correctly rounded full division
            _current_approx.set_attributes(tag = "final_approx", debug = debug_lftolx)
            current_div_approx = _vx * _current_approx
            num_dividend_mult_iteration = 1
            for i in range(num_dividend_mult_iteration):
                current_div_approx = dividend_mult(current_div_approx, _current_approx, _vx, _vy, i)


            # last iteration
            yerr_last = FMSN(current_div_approx, _vy, _vx) #, clearprevious = True)
            Attributes.unset_default_rounding_mode()
            Attributes.unset_default_silent()
            last_div_approx = FMA(yerr_last, _current_approx, current_div_approx, rounding_mode = ML_GlobalRoundMode)

            yerr_last.set_attributes(tag = "yerr_last", debug = debug_lftolx)

            pre_result = last_div_approx
            pre_result.set_attributes(tag = "unscaled_div_result", debug = debug_lftolx)
            if scale_result != None:
                #result = pre_result * ExponentInsertion(ex) * ExponentInsertion(-ey)
                scale_factor_0 = Max(Min(scale_result, 950), -950, tag = "scale_factor_0", debug = debugd)
                scale_factor_1 = Max(Min(scale_result - scale_factor_0, 950), -950, tag = "scale_factor_1", debug = debugd)
                scale_factor_2 = scale_result - (scale_factor_1 + scale_factor_0)
                scale_factor_2.set_attributes(debug = debugd, tag = "scale_factor_2")
                
                result = ((pre_result * ExponentInsertion(scale_factor_0)) * ExponentInsertion(scale_factor_1)) * ExponentInsertion(scale_factor_2)
            else:
                result = pre_result
            result.set_attributes(tag = "result", debug = debug_lftolx)

            ext_pre_result = FMA(yerr_last, _current_approx, current_div_approx, precision = ML_DoubleDouble, tag = "ext_pre_result", debug = debug_ddtolx)
            subnormal_pre_result = SpecificOperation(ext_pre_result, ex - ey, precision = self.precision, specifier = SpecificOperation.Subnormalize, tag = "subnormal_pre_result", debug = debug_lftolx)
            sub_scale_factor = ex - ey
            sub_scale_factor_0 = Max(Min(sub_scale_factor, 950), -950, tag = "sub_scale_factor_0", debug = debugd)
            sub_scale_factor_1 = Max(Min(sub_scale_factor - sub_scale_factor_0, 950), -950, tag = "sub_scale_factor_1", debug = debugd)
            sub_scale_factor_2 = sub_scale_factor - (sub_scale_factor_1 + sub_scale_factor_0)
            sub_scale_factor_2.set_attributes(debug = debugd, tag = "sub_scale_factor_2")
            #subnormal_result = (subnormal_pre_result * ExponentInsertion(ex, tag ="sr_ex_ei")) * ExponentInsertion(-ey, tag = "sr_ey_ei")
            subnormal_result = (subnormal_pre_result * ExponentInsertion(sub_scale_factor_0)) * ExponentInsertion(sub_scale_factor_1, tag = "sr_ey_ei") * ExponentInsertion(sub_scale_factor_2)
            subnormal_result.set_attributes(debug = debug_lftolx, tag = "subnormal_result")
            return result, subnormal_result, _current_approx, inv_iteration_list


        def bit_match(fp_optree, bit_id, likely = False, **kwords):
            return NotEqual(BitLogicAnd(TypeCast(fp_optree, precision = ML_Int64), 1 << bit_id), 0, likely = likely, **kwords)


        def extract_and_inject_sign(sign_source, sign_dest, int_precision = ML_Int64, fp_precision = self.precision, **kwords):
            int_sign_dest = sign_dest if isinstance(sign_dest.get_precision(), ML_Fixed_Format) else TypeCast(sign_dest, precision = int_precision)
            return TypeCast(BitLogicOr(BitLogicAnd(TypeCast(sign_source, precision = int_precision), 1 << (self.precision.bit_size - 1)), int_sign_dest), precision = fp_precision)


        x_zero = Test(vx, specifier = Test.IsZero, likely = False)
        y_zero = Test(vy, specifier = Test.IsZero, likely = False)

        comp_sign = Test(vx, vy, specifier = Test.CompSign, tag = "comp_sign", debug = debuglx )

        y_nan = Test(vy, specifier = Test.IsNaN, likely = False)

        x_snan = Test(vx, specifier = Test.IsSignalingNaN, likely = False)
        y_snan = Test(vy, specifier = Test.IsSignalingNaN, likely = False)

        x_inf = Test(vx, specifier = Test.IsInfty, likely = False, tag = "x_inf")
        y_inf = Test(vy, specifier = Test.IsInfty, likely = False, tag = "y_inf", debug = debugd)


        scheme = None
        gappa_vx, gappa_vy = None, None
        gappa_init_approx = None
        gappa_current_approx = None

        if isinstance(processor, K1B_Processor):
            print "K1B specific generation"

            gappa_vx = vx
            gappa_vy = vy

            fast_init_approx = DivisionSeed(vx, vy, precision = self.precision, tag = "fast_init_approx", debug = debug_lftolx)
            slow_init_approx = DivisionSeed(scaled_vx, scaled_vy, precision = self.precision, tag = "slow_init_approx", debug = debug_lftolx)

            gappa_init_approx = fast_init_approx

            specific_case           = bit_match(fast_init_approx, 0, tag = "b0_specific_case_bit", debug = debugd)
            y_subnormal_or_zero     = bit_match(fast_init_approx, 1, tag = "b1_y_sub_or_zero", debug = debugd)
            x_subnormal_or_zero     = bit_match(fast_init_approx, 2, tag = "b2_x_sub_or_zero", debug = debugd)
            y_inf_or_nan            = bit_match(fast_init_approx, 3, tag = "b3_y_inf_or_nan", debug = debugd)
            inv_underflow           = bit_match(fast_init_approx, 4, tag = "b4_inv_underflow", debug = debugd)
            x_inf_or_nan            = bit_match(fast_init_approx, 5, tag = "b5_x_inf_or_nan", debug = debugd)
            mult_error_underflow    = bit_match(fast_init_approx, 6, tag = "b6_mult_error_underflow", debug = debugd)
            mult_dividend_underflow = bit_match(fast_init_approx, 7, tag = "b7_mult_dividend_underflow", debug = debugd)
            mult_dividend_overflow  = bit_match(fast_init_approx, 8, tag = "b8_mult_dividend_overflow", debug = debugd)
            direct_result_flag      = bit_match(fast_init_approx, 9, tag = "b9_direct_result_flag", debug = debugd)
            div_overflow            = bit_match(fast_init_approx, 10, tag = "b10_div_overflow", debug = debugd)

            # bit11/eb large = bit_match(fast_init_approx, 11) 
            # bit12 = bit_match(fast_init_approx, 11)

            #slow_result, slow_result_subnormal, _, _ = compute_div(slow_init_approx, scaled_vx, scaled_vy, scale_result = (ExponentInsertion(ex, tag = "eiy_sr"), ExponentInsertion(-ey, tag ="eiy_sr")))
            slow_result, slow_result_subnormal, _, _ = compute_div(slow_init_approx, scaled_vx, scaled_vy, scale_result = ex - ey)
            fast_result, fast_result_subnormal, fast_current_approx, inv_iteration_list = compute_div(fast_init_approx, vx, vy, scale_result = None)
            gappa_current_approx = fast_current_approx

            pre_scheme = ConditionBlock(NotEqual(specific_case, 0, tag = "specific_case", likely = True, debug = debugd),
                Return(fast_result),
                ConditionBlock(Equal(direct_result_flag, 0, tag = "direct_result_case"),
                    Return(fast_init_approx),
                    ConditionBlock(x_subnormal_or_zero | y_subnormal_or_zero | inv_underflow | mult_error_underflow | mult_dividend_overflow | mult_dividend_underflow,
                        ConditionBlock(x_zero | y_zero,
                            Return(fast_init_approx),
                            ConditionBlock(Test(slow_result, specifier = Test.IsSubnormal),
                                Return(slow_result_subnormal),
                                Return(slow_result)
                            ),
                        ),
                        ConditionBlock(x_inf_or_nan,
                            Return(fast_init_approx),
                            ConditionBlock(y_inf_or_nan,
                                Return(fast_init_approx),
                                ConditionBlock(NotEqual(div_overflow, 0, tag = "div_overflow_case"),
                                    Return(RoundedSignedOverflow(fast_init_approx, tag = "signed_inf")),
                                    #Return(extract_and_inject_sign(fast_init_approx, FP_PlusInfty(self.precision) , tag = "signed_inf")),
                                    Return(FP_SNaN(self.precision))
                                )
                            )
                        )
                    )
                )
            )

            scheme = Statement(fast_result, pre_scheme)

        else:
            print "generic generation"

            x_inf_or_nan = Test(vx, specifier = Test.IsInfOrNaN, likely = False)
            y_inf_or_nan = Test(vy, specifier = Test.IsInfOrNaN, likely = False, tag = "y_inf_or_nan", debug = debugd)

            result, subnormal_result, gappa_current_approx, inv_iteration_list = compute_div(current_approx_std, scaled_vx, scaled_vy, scale_result = (ExponentInsertion(ex), ExponentInsertion(-ey)))
            gappa_vx = scaled_vx
            gappa_vy = scaled_vy
            gappa_init_approx = init_approx

            # x inf and y inf 
            pre_scheme = ConditionBlock(x_inf_or_nan, 
                ConditionBlock(x_inf,
                    ConditionBlock(y_inf_or_nan, 
                        Statement(
                            ConditionBlock(y_snan, Raise(ML_FPE_Invalid)),
                            Return(FP_QNaN(self.precision)),
                        ),
                        ConditionBlock(comp_sign, Return(FP_MinusInfty(self.precision)), Return(FP_PlusInfty(self.precision)))
                    ),
                    Statement(
                        ConditionBlock(x_snan, Raise(ML_FPE_Invalid)),
                        Return(FP_QNaN(self.precision))
                    )
                ),
                ConditionBlock(x_zero,
                    ConditionBlock(y_zero | y_nan,
                        Statement(
                            ConditionBlock(y_snan, Raise(ML_FPE_Invalid)),
                            Return(FP_QNaN(self.precision))
                        ),
                        Return(vx)
                    ),
                    ConditionBlock(y_inf_or_nan,
                        ConditionBlock(y_inf,
                            Return(Select(comp_sign, FP_MinusZero(self.precision), FP_PlusZero(self.precision))),
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
                            ConditionBlock(Test(result, specifier = Test.IsSubnormal, likely = False),
                                Statement(
                                    ConditionBlock(Comparison(yerr_last, 0, specifier = Comparison.NotEqual, likely = True),
                                        Statement(Raise(ML_FPE_Inexact, ML_FPE_Underflow))
                                    ),
                                    Return(subnormal_result),
                                ),
                                Statement(
                                    ConditionBlock(Comparison(yerr_last, 0, specifier = Comparison.NotEqual, likely = True),
                                        Raise(ML_FPE_Inexact)
                                    ),
                                    Return(result)
                                )
                            )
                        )
                    )
                )
            )
            rnd_mode = GetRndMode()
            scheme = Statement(rnd_mode, SetRndMode(ML_RoundToNearest), yerr_last, SetRndMode(rnd_mode), pre_result, ClearException(), result, pre_scheme)




        opt_eng = OptimizationEngine(processor)

        # fusing FMA
        if fuse_fma:
            print "MDL fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)


        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)


        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        #print "silencing operation"
        #opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(scheme)

        #print scheme.get_str(depth = None, display_precision = True)

        # check processor support
        print "checking processor support"
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        #opt_eng.factorize_fast_path(scheme)

        print "Gappa script generation"
        
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        self.result.add_header("support_lib/ml_special_values.h")

        output_stream = open(output_file, "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()
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
            print "eval_error: ", eval_error
        except:
            print "error during gappa run"



if __name__ == "__main__":
    # auto-test
    num_iter        = int(extract_option_value("--num-iter", "3"))

    arg_template = ML_ArgTemplate(default_function_name = "new_div", default_output_file = "new_div.c" )
    arg_template.sys_arg_extraction()


    ml_div          = ML_Division(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  num_iter                  = num_iter,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
