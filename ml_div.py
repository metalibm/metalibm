# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from core.attributes import ML_Debug
from core.ml_operations import *
from core.ml_formats import *
from code_generation.c_code_generator import CCodeGenerator
from code_generation.generic_processor import GenericProcessor
from code_generation.code_object import CodeObject
from code_generation.code_function import CodeFunction
from code_generation.generator_utility import C_Code 
from core.ml_optimization_engine import OptimizationEngine
from core.polynomials import *
from core.ml_table import ML_Table

from kalray_proprietary.k1a_processor import K1A_Processor
from kalray_proprietary.k1b_processor import K1B_Processor
from code_generation.x86_processor import X86_FMA_Processor, X86_SSE_Processor
from code_generation.gappa_code_generator import GappaCodeGenerator

from utility.gappa_utils import execute_gappa_script_extract
from ml_functions.ml_template import ML_ArgTemplate

from utility.common import test_flag_option, extract_option_value  


class ML_Division:
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

        class NR_Iteration: 
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
        debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s)" % v)
        debug_ddtolx  = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v))
        debug_dd      = ML_Debug(display_format = "{.hi=%lf, .lo=%lf}", pre_process = lambda v: "%s.hi, %s.lo" % (v, v))

        ex = Min(ExponentExtraction(vx, tag = "ex", debug = debugd), 1020)
        ey = Min(ExponentExtraction(vy, tag = "ey", debug = debugd), 1020)

        scaling_factor_x = ExponentInsertion(-ex) #ConditionalAllocation(Abs(ex) > 100, -ex, 0)
        scaling_factor_y = ExponentInsertion(-ey) #ConditionalAllocation(Abs(ey) > 100, -ey, 0)

        scaled_vx = vx * scaling_factor_x
        scaled_vy = vy * scaling_factor_y

        scaled_vx.set_attributes(debug = debug_lftolx, tag = "scaled_vx")
        scaled_vy.set_attributes(debug = debug_lftolx, tag = "scaled_vy")

        px = Conversion(scaled_vx, precision = ML_Binary32, tag = "px", debug=debugf) if self.precision != ML_Binary32 else vx
        py = Conversion(scaled_vy, precision = ML_Binary32, tag = "py", debug=debugf) if self.precision != ML_Binary32 else vy

        pre_init_approx = DivisionSeed(px, py, precision = ML_Binary32, tag = "seed", debug = debugf)  
        init_approx = Conversion(pre_init_approx, precision = self.precision, tag = "seedd", debug = debug_lftolx) if self.precision != ML_Binary32 else pre_init_approx

        current_approx = init_approx 
        # correctly-rounded inverse computation
        num_iteration = num_iter
        inv_iteration_list = []

        Attributes.set_default_rounding_mode(ML_RoundToNearest)
        Attributes.set_default_silent(True)

        for i in xrange(num_iteration):
            new_iteration = NR_Iteration(current_approx, scaled_vy, force_fma = False if (i != num_iteration - 1) else True)
            inv_iteration_list.append(new_iteration)
            current_approx = new_iteration.get_new_approx()
            current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_lftolx)


        def dividend_mult(div_approx, inv_approx, dividend, divisor, index, force_fma = False):
            yerr = dividend - div_approx * divisor
            #yerr = FMSN(div_approx, divisor, dividend)
            yerr.set_attributes(tag = "yerr%d" % index, debug = debug_lftolx)
            new_div = div_approx + yerr * inv_approx
            #new_div = FMA(yerr, inv_approx, div_approx)
            new_div.set_attributes(tag = "new_div%d" % index, debug = debug_lftolx)
            return new_div

        # multiplication correction iteration
        # to get correctly rounded full division
        current_approx.set_attributes(tag = "final_approx", debug = debug_lftolx)
        current_div_approx = scaled_vx * current_approx
        num_dividend_mult_iteration = 1
        for i in xrange(num_dividend_mult_iteration):
            current_div_approx = dividend_mult(current_div_approx, current_approx, scaled_vx, scaled_vy, i)


        # last iteration
        yerr_last = FMSN(current_div_approx, scaled_vy, scaled_vx) #, clearprevious = True)
        Attributes.unset_default_rounding_mode()
        Attributes.unset_default_silent()
        last_div_approx = FMA(yerr_last, current_approx, current_div_approx)

        yerr_last.set_attributes(tag = "yerr_last", debug = debug_lftolx)

        pre_result = last_div_approx
        pre_result.set_attributes(tag = "unscaled_div_result", debug = debug_lftolx)
        result = pre_result * ExponentInsertion(ex) * ExponentInsertion(-ey)
        result.set_attributes(tag = "result", debug = debug_lftolx)


        x_inf_or_nan = Test(vx, specifier = Test.IsInfOrNaN, likely = False)
        y_inf_or_nan = Test(vy, specifier = Test.IsInfOrNaN, likely = False, tag = "y_inf_or_nan", debug = debugd)
        comp_sign = Test(vx, vy, specifier = Test.CompSign, tag = "comp_sign", debug = debuglx )
        x_zero = Test(vx, specifier = Test.IsZero, likely = False)
        y_zero = Test(vy, specifier = Test.IsZero, likely = False)

        y_nan = Test(vy, specifier = Test.IsNaN, likely = False)

        x_snan = Test(vx, specifier = Test.IsSignalingNaN, likely = False)
        y_snan = Test(vy, specifier = Test.IsSignalingNaN, likely = False)

        x_inf = Test(vx, specifier = Test.IsInfty, likely = False, tag = "x_inf")
        y_inf = Test(vy, specifier = Test.IsInfty, likely = False, tag = "y_inf", debug = debugd)

        # determining an extended precision 
        ext_precision_map = {
            ML_Binary32: ML_Binary64,
            ML_Binary64: ML_DoubleDouble,
        }
        ext_precision = ext_precision_map[self.precision]

        ext_pre_result = FMA(yerr_last, current_approx, current_div_approx, precision = ext_precision, tag = "ext_pre_result", debug = debug_ddtolx)
        subnormal_result = None
        if isinstance(ext_precision, ML_Compound_FP_Format):
            subnormal_pre_result = SpecificOperation(ext_pre_result, ex - ey, precision = self.precision, specifier = SpecificOperation.Subnormalize, tag = "subnormal_pre_result", debug = debug_lftolx)
            subnormal_result = (subnormal_pre_result * ExponentInsertion(ex)) * ExponentInsertion(-ey)
        else:
            subnormal_result = Conversion(ext_pre_result * ExponentInsertion(ex - ey, tag = "final_scaling_factor", precision = ext_precision), precision = self.precision)


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


        processor = target

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
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        #opt_eng.factorize_fast_path(scheme)
        
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
            init_approx.get_handle().get_node(): seed_var,
            scaled_vx.get_handle().get_node(): Variable("x", precision = self.precision, interval = Interval(1, 2)),
            scaled_vy.get_handle().get_node(): Variable("y", precision = self.precision, interval = Interval(1, 2)),
        }
        G1 = Constant(1, precision = ML_Exact)
        exact = G1 / scaled_vy
        exact.set_precision(ML_Exact)
        exact.set_tag("div_exact")
        gappa_goal = current_approx.get_handle().get_node() - exact
        gappa_goal.set_precision(ML_Exact)
        gappacg = GappaCodeGenerator(target, declare_cst = False, disable_debug = True)
        gappa_code = gappacg.get_interval_code(gappa_goal, cg_eval_error_copy_map)

        new_exact_node = exact.get_handle().get_node()

        for nr in inv_iteration_list:
            nr.get_hint_rules(gappacg, gappa_code, new_exact_node)

        seed_wrt_exact = seed_var - new_exact_node
        seed_wrt_exact.set_precision(ML_Exact)
        gappacg.add_hypothesis(gappa_code, seed_wrt_exact, Interval(-S2**-7, S2**-7))

        eval_error = execute_gappa_script_extract(gappa_code.get(gappacg))["goal"]
        print "eval_error: ", eval_error



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
