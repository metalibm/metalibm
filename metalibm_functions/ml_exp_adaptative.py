# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.generator_utility import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator


from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp



class ML_Exponential:
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "expf.c", 
                 function_name = "expf"):

        # declaring target and instantiating optimization engine
        processor = target
        self.precision = precision
        opt_eng = OptimizationEngine(processor)
        gappacg = GappaCodeGenerator(processor, declare_cst = True, disable_debug = True)

        # declaring CodeFunction and retrieving input variable
        self.function_name = function_name
        exp_implementation = CodeFunction(self.function_name, output_format = self.precision)
        vx = exp_implementation.add_input_variable("x", self.precision) 


        Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")

        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
            kwords["arg_value"] = vx
            kwords["function_name"] = self.function_name
            return RaiseReturn(*args, **kwords)


        test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
        test_nan = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
        test_positive = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

        test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
        return_snan = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

        # return in case of infinity input
        infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(self.precision)), Return(FP_PlusZero(self.precision))))
        # return in case of specific value input (NaN or inf)
        specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(self.precision))), infty_return)
        # return in case of standard (non-special) input

        # exclusion of early overflow and underflow cases
        precision_emax      = self.precision.get_emax()
        precision_max_value = S2 * S2**precision_emax 
        exp_overflow_bound  = ceil(log(precision_max_value))
        early_overflow_test = Comparison(vx, exp_overflow_bound, likely = False, specifier = Comparison.Greater)
        early_overflow_return = Statement(ClearException(), ExpRaiseReturn(ML_FPE_Inexact, ML_FPE_Overflow, return_value = FP_PlusInfty(self.precision)))

        precision_emin = self.precision.get_emin_subnormal()
        precision_min_value = S2 ** precision_emin
        exp_underflow_bound = floor(log(precision_min_value))


        early_underflow_test = Comparison(vx, exp_underflow_bound, likely = False, specifier = Comparison.Less)
        early_underflow_return = Statement(ClearException(), ExpRaiseReturn(ML_FPE_Inexact, ML_FPE_Underflow, return_value = FP_PlusZero(self.precision)))


        sollya_prec_map = {ML_Binary32: binary32, ML_Binary64: binary64}


        # constant computation
        invlog2 = round(1/log(2), sollya_prec_map[self.precision], RN)

        interval_vx = Interval(exp_underflow_bound, exp_overflow_bound)
        interval_fk = interval_vx * invlog2
        interval_k = Interval(floor(inf(interval_fk)), ceil(sup(interval_fk)))


        log2_hi_precision = self.precision.get_field_size() - (ceil(log2(sup(abs(interval_k)))) + 2)
        Log.report(Log.Info, "log2_hi_precision: "), log2_hi_precision
        invlog2_cst = Constant(invlog2, precision = self.precision)
        log2_hi = round(log(2), log2_hi_precision, RN) 
        log2_lo = round(log(2) - log2_hi, sollya_prec_map[self.precision], RN)

        # argument reduction
        unround_k = vx * invlog2
        unround_k.set_attributes(tag = "unround_k", debug = ML_Debug(display_format = "%f"))
        k = NearestInteger(unround_k, precision = self.precision, debug = ML_Debug(display_format = "%f"))
        ik = NearestInteger(unround_k, precision = ML_Int32, debug = ML_Debug(display_format = "%d"), tag = "ik")
        ik.set_tag("ik")
        k.set_tag("k")
        exact_pre_mul = (k * log2_hi)
        exact_pre_mul.set_attributes(exact= True)
        exact_hi_part = vx - exact_pre_mul
        exact_hi_part.set_attributes(exact = True)
        r =  exact_hi_part - k * log2_lo
        r.set_tag("r")
        r.set_attributes(debug = ML_Debug(display_format = "%f"))

        opt_r = opt_eng.optimization_process(r, self.precision, copy = True, fuse_fma = fuse_fma)

        tag_map = {}
        opt_eng.register_nodes_by_tag(opt_r, tag_map)

        cg_eval_error_copy_map = {
            vx: Variable("x", precision = self.precision, interval = interval_vx),
            tag_map["k"]: Variable("k", interval = interval_k, precision = self.precision)
        }
        #try:
        if 1:
            #eval_error = gappacg.get_eval_error(opt_r, cg_eval_error_copy_map, gappa_filename = "red_arg.g")
            eval_error = gappacg.get_eval_error_v2(opt_eng, opt_r, cg_eval_error_copy_map, gappa_filename = "red_arg.g")
            Log.report(Log.Info, "eval error: %s" % eval_error)
        #except:
        #    Log.report(Log.Info, "gappa error evaluation failed")
        print r.get_str(depth = None, display_precision = True, display_attribute = True)
        print opt_r.get_str(depth = None, display_precision = True, display_attribute = True)

        approx_interval = Interval(-log(2)/2, log(2)/2)

        local_ulp = sup(ulp(exp(approx_interval), self.precision))
        print "ulp: ", local_ulp 
        error_goal = local_ulp #S2**-(self.precision.get_field_size()+1)
        error_goal_approx = S2**-1 * error_goal

        Log.report(Log.Info, "\033[33;1m building mathematical polynomial \033[0m\n")
        poly_degree = sup(guessdegree(exp(x), approx_interval, error_goal_approx)) #- 1
        init_poly_degree = poly_degree

        return


        while 1: 
            Log.report(Log.Info, "attempting poly degree: %d" % poly_degree)
            poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(exp(x), poly_degree, [self.precision]*(poly_degree+1), approx_interval, absolute)

            Log.report(Log.Info, "poly approx error: %s" % poly_approx_error)

            Log.report(Log.Info, "\033[33;1m generating polynomial evaluation scheme \033[0m")
            poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, r, unified_precision = self.precision)
            poly.set_tag("poly")

            # optimizing poly before evaluation error computation
            opt_poly = opt_eng.optimization_process(poly, self.precision)

            #print "poly: ", poly.get_str(depth = None, display_precision = True)
            #print "opt_poly: ", opt_poly.get_str(depth = None, display_precision = True)

            # evaluating error of the polynomial approximation
            r_gappa_var = Variable("r", precision = self.precision, interval = approx_interval)
            poly_error_copy_map = {
                r.get_handle().get_node(): r_gappa_var
            }
            gappacg = GappaCodeGenerator(target, declare_cst = False, disable_debug = True)
            poly_eval_error = gappacg.get_eval_error_v2(opt_eng, poly.get_handle().get_node(), poly_error_copy_map, gappa_filename = "gappa_poly.g")
            Log.report(Log.Info, "poly evaluation error: %s" % poly_eval_error)

            global_poly_error = poly_eval_error + poly_approx_error
            global_rel_poly_error = global_poly_error / exp(approx_interval)
            print "global_poly_error: ", global_poly_error, global_rel_poly_error 
            flag = local_ulp > sup(abs(global_rel_poly_error))
            print "test: ", flag
            if flag: break
            else:
                if poly_degree > init_poly_degree + 5:
                    Log.report(Log.Error, "poly degree search did not converge")
                poly_degree += 1



        late_overflow_test = Comparison(ik, self.precision.get_emax(), specifier = Comparison.Greater, likely = False, debug = True, tag = "late_overflow_test")
        overflow_exp_offset = (self.precision.get_emax() - self.precision.get_field_size() / 2)
        diff_k = ik - overflow_exp_offset 
        diff_k.set_attributes(debug = ML_Debug(display_format = "%d"), tag = "diff_k")
        late_overflow_result = (ExponentInsertion(diff_k) * poly) * ExponentInsertion(overflow_exp_offset)
        late_overflow_result.set_attributes(silent = False, tag = "late_overflow_result", debug = debugf)
        late_overflow_return = ConditionBlock(Test(late_overflow_result, specifier = Test.IsInfty, likely = False), ExpRaiseReturn(ML_FPE_Overflow, return_value = FP_PlusInfty(self.precision)), Return(late_overflow_result))

        late_underflow_test = Comparison(k, self.precision.get_emin_normal(), specifier = Comparison.LessOrEqual, likely = False)
        underflow_exp_offset = 2 * self.precision.get_field_size()
        late_underflow_result = (ExponentInsertion(ik + underflow_exp_offset) * poly) * ExponentInsertion(-underflow_exp_offset)
        late_underflow_result.set_attributes(debug = ML_Debug(display_format = "%e"), tag = "late_underflow_result", silent = False)
        test_subnormal = Test(late_underflow_result, specifier = Test.IsSubnormal)
        late_underflow_return = Statement(ConditionBlock(test_subnormal, ExpRaiseReturn(ML_FPE_Underflow, return_value = late_underflow_result)), Return(late_underflow_result))

        std_result = poly * ExponentInsertion(ik, tag = "exp_ik", debug = debug_lftolx)
        std_result.set_attributes(tag = "std_result", debug = debug_lftolx)
        result_scheme = ConditionBlock(late_overflow_test, late_overflow_return, ConditionBlock(late_underflow_test, late_underflow_return, Return(std_result)))
        std_return = ConditionBlock(early_overflow_test, early_overflow_return, ConditionBlock(early_underflow_test, early_underflow_return, result_scheme))

        # main scheme
        Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
        scheme = ConditionBlock(test_nan_or_inf, Statement(ClearException(), specific_return), std_return)

        #print scheme.get_str(depth = None, display_precision = True)

        # fusing FMA
        if fuse_fma: 
            Log.report(Log.Info, "\033[33;1m MDL fusing FMA \033[0m")
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        Log.report(Log.Info, "\033[33;1m MDL abstract scheme \033[0m")
        opt_eng.instantiate_abstract_precision(scheme, None)

        Log.report(Log.Info, "\033[33;1m MDL instantiated scheme \033[0m")
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)


        Log.report(Log.Info, "\033[33;1m subexpression sharing \033[0m")
        opt_eng.subexpression_sharing(scheme)

        Log.report(Log.Info, "\033[33;1m silencing operation \033[0m")
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(scheme)

        # check processor support
        Log.report(Log.Info, "\033[33;1m checking processor support \033[0m")
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        if fast_path_extract:
            Log.report(Log.Info, "\033[33;1m factorizing fast path\033[0m")
            opt_eng.factorize_fast_path(scheme)
        
        Log.report(Log.Info, "\033[33;1m generating source code \033[0m")
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        #self.result.add_header("support_lib/ml_types.h")
        self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header_comment("polynomial degree for exp(x): %d" % poly_degree)
        self.result.add_header_comment("sollya polynomial for exp(x): %s" % poly_object.get_sollya_object())
        if debug_flag:
            self.result.add_header("stdio.h")
            self.result.add_header("inttypes.h")
        output_stream = open(output_file, "w")#"%s.c" % exp_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()



if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "new_exp", default_output_file = "new_exp.c" )
    # argument extraction 
    arg_template.sys_arg_extraction()


    ml_exp          = ML_Exponential(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
