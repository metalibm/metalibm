# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from core.attributes import ML_Debug
from core.ml_operations import *
from core.ml_formats import *
from code_generation.c_code_generator import CCodeGenerator
from code_generation.generic_processor import GenericProcessor
from code_generation.code_object import CodeObject
from code_generation.code_element import CodeFunction
from code_generation.generator_utility import C_Code 
from core.ml_optimization_engine import OptimizationEngine
from core.polynomials import *
from core.ml_table import ML_Table

from code_generation.gappa_code_generator import GappaCodeGenerator


from utility.ml_template import ML_ArgTemplate

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
        exp_implementation = CodeFunction(self.function_name, output_format = ML_Binary32)
        vx = exp_implementation.add_input_variable("x", ML_Binary32) 


        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
            kwords["arg_value"] = vx
            kwords["function_name"] = self.function_name
            return RaiseReturn(*args, **kwords)


        test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
        test_nan = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
        test_positive = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

        test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
        return_snan = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(ML_Binary32)))

        # return in case of infinity input
        infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(ML_Binary32)), Return(FP_PlusZero(ML_Binary32))))
        # return in case of specific value input (NaN or inf)
        specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(ML_Binary32))), infty_return)
        # return in case of standard (non-special) input

        # exclusion of early overflow and underflow cases
        early_overflow_test = Comparison(vx, 89, likely = False, specifier = Comparison.Greater)
        early_overflow_return = Statement(ClearException(), ExpRaiseReturn(ML_FPE_Inexact, ML_FPE_Overflow, return_value = FP_PlusInfty(ML_Binary32)))
        early_underflow_test = Comparison(vx, -104, likely = False, specifier = Comparison.Less)
        early_underflow_return = Statement(ClearException(), ExpRaiseReturn(ML_FPE_Inexact, ML_FPE_Underflow, return_value = FP_PlusZero(ML_Binary32)))

        # constant computation
        invlog2 = round(1/log(2), binary32, RN)
        invlog2_cst = Constant(invlog2, precision = ML_Binary32)
        log2_hi = round(log(2), 16, RN) 
        log2_lo = round(log(2) - log2_hi, binary32, RN)

        # argument reduction
        unround_k = vx * invlog2
        unround_k.set_attributes(tag = "unround_k", debug = ML_Debug(display_format = "%f"))
        k = NearestInteger(unround_k, precision = ML_Binary32, debug = ML_Debug(display_format = "%d"))
        ik = NearestInteger(unround_k, precision = ML_Int32, debug = ML_Debug(display_format = "%d"), tag = "ik")
        ik.set_tag("ik")
        k.set_tag("k")
        r = (vx - k * log2_hi) - k * log2_lo
        r.set_tag("r")
        r.set_attributes(debug = ML_Debug(display_format = "%f"))

        opt_r = opt_eng.optimization_process(r, ML_Binary32, copy = True, fuse_fma = fuse_fma)

        tag_map = {}
        opt_eng.register_nodes_by_tag(opt_r, tag_map)

        cg_eval_error_copy_map = {
            vx: Variable("x", precision = ML_Binary32, interval = Interval(-104, 89)),
            tag_map["k"]: Variable("k", interval = Interval(-100, 100), precision = ML_Binary32)
        }
        try:
            eval_error = gappacg.get_eval_error(opt_r, cg_eval_error_copy_map)
            print "eval error: ", eval_error
        except:
            print "gappa error evaluation failed"

        print "building mathematical polynomial"
        approx_interval = Interval(-log(2)/2, log(2)/2)
        poly_degree = sup(guessdegree(exp(x), approx_interval, S2**-(self.precision.get_field_size()+1))) 
        poly_object = Polynomial.build_from_approximation(exp(x), poly_degree, [ML_Binary32]*(poly_degree+1), approx_interval, absolute)

        print "generating polynomial evaluation scheme"
        poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, r, unified_precision = ML_Binary32)
        poly.set_tag("poly")

        debug_f = ML_Debug(display_format = "%f")


        late_overflow_test = Comparison(ik, 127, specifier = Comparison.Greater, likely = False, debug = True, tag = "late_overflow_test")
        diff_k = ik - 120
        diff_k.set_attributes(debug = ML_Debug(display_format = "%d"), tag = "diff_k")
        late_overflow_result = (ExponentInsertion(diff_k) * poly) * ExponentInsertion(120)
        late_overflow_result.set_attributes(silent = False, tag = "late_overflow_result", debug = debug_f)
        late_overflow_return = ConditionBlock(Test(late_overflow_result, specifier = Test.IsInfty, likely = False), ExpRaiseReturn(ML_FPE_Overflow, return_value = FP_PlusInfty(ML_Binary32)), Return(late_overflow_result))

        late_underflow_test = Comparison(k, -126, specifier = Comparison.LessOrEqual, likely = False)
        late_underflow_result = (ExponentInsertion(ik + 50) * poly) * ExponentInsertion(-50)
        late_underflow_result.set_attributes(debug = ML_Debug(display_format = "%f"), tag = "late_underflow_result", silent = False)
        test_subnormal = Test(late_underflow_result, specifier = Test.IsSubnormal)
        late_underflow_return = Statement(ConditionBlock(test_subnormal, ExpRaiseReturn(ML_FPE_Underflow, return_value = late_underflow_result)), Return(late_underflow_result))

        std_result = poly * ExponentInsertion(k, debug = ML_Debug(display_format = "%x"))
        std_result.set_debug(ML_Debug(display_format = "%f"))
        result_scheme = ConditionBlock(late_overflow_test, late_overflow_return, ConditionBlock(late_underflow_test, late_underflow_return, Return(std_result)))
        std_return = ConditionBlock(early_overflow_test, early_overflow_return, ConditionBlock(early_underflow_test, early_underflow_return, result_scheme))

        # main scheme
        print "MDL scheme"
        scheme = ConditionBlock(test_nan_or_inf, Statement(ClearException(), specific_return), std_return)

        #print scheme.get_str(depth = None, display_precision = True)

        # fusing FMA
        if fuse_fma: 
            print "MDL fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = ML_Binary32)


        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        print "silencing operation"
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(scheme)

        # check processor support
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        if fast_path_extract:
            opt_eng.factorize_fast_path(scheme)
        
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        self.result.add_header("support_lib/ml_types.h")
        self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        output_stream = open(output_file, "w")#"%s.c" % exp_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "new_div", default_output_file = "new_div.c" )
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
