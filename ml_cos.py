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
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_Cosine:
    def __init__(self, 
                 precision = ML_Binary32, 
                 accuracy  = ML_Faithful,
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "cosf.c", 
                 function_name = "cosf"):

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
        if debug_flag: 
            Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
            kwords["arg_value"] = vx
            kwords["function_name"] = self.function_name
            return RaiseReturn(*args, **kwords)


        test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
        test_nan        = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
        test_positive   = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

        test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
        return_snan        = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

        # return in case of infinity input
        infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(self.precision)), Return(FP_PlusZero(self.precision))))
        # return in case of specific value input (NaN or inf)
        specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(self.precision))), infty_return)
        # return in case of standard (non-special) input

        sollya_precision = precision.get_sollya_object()

        # argument reduction
        frac_pi_index = 2
        frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, RN)
        inv_frac_pi = round(pi / S2**frac_pi_index, sollya_precision, RN)
        # computing k = E(x * frac_pi)
        k = NearestInteger(vx * frac_pi)

        inv_frac_pi_cst = Constant(inv_frac_pi)

        red_x = vx - inv_frac_pi_cst * k 

        approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

        error_goal_approx = S2**-24


        Log.report(Log.Info, "\033[33;1m building mathematical polynomial \033[0m\n")
        poly_degree = int(sup(guessdegree(cos(x), approx_interval, error_goal_approx)))
        Log.report(Log.Info, "poly degree is %d\n" % poly_degree)


        error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

        polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
        #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme


        print poly_degree.__class__

        format_list = [self.precision] * (poly_degree + 1)

        poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(cos(x), poly_degree, format_list, approx_interval, absolute, error_function = error_function)

        Log.report(Log.Info, "fpminimax polynomial is %s " % poly_object.get_sollya_object())



        poly = polynomial_scheme_builder(poly_object, red_x, unified_precision = self.precision)

        result = Return(poly)


        # main scheme
        Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
        scheme = Statement(result)

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
        #display(decimal)
        self.result.add_header_comment("polynomial degree  for  exp(x): %d" % poly_degree)
        self.result.add_header_comment("sollya polynomial  for  exp(x): %s" % poly_object.get_sollya_object())
        self.result.add_header_comment("polynomial approximation error: %s" % poly_approx_error)
        self.result.add_header_comment("polynomial evaluation    error: %s" % poly_eval_error)
        if debug_flag:
            self.result.add_header("stdio.h")
            self.result.add_header("inttypes.h")
        output_stream = open(output_file, "w")#"%s.c" % exp_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()



if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "new_cos", default_output_file = "new_cos.c" )
    # argument extraction 
    parse_arg_index_list = arg_template.sys_arg_extraction()
    arg_template.check_args(parse_arg_index_list)


    ml_exp          = ML_Cosine(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  accuracy                  = arg_template.accuracy,
                                  output_file               = arg_template.output_file)
