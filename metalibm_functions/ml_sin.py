# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

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
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.common import test_flag_option, extract_option_value  

class ML_Sine:
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "sinf.c", 
                 function_name = "sinf"):
        # declaring CodeFunction and retrieving input variable
        self.function_name  = function_name
        self.precision      = precision
        self.processor      = target
        func_implementation = CodeFunction(self.function_name, output_format = self.precision)
        vx = func_implementation.add_input_variable("x", self.precision) 

        sollya_precision = self.precision.sollya_object



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

        int_precision = ML_Int64 if self.precision is ML_Binary64 else ML_Int32

        inv_pi_value = 1 / pi 

        # argument reduction
        mod_pi_x = NearestInteger(vx * inv_pi_value)
        red_vx = vx - mod_pi_x * pi


        approx_interval = Interval(0, pi/2)


        poly_degree = sup(guessdegree(sin(x)/x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        global_poly_object = Polynomial.build_from_approximation(sin(x)/x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, absolute)
        poly_object = global_poly_object#.sub_poly(start_index = 1)

        print "generating polynomial evaluation scheme"
        _poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, red_vx, unified_precision = self.precision)
        _poly.set_attributes(tag = "poly", debug = debug_lftolx)
        print global_poly_object.get_sollya_object()

        pre_result = vx * _poly

        result = pre_result
        result.set_attributes(tag = "result", debug = debug_lftolx)

        # main scheme
        print "MDL scheme"
        scheme = Statement(Return(result))

        #print scheme.get_str(depth = None, display_precision = True)

        opt_eng = OptimizationEngine(self.processor)

        # fusing FMA
        print "MDL fusing FMA"
        scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        #print scheme.get_str(depth = None, display_precision = True)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = ML_Binary32)


        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        print "silencing operation"
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        func_implementation.set_scheme(scheme)

        # check processor support
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        opt_eng.factorize_fast_path(scheme)
        #print scheme.get_str(depth = None, display_precision = True)
        
        cg = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = func_implementation.get_definition(cg, C_Code, static_cst = True)
        #print self.result.get(cg)
        output_stream = open("%s.c" % func_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "new_sin", default_output_file = "new_sin.c" )
    arg_template.sys_arg_extraction()


    ml_sin          = ML_Sine(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
