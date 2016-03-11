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
from code_generation.code_constant import C_Code 
from core.ml_optimization_engine import OptimizationEngine
from core.polynomials import *
from core.ml_table import ML_Table

from kalray_proprietary.k1a_processor import K1A_Processor
from kalray_proprietary.k1b_processor import K1B_Processor
from code_generation.x86_processor import X86_FMA_Processor, X86_SSE_Processor
from code_generation.gappa_code_generator import GappaCodeGenerator

from utility.gappa_utils import execute_gappa_script_extract
from ml_functions.ml_template import ML_ArgTemplate, precision_map

from utility.common import test_flag_option, extract_option_value  

class ML_FastExpi:
    def __init__(self, 
                 input_precision = ML_Int32, 
                 output_precision = ML_Binary32,
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "fast_expi.c", 
                 function_name = "fast_expi"):
        # declaring CodeFunction and retrieving input variable
        self.function_name = function_name
        self.processor = target
        func_implementation = CodeFunction(self.function_name, output_format = output_precision)
        vx = func_implementation.add_input_variable("x", input_precision) 


        # debug utilities
        debugf        = ML_Debug(display_format = "%f")
        debuglf       = ML_Debug(display_format = "%lf")
        debugx        = ML_Debug(display_format = "%x")
        debuglx       = ML_Debug(display_format = "%\"PRIx64\"", )
        debugd        = ML_Debug(display_format = "%d", pre_process = lambda v: "(int) %s" % v)
        debugld        = ML_Debug(display_format = "%ld")
        #debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s)" % v)
        debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\" ev=%x", pre_process = lambda v: "double_to_64b_encoding(%s), __k1_fpu_get_exceptions()" % v)
        debug_ddtolx  = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v))
        debug_dd      = ML_Debug(display_format = "{.hi=%lf, .lo=%lf}", pre_process = lambda v: "%s.hi, %s.lo" % (v, v))

        bias = 2**(output_precision.get_exponent_size() - 1) - 1
        bound_exp = Max(vx + bias, 2**output_precision.get_exponent_size() - 1) 
        scheme = ExponentInsertion(bound_exp, specifier = ExponentInsertion.NoOffset)

        #print scheme.get_str(depth = None, display_precision = True)

        opt_eng = OptimizationEngine(self.processor)

        # fusing FMA
        print "MDL fusing FMA"
        scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        #print scheme.get_str(depth = None, display_precision = True)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = output_precision)


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
        self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        #print self.result.get(cg)
        output_stream = open("%s.c" % func_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "fast_expi", default_output_file = "fast_expi.c" )
    arg_template.sys_arg_extraction()

    input_precision = precision_map[extract_option_value("--input_precision", "int32")]
    output_precision = precision_map[extract_option_value("--output_precision", "binary32")]


    ml_fast_exp_i          = ML_FastExpi(
                                  input_precision           = input_precision,
                                  output_precision          = output_precision,
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
