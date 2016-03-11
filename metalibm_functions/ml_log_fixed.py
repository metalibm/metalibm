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

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value

class ML_Logarithm:
    def __init__(self,
                 precision = ML_Binary64,
                 abs_accuracy = S2**-24,
                 libm_compliant = True,
                 debug_flag = False,
                 fuse_fma = True,
                 fast_path_extract = True,
                 target = GenericProcessor(),
                 output_file = "log_fixed.c",
                 function_name = "log_fixed"):
        # declaring CodeFunction and retrieving input variable
        self.function_name = function_name
        self.precision = precision
        self.processor = target
        func_implementation = CodeFunction(self.function_name, output_format = self.precision)
        vx = func_implementation.add_input_variable("x", self.precision) 

        sollya_precision = self.precision.sollya_object

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


        vx_exp = RawSignExpExtraction(vx, tag = "vx_exp", precision = ML_Int32, debug = debugd)
        vx_exp_u = Conversion(vx_exp, precision = ML_UInt32)
        vx_exp_u.set_precision(ML_UInt32)
        tt = CountLeadingZeros(vx_exp_u)
        tt_u = Conversion(tt, precision = ML_UInt32)
        t = tt_u + vx_exp_u;
        scheme = Statement(Return(t))

        #print scheme.get_str(depth = None, display_precision = True)

        opt_eng = OptimizationEngine(self.processor)

        # fusing FMA
        if fuse_fma:
            print "MDL fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        #print scheme.get_str(depth = None, display_precision = True)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)

        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        print "silencing operation"
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        func_implementation.set_scheme(scheme)

        # check processor support
        opt_eng.check_processor_support(scheme)

        #print scheme.get_str(depth = None, display_precision = True)

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
    arg_template = ML_ArgTemplate(default_function_name = "log_fixed", default_output_file = "log_fixed.c")
    arg_template.sys_arg_extraction()

    ml_log = ML_Logarithm(precision          = arg_template.precision,
                          libm_compliant     = arg_template.libm_compliant,
                          debug_flag         = arg_template.debug_flag,
                          target             = arg_template.target,
                          fuse_fma           = arg_template.fuse_fma,
                          fast_path_extract  = arg_template.fast_path,
                          function_name      = arg_template.function_name,
                          output_file        = arg_template.output_file)
