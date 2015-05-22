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
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  

from metalibm_core.ml_function import ML_Function

class ML_SinCospi(ML_Function):
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 processor = GenericProcessor(), 
                 output_file = "sincospif.c", 
                 basename = "sincospi"):
        # declaring CodeFunction and retrieving input variable

        ML_Function.__init__(self,
                             basename=basename,
                             precision=precision,
                             processor=processor)

        #print scheme.get_str(depth = None, display_precision = True)
        vx = self.implementation.add_input_variable("x", self.precision) 
        vy = self.implementation.add_input_variable("y", self.precision) 


        # main scheme
        Log.report(Log.Info, "Construction of the initial MDL scheme")
        scheme = Statement(Return(vx))

        # fusing FMA
        if fuse_fma: 
            Log.report(Log.Info, "Fusing FMAs")
            scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

        Log.report(Log.Info, "Infering types")
        self.opt_engine.instantiate_abstract_precision(scheme, None)


        Log.report(Log.Info, "Instantiating precisions")
        self.opt_engine.instantiate_precision(scheme, default_precision = self.precision)


        Log.report(Log.Info, "Subexpression sharing")
        self.opt_engine.subexpression_sharing(scheme)

        Log.report(Log.Info, "Silencing exceptions in internal operations")
        self.opt_engine.silence_fp_operations(scheme)

        # registering scheme as function implementation
        self.implementation.set_scheme(scheme)

        # check processor support
        Log.report(Log.Info, "Checking processor support")
        self.opt_engine.check_processor_support(scheme)

        # factorizing fast path
        if fast_path_extract:
            Log.report(Log.Info, "Factorizing fast path")
            self.opt_engine.factorize_fast_path(scheme)


        
        cg = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = self.implementation.get_definition(cg, C_Code, static_cst = True)
        #self.result.add_header("support_lib/ml_special_values.h")
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        #print self.result.get(cg)
        output_stream = open("%s.c" % self.implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "sincospi", default_output_file = "sincospi.c" )
    arg_template.sys_arg_extraction()


    ml_log          = ML_SinCospi(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  processor                 = arg_template.target,  # TODO rename target into processor 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  basename                  = arg_template.function_name,
                                  output_file               = arg_template.output_file)
