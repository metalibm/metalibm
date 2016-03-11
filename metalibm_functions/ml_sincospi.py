# -*- coding: utf-8 -*-

import sys

from pythonsollya import *


from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
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

from metalibm_core.ml_function import ML_Function

class ML_SinCospi(ML_Function):
    def __init__(self, 
                 base_name = "sincospi",
                 name=None,
                 output_file = None,
                 io_precision = ML_Binary32, 
                 libm_compliant = True, 
                 processor = GenericProcessor(), 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 debug_flag = False
             ):
        # declaring CodeFunction and retrieving input variable
        ML_Function.__init__(self,
                             base_name = base_name,
                             name=name,
                             output_file = name,
                             io_precision = io_precision,
                             libm_compliant = libm_compliant,
                             processor=processor,
                             fuse_fma = fuse_fma,
                             fast_path_extract = fast_path_extract,
                             debug_flag = debug_flag)

        # main scheme
        Log.report(Log.Info, "Construction of the initial evaluation scheme")

        vx = self.implementation.add_input_variable("x", self.io_precision) 
        self.evalScheme = Statement(Return(vx))

        self.opt_engine.optimization_process(self.evalScheme, self.io_precision)
        self.generate_C()


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "sincospi", default_output_file = "sincospi.c" )
    arg_template.sys_arg_extraction()
    ml_sincospi     = ML_SinCospi(base_name                 = arg_template.function_name,
                                  io_precision              = arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  processor                 = arg_template.target,  # TODO rename target into processor 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  output_file               = arg_template.output_file
    )
