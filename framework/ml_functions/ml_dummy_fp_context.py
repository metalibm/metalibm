# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from core.attributes import ML_Debug, Attributes
from core.ml_operations import *
from core.ml_formats import *
from code_generation.c_code_generator import CCodeGenerator
from code_generation.gappa_code_generator import GappaCodeGenerator
from code_generation.generic_processor import GenericProcessor
from code_generation.code_object import CodeObject, GappaCodeObject, CodeFunction
from code_generation.generator_utility import C_Code, Gappa_Code 
from core.ml_optimization_engine import OptimizationEngine
from core.polynomials import *
from core.ml_table import ML_Table

from utility.gappa_utils import execute_gappa_script_extract

from kalray_proprietary.k1a_processor import K1A_Processor
from code_generation.x86_processor import X86_SSE_Processor

from ml_functions.ml_template import ML_ArgTemplate


class ML_Dummy:
    def __init__(self, 
            precision         = ML_Binary32, 
            libm_compliant    = False, 
            debug_flag        = False, 
            target            = GenericProcessor(), 
            fuse_fma          = True,
            fast_path_extract = True,
            function_name     = "dummy", 
            output_file       = "dummy.c"):
        # init attributes
        self.precision = precision

        # declaring CodeFunction and retrieving input variable
        func_implementation = CodeFunction(function_name, output_format = ML_Binary32)
        vx = func_implementation.add_input_variable("x", self.precision) 
        vy = func_implementation.add_input_variable("y", self.precision) 
        vx.set_interval(Interval(-0.125, 0.125))
        vx.set_max_abs_error(S2**-10)

        Attributes.set_default_rounding_mode(ML_RoundToNearest)
        Attributes.set_default_silent(True)
        sub_scheme = vx + vy * (1 + vx * vy)
        Attributes.unset_default_silent()
        Attributes.unset_default_rounding_mode()

        #scheme = Statement(SaveFPContext(), SetRndMode(ML_RoundToNearest), sub_scheme, RestoreFPContext(), Return(sub_scheme))
        #scheme = Statement(SaveFPContext(), SetRndMode(ML_RoundToNearest), sub_scheme, RestoreFPContext(), Return(sub_scheme), )
        rnd_mode = GetRndMode(tag = "rnd_mode")
        scheme = Statement(rnd_mode, sub_scheme, SetRndMode(rnd_mode), Return(sub_scheme))

        print "scheme"
        print scheme.get_str(depth = None, display_precision = True)

        print "abstractinterval: ", scheme.get_interval()

        processor = target

        opt_eng = OptimizationEngine(processor)

        abstract_scheme = scheme.copy({})

        # fusing FMA
        if fuse_fma:
            print "fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)


        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        #print scheme.get_str(depth = None, display_precision = True)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = ML_Binary32)

        #print scheme.get_str(depth = None, display_precision = True)

        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)


        # check processor support
        print "checking processor support"
        opt_eng.check_processor_support(scheme)

        print "pre-factorize scheme"
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

        # factorizing fast path
        #if fast_path_extract:
        #    opt_eng.factorize_fast_path(scheme)

        print "post-factorize scheme"
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

        #print "abstract"
        #print abstract_scheme.get_str(depth = None, display_precision = True)
        #print "instantiated scheme"
        #print scheme.get_str(depth = None, display_precision = True, memoization_map = {})


        tag_map = {}
        opt_eng.register_nodes_by_tag(scheme, tag_map)
        print "tag_map: ", tag_map

        # registering scheme as function implementation
        func_implementation.set_scheme(scheme)
        
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = func_implementation.get_definition(cg, C_Code, static_cst = True)
        output_stream = open(output_file, "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()

        #print "instantiating gappa code generator"
        #gappacg = GappaCodeGenerator(processor, declare_cst = True, disable_debug = True)
        #eval_error = gappacg.get_eval_error_v2(opt_eng, poly.get_handle().get_node(), {vx: Variable("x", precision = ML_Binary32, interval = vx.get_interval(), max_abs_error = vx.get_max_abs_error())})
        #print "eval_error: ", eval_error




if __name__ == "__main__":
    # auto-test
    #libm_compliant = True if "--libm" in sys.argv else False
    #debug_flag = True if "--debug" in sys.argv else False
    #target = GenericProcessor() if not "--target" in sys.argv else {"k1a": K1A_Processor(), "sse": X86_SSE_Processor()}[sys.argv[sys.argv.index("--target")+1]]
    #ml_dummy = ML_Dummy(precision = ML_Binary32, libm_compliant = libm_compliant, debug_flag = debug_flag, target = target)

    arg_template = ML_ArgTemplate(default_function_name = "dummy", default_output_file = "dummy.c" )
    arg_template.sys_arg_extraction()


    ml_dummy          = ML_Dummy(precision                 = arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
