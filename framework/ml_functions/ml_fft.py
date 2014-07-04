# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from core.attributes import ML_Debug
from core.ml_operations import *
from core.ml_formats import *
from code_generation.c_code_generator import CCodeGenerator
from code_generation.generic_processor import GenericProcessor
from code_generation.code_object import CodeObject, CodeFunction
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


class ML_FFT:
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 nb_stage = 3,
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "radix4.c", 
                 function_name = "butterfly"):

        # declaring CodeFunction and retrieving input variable
        self.precision = precision
        self.function_name = function_name
        exp_implementation = CodeFunction(self.function_name, output_format = precision)


        # input variables
        xr = []
        xi = []
        input_list = []
        for i in xrange(nb_stage+1):
          for j in xrange(4):
            print ">>>>>>>>>>>>> %d" % (i*4 + j)
            xr.append(exp_implementation.add_input_variable("x%dr" % (i*4 + j), self.precision))
            xi.append(exp_implementation.add_input_variable("x%di" % (i*4 + j), self.precision))
            input_list += [xr[(i*4 + j)], xi[(i*4 + j)]]

        for input in input_list: input.set_interval(Interval(0, 1))

        wr = exp_implementation.add_input_variable("wr", precision)
        wi = exp_implementation.add_input_variable("wi", precision)
        wr.set_interval(Interval(-1, 1))
        wi.set_interval(Interval(-1, 1))

        for i in xrange(0, (nb_stage*4), 4):
            b0r = (xr[i] + xr[i+1]) + (xr[i+2] + xr[i+3]) 
            b0i = (xi[i] + xi[i+1]) + (xi[i+2] + xi[i+3])
            b1r = (xr[i] + xr[i+1]) - (xr[i+2] + xr[i+3])
            b1i = (xi[i] - xi[i+1]) - (xi[i+2] - xi[i+3])
            b2r = (xr[i] - xr[i+1]) + (xr[i+2] - xr[i+3])
            b2i = (xi[i] - xi[i+1]) + (xi[i+2] - xi[i+3])
            b3r = (xr[i] - xr[i+1]) - (xr[i+2] - xr[i+3])
            b3i = (xi[i] + xi[i+1]) - (xi[i+2] + xi[i+3])
            print "!!!!!!!! %d" %i 
            xr[i+4] = b0r * wr - b0i * wi
            xi[i+4] = b0i * wr + b0r * wi   
            xr[i+5] = b1r * wr - b1i * wi
            xi[i+5] = b1i * wr + b1r * wi   
            xr[i+6] = b2r * wr - b2i * wi
            xi[i+6] = b2i * wr + b2r * wi   
            xr[i+7] = b3r * wr - b3i * wi
            xi[i+7] = b3i * wr + b3r * wi   

        #for i in xrange(4):
         #   bir.set_tag("b%dr" %i)
          #  bii.set_tag("b%di" %i)
            #xr[i].set_tag("res%dr" %i)
            #xi[i].set_tag("res%di" %i)

        # list to be extended with new operations for error computation
        result = [xr[nb_stage*4], xi[nb_stage*4], xr[nb_stage*4 + 1], xi[nb_stage*4 + 1],xr[nb_stage*4 + 2], xi[nb_stage*4 + 2], xr[nb_stage*4 + 3], xi[nb_stage*4 + 3]] 
        result_tag = ["xr%d" % (nb_stage*4), "xi%d" % (nb_stage*4), "xr%d" % (nb_stage*4 + 1), "xi%d" % (nb_stage*4 + 1), "xr%d" % (nb_stage*4 + 2), "xi%d" % (nb_stage*4 + 2), "xr%d" % (nb_stage*4 + 3), "xi%d" % (nb_stage*4 + 3)] 

        for subresult, subresult_tag in zip(result, result_tag):
          subresult.set_tag(subresult_tag)

        scheme = [Statement((sub_result)) for sub_result in result]


        global_scheme = Statement()
        for sub_scheme in scheme:
            global_scheme.add(sub_scheme)


        processor = target

        opt_eng = OptimizationEngine(processor)

        # fusing FMA
        if fuse_fma:
            print "MDL fusing FMA"
            global_scheme = opt_eng.fuse_multiply_add(global_scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(global_scheme, None)


        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(global_scheme, default_precision = self.precision)


        #print "subexpression sharing"
        #opt_eng.subexpression_sharing(global_scheme)

        #print "silencing operation"
        #opt_eng.silence_fp_operations(global_scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(global_scheme)

        # check processor support
        opt_eng.check_processor_support(global_scheme)

        # factorizing fast path
        opt_eng.factorize_fast_path(global_scheme)
        
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        self.result.add_header("support_lib/ml_special_values.h")

        output_stream = open(output_file, "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()


        eval_error = {}
        for sub_result in result:
            var_copy_map = {}
            for input_var in input_list:
              var_copy_map[input_var] = Variable(input_var.get_tag(), precision = self.precision, interval = Interval(0, 1))
            var_copy_map[wr] = Variable("wr", precision = self.precision, interval = Interval(-1, 1))
            var_copy_map[wi] = Variable("wi", precision = self.precision, interval = Interval(-1, 1))
            gappacg = GappaCodeGenerator(target, declare_cst = False, disable_debug = True)
            eval_error[sub_result.get_tag()] = gappacg.get_eval_error_v2(opt_eng, sub_result.get_handle().get_node(), var_copy_map, gappa_filename = "gappa_%s.g" % sub_result.get_tag())

        for tag in eval_error:
            print "eval_error for %s: " % tag, eval_error[tag]



if __name__ == "__main__":
    # auto-test
    nb_stage        = int(extract_option_value("--nb-stage", "3"))

    arg_template = ML_ArgTemplate()
    arg_template.sys_arg_extraction()


    ml_div          = ML_FFT(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  nb_stage                  = nb_stage,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file)
