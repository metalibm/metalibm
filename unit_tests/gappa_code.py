# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
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

from metalibm_core.utility.debug_utils import * 

class ML_UT_GappaCode(ML_Function("ml_ut_gappa_code")):
  def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = MPFRProcessor(), 
                 output_file = "ut_gappa_code.c", 
                 function_name = "ut_gappa_code"):
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_gappa_code",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag
    )

    self.precision = precision


  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", ML_Binary32)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))

    # declaring free Variable y
    vy = Variable("y", precision = ML_Exact)

    # declaring expression with vx variable
    expr = vx * vx - vx * 2
    # declaring second expression with vx variable
    expr2 = vx * vx - vx

    # optimizing expressions (defining every unknown precision as the
    # default one + some optimization as FMA merging if enabled)
    opt_expr = self.optimise_scheme(expr)
    opt_expr2 = self.optimise_scheme(expr2)

    # setting specific tag name for optimized expression (to be extracted 
    # from gappa script )
    opt_expr.set_tag("goal")
    opt_expr2.set_tag("new_goal")
    
    # defining default goal to gappa execution
    gappa_goal = opt_expr 

    # declaring EXACT expression to be used as hint in Gappa's script
    annotation = self.opt_engine.exactify(vy * (1 / vy))

    # the dict var_bound is used to limit the DAG part to be explored when
    # generating the gappa script, each pair (key, value), indicate a node to stop at <key>
    # and a node to replace it with during the generation: <node>,
    # <node> must be a Variable instance with defined interval
    # vx.get_handle().get_node() is used to retrieve the node instanciating the abstract node <vx>
    # after the call to self.optimise_scheme
    var_bound = {
      vx.get_handle().get_node(): Variable("x", precision = ML_Binary32, interval = vx.get_interval())
    } 
    # generating gappa code to determine interval for <opt_expr>
    gappa_code = self.gappa_engine.get_interval_code(opt_expr, var_bound)

    # add a manual hint to the gappa code
    # which state thtat vy * (1 / vy) -> 1 { vy <> 0 };
    self.gappa_engine.add_hint(gappa_code, annotation, Constant(1, precision = ML_Exact), Comparison(vy, Constant(0, precision = ML_Integer), specifier = Comparison.NotEqual, precision = ML_Bool))
    
    # adding the expression <opt_expr2> as an extra goal in the gappa script
    self.gappa_engine.add_goal(gappa_code, opt_expr2)

    # executing gappa on the script generated from <gappa_code>
    # extract the result and store them into <gappa_result>
    # which is a dict indexed by the goals' tag
    gappa_result = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))


    print "eval error: ", gappa_result["goal"], gappa_result["new_goal"]

    # dummy scheme to make functionnal code generation
    scheme = Statement(Return(vx))

    return scheme

if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_ut_loop_operation", default_output_file = "new_ut_loop_operation.c" )
  arg_template.sys_arg_extraction()


  ml_ut_gappa_code = ML_UT_GappaCode(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)

  ml_ut_gappa_code.gen_implementation(display_after_gen = False, display_after_opt = False)

