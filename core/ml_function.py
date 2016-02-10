# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          
# last-modified:    Feb  5th, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import *  
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.ml_call_externalizer import CallExternalizer
from metalibm_core.core.ml_vectorizer import StaticVectorizer

from metalibm_core.code_generation.code_object import NestedCode
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import FunctionOperator

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.common import ML_NotImplemented
from metalibm_core.utility.debug_utils import *

## \defgroup ml_function ml_function
## @{


## standardized function name geneation
#  @param base_name string name of the mathematical function
#  @param io_precisions list of output, input formats (outputs followed by inputs)
#  @param in_arity integer number of input arguments
#  @param out_arity integer number of function results
def libc_naming(base_name, io_precisions, in_arity = 1, out_arity = 1):
  precision_map = {
    ML_Binary32: "sf",
    ML_Binary64: "df",
    ML_Int32: "si",
    ML_Int64: "sd",
    ML_UInt32: "ui",
    ML_UInt64: "ud"
  }
  format_list = [io_precisions[0]] * (in_arity + out_arity) if len(io_precisions) == 1 else io_precisions
  format_suffix = ""
  previous_format = None
  counter = 0
  for precision in format_list:
    if precision != previous_format:
      if counter != 0:
        format_suffix += "%s%d" % (precision_map[previous_format], counter)
      counter = 0
      previous_format = precision
    counter += 1
  return base_name + format_suffix
  


## Base class for all metalibm function (metafunction)
class ML_FunctionBasis(object):
  name = "function_basis"

  ## constructor
  #  @param base_name string function name (without precision considerations)
  #  @param function_name 
  #  @param output_file string name of source code output file
  #  @param io_precisions input/output ML_Format list
  #  @param abs_accuracy absolute accuracy
  #  @param libm_compliant boolean flag indicating whether or not the function should be compliant with standard libm specification (wrt exception, error ...)
  #  @param processor GenericProcessor instance, target of the implementation
  #  @param fuse_fma boolean flag indicating whether or not fusing Multiply+Add optimization must be applied
  #  @param fast_path_extract boolean flag indicating whether or not fast path extraction optimization must be applied
  #  @param debug_flag boolean flag, indicating whether or not debug code must be generated 
  def __init__(self,
             # Naming
             base_name = "unknown_function",
             function_name=None,
             output_file = None,
             # Specification
             io_precisions = [ML_Binary32], 
             abs_accuracy = None,
             libm_compliant = True,
             # Optimization parameters
             processor = GenericProcessor(),
             fuse_fma = True, 
             fast_path_extract = True,
             # Debug verbosity
             debug_flag = False,
             vector_size = 1
         ):
    # io_precisions must be a list
    #     -> with a single element
    # XOR -> with as many elements as function arity (input + output arities)
    self.io_precisions = io_precisions

    # Naming logic, using provided information if available, otherwise deriving from base_name
    # base_name is e.g. exp
    # function_name is e.g. expf or expd or whatever 
    self.function_name = function_name if function_name else libc_naming(base_name, self.io_precisions)

    self.output_file = output_file if output_file else self.function_name + ".c"

    self.debug_flag = debug_flag

    self.vector_size = vector_size

    # TODO: FIX which i/o precision to select
    # TODO: incompatible with fixed-point formats
    # self.sollya_precision = self.get_output_precision().get_sollya_object()

    self.abs_accuracy = abs_accuracy if abs_accuracy else S2**(-self.get_output_precision().get_precision())

    self.libm_compliant = libm_compliant
    
    self.processor = processor

    self.fuse_fma = fuse_fma
    self.fast_path_extract = fast_path_extract

    self.implementation = CodeFunction(self.function_name, output_format = self.get_output_precision())
    self.opt_engine = OptimizationEngine(self.processor)
    self.gappa_engine = GappaCodeGenerator(self.processor, declare_cst = True, disable_debug = True)

    self.C_code_generator = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not self.debug_flag, libm_compliant = self.libm_compliant)
    self.main_code_object = NestedCode(self.C_code_generator, static_cst = True)

    self.call_externalizer = CallExternalizer(self.main_code_object)

  def get_vector_size(self):
    return self.vector_size

  ## compute the evaluation error of an ML_Operation node
  #  @param optree ML_Operation object whose evaluation error is computed
  #  @param variable_copy_map dict(optree -> optree) used to delimit the bound of optree
  #  @param goal_precision ML_Format object, precision used for evaluation goal
  #  @param gappa_filename string, name of the file where the gappa proof of the evaluation error will be dumped
  #  @return numerical value of the evaluation error
  def get_eval_error(self, optree, variable_copy_map = {}, goal_precision = ML_Exact, gappa_filename = "gappa_eval_error.g", relative_error = False):
    """ wrapper for GappaCodeGenerator get_eval_error_v2 function """
    copy_map = {}
    for leaf in variable_copy_map: 
      copy_map[leaf] = variable_copy_map[leaf]
    opt_optree = self.optimise_scheme(optree, copy = copy_map, verbose = False)
    new_variable_copy_map = {}
    for leaf in variable_copy_map:
      new_variable_copy_map[leaf.get_handle().get_node()] = variable_copy_map[leaf]
    return self.gappa_engine.get_eval_error_v2(self.opt_engine, opt_optree, new_variable_copy_map if variable_copy_map != None else {}, goal_precision, gappa_filename, relative_error = relative_error)

  ## name generation
  #  @param base_name string, name to be extended for unifiquation
  def uniquify_name(self, base_name):
    """ return a unique identifier, combining base_name + function_name """
    return "%s_%s" % (self.function_name, base_name)

  ## emulation code generation
  def generate_emulate(self):
    raise ML_NotImplemented()


  ## generation the wrapper to the emulation code
  #  @param test_input Variable where the test input is read from
  #  @param mpfr_rnd Variable object used as precision paramater for mpfr calls
  #  @param test_output Variable where emulation result is copied to
  #  @param test_ternary Variable where mpfr ternary status is copied to
  #  @return tuple code_object, code_generator 
  def generate_emulate_wrapper(self, test_input   = Variable("vx", precision = ML_Mpfr_t), mpfr_rnd = Variable("rnd", precision = ML_Int32), test_output = Variable("result", precision = ML_Mpfr_t, var_type = Variable.Local), test_ternary = Variable("ternary", precision = ML_Int32, var_type = Variable.Local)):
    scheme = self.generate_emulate(test_ternary, test_output, test_input, mpfr_rnd)

    wrapper_processor = MPFRProcessor()

    code_generator = CCodeGenerator(wrapper_processor, declare_cst = False, disable_debug = True, libm_compliant = self.libm_compliant)
    code_object = NestedCode(code_generator, static_cst = True)
    code_generator.generate_expr(code_object, scheme, folded = False, initial = False)
    return code_object, code_generator


  def get_output_precision(self):
    return self.io_precisions[0]

  def get_input_precision(self):
    return self.io_precisions[-1]


  def get_sollya_precision(self):
    """ return the main precision use for sollya calls """
    return self.sollya_precision


  def generate_scheme(self):
    """ generate MDL scheme for function implementation """
    Log.report(Log.Error, "generate_scheme must be overloaded by ML_FunctionBasis child")

  ## 
  # @return main_scheme, [list of sub-CodeFunction object]
  def generate_function_list(self):
    self.implementation.set_scheme(self.generate_scheme())
    return [self.implementation]

  ## submit operation node to a standard optimization procedure
  #  @param pre_scheme ML_Operation object to be optimized
  #  @param copy  dict(optree -> optree) copy map to be used while duplicating pre_scheme (if None disable copy)
  #  @param enable_subexpr_sharing boolean flag, enables sub-expression sharing optimization
  #  @param verbose boolean flag, enable verbose mode
  #  @return optimizated scheme 
  def optimise_scheme(self, pre_scheme, copy = None, enable_subexpr_sharing = True, verbose = True):
    """ default scheme optimization """
    # copying when required
    scheme = pre_scheme if copy is None else pre_scheme.copy(copy)
    # fusing FMA
    if self.fuse_fma:
      if verbose: print "MDL fusing FMA"
      scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

    if verbose: print "MDL abstract scheme"
    self.opt_engine.instantiate_abstract_precision(scheme, None)

    if verbose: print "MDL instantiated scheme"
    self.opt_engine.instantiate_precision(scheme, default_precision = None)

    if enable_subexpr_sharing:
      if verbose: print "subexpression sharing"
      self.opt_engine.subexpression_sharing(scheme)

    if verbose: print "silencing operation"
    self.opt_engine.silence_fp_operations(scheme)

    if verbose: print "checking processor support"
    self.opt_engine.check_processor_support(scheme)

    return scheme


  ## 
  #  @return main code object associted with function implementation
  def get_main_code_object(self):
    return self.main_code_object

  ## generate C code for function implenetation 
  #  Code is generated within the main code object
  #  and dumped to a file named after implementation's name
  #  @param code_function_list list of CodeFunction to be generated (as sub-function )
  #  @return void
  def generate_C(self, code_function_list):
    """ Final C generation, once the evaluation scheme has been optimized"""
    # registering scheme as function implementation
    #self.implementation.set_scheme(scheme)
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object
    for code_function in code_function_list:
      self.result = code_function.add_definition(self.C_code_generator, C_Code, code_object, static_cst = True)

    # adding headers
    self.result.add_header("support_lib/ml_special_values.h")
    self.result.add_header("math.h")
    self.result.add_header("stdio.h")
    self.result.add_header("inttypes.h")

    Log.report(Log.Info, "Generating C code in " + self.implementation.get_name() + ".c")
    output_stream = open("%s.c" % self.implementation.get_name(), "w")
    output_stream.write(self.result.get(self.C_code_generator))
    output_stream.close()

  def gen_implementation(self, display_after_gen = False, display_after_opt = False, enable_subexpr_sharing = True):
    # generate scheme
    if self.get_vector_size() == 1:
      # scalar implementation
      code_function_list = self.generate_function_list()
    else:
      # generating scalar scheme
      code_function_list = self.generate_function_list()
      scalar_scheme = self.implementation.get_scheme()
      scalar_arg_list = self.implementation.get_arg_list()
      self.implementation.clear_arg_list()

      code_function_list = self.generate_vector_implementation(scalar_scheme, scalar_arg_list, self.get_vector_size())
      

    for code_function in code_function_list:
      scheme = code_function.get_scheme()
      if display_after_gen:
        print "function %s, after gen " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

      # optimize scheme
      opt_scheme = self.optimise_scheme(scheme, enable_subexpr_sharing = enable_subexpr_sharing)

      if display_after_opt:
        print "function %s, after opt " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

    # generate C code to implement scheme
    self.generate_C(code_function_list)


  ## externalized an optree: generate a CodeFunction which compute the 
  #  given optree inside a sub-function and returns it as a result
  # @param optree ML_Operation object to be externalized
  # @param arg_list list of ML_Operation objects to be used as arguments
  # @return pair ML_Operation, CodeFunction
  def externalize_call(self, optree, arg_list, tag = "foo", result_format = None, name_factory = None):
    ext_function = self.call_externalizer.externalize_call(optree, arg_list, tag, result_format)
    return ext_function.get_function_object()(*arg_list), ext_function


  def generate_vector_implementation(self, scalar_scheme, scalar_arg_list, vector_size = 2):
    # declaring optimizer
    self.opt_engine.set_boolean_format(ML_Bool)
    self.vectorizer = StaticVectorizer(self.opt_engine)

    callback_name = self.uniquify_name("scalar_callback")

    scalar_callback_function = self.call_externalizer.externalize_call(scalar_scheme, scalar_arg_list, callback_name, self.precision)

    print "[SV] optimizing Scalar scheme"
    scalar_scheme = self.optimise_scheme(scalar_scheme)

    scalar_callback          = scalar_callback_function.get_function_object()

    print "[SV] vectorizing scheme"
    vec_arg_list, vector_scheme, vector_mask = self.vectorizer.vectorize_scheme(scalar_scheme, scalar_arg_list, vector_size, self.call_externalizer, self.get_output_precision())

    vector_output_format = self.vectorizer.vectorize_format(self.precision, vector_size)


    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    vec_res = Variable("vec_res", precision = vector_output_format, var_type = Variable.Local)

    vec_elt_arg_tuple = tuple(VectorElementSelection(vec_arg, vi, precision = self.precision) for vec_arg in vec_arg_list)

    vector_mask.set_attributes(tag = "vector_mask", debug = debug_multi)

    print "[SV] building vectorized main statement"
    function_scheme = Statement(
      vector_scheme,
      ConditionBlock(
        Test(vector_mask, specifier = Test.IsMaskNotAnyZero, precision = ML_Bool, likely = True, debug = debug_multi),
        Return(vector_scheme),
        Statement(
          ReferenceAssign(vec_res, vector_scheme),
          Loop(
            ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
            vi < Constant(vector_size, precision = ML_Int32),
            Statement(
              ConditionBlock(
                Likely(VectorElementSelection(vector_mask, vi, precision = ML_Bool), None),
                ReferenceAssign(VectorElementSelection(vec_res, vi, precision = self.precision), scalar_callback(*vec_elt_arg_tuple))
              ),
              ReferenceAssign(vi, vi + 1)
            ),
          ),
          Return(vec_res)
        )
      )
    )

    print "vectorized_scheme: ", function_scheme.get_str(depth = None, display_precision = True, memoization_map = {})

    for vec_arg in vec_arg_list:
      self.implementation.register_new_input_variable(vec_arg)
    self.implementation.set_output_format(vector_output_format)

    # dummy scheme to make functionnal code generation
    self.implementation.set_scheme(function_scheme)

    print "[SV] end of generate_function_list"
    return [scalar_callback_function, self.implementation]


  # Currently mostly empty, to be populated someday
  def gen_emulation_code(self, precode, code, postcode):
    """generate C code that emulates the function, typically using MPFR.
    precode is declaration code (before the test loop)
    postcode is clean-up code (after the test loop)
    Takes the input and output names from input_list and output_list.
    Must postfix output names with "ref_", "ref_ru_", "ref_rd_"

    This class method performs commonly used initializations. 
    It initializes the MPFR versions of the inputs and outputs, 
    with the same names prefixed with "mp" and possibly postfixed with "rd" and "ru".

    It should be overloaded by actual metafunctions, and called by the overloading function. 
    """

  @staticmethod
  def get_name():
    return ML_FunctionBasis.function_name


        
def ML_Function(name):
  new_class = type(name, (ML_FunctionBasis,), {"function_name": name})
  new_class.get_name = staticmethod(lambda: name) 
  return new_class

# end of Doxygen's ml_function group
## @}

