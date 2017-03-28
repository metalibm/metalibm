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

from sollya import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import *  
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.ml_call_externalizer import CallExternalizer
from metalibm_core.core.ml_vectorizer import StaticVectorizer

from metalibm_core.code_generation.code_object import NestedCode
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.core.passes import Pass

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.ml_template import ArgDefault

import random
import subprocess

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
  
## default argument template to be used when no specific value
#  are given for a specific parameter
class DefaultArgTemplate:
  base_name = "unknown_function"
  function_name = None
  output_file = None
  # Specification
  precision = ML_Binary32
  io_precisions = [ML_Binary32]
  abs_accuracy = None
  libm_compliant = True
  # Optimization parameters
  target = GenericProcessor()
  fuse_fma = True
  fast_path_extract = True
  # Debug verbosity
  debug = False
  vector_size = 1
  language = C_Code
  auto_test = False
  auto_test_execute = False
  auto_test_range = Interval(0, 1)
  auto_test_std   = False
  # list of pre-code generation opt passe names (string tag)
  pre_gen_passes = []

  def __init__(self, **kw):
    for key in kw:
      setattr(self, key, kw[key])

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
             base_name = ArgDefault("unknown_function", 2),
             function_name= ArgDefault(None, 2),
             output_file = ArgDefault(None, 2),
             # Specification
             io_precisions = ArgDefault([ML_Binary32], 2), 
             abs_accuracy = ArgDefault(None, 2),
             libm_compliant = ArgDefault(True, 2),
             # Optimization parameters
             processor = ArgDefault(GenericProcessor(), 2),
             fuse_fma = ArgDefault(True, 2), 
             fast_path_extract = ArgDefault(True, 2),
             # Debug verbosity
             debug_flag = ArgDefault(False, 2),
             vector_size = ArgDefault(1, 2),
             language = ArgDefault(C_Code, 2),
             auto_test = ArgDefault(False, 2),
             auto_test_range = ArgDefault(Interval(-1, 1), 2),
             auto_test_std = ArgDefault(False, 2),
             arg_template = DefaultArgTemplate 
         ):
    # selecting argument values among defaults
    base_name = ArgDefault.select_value([base_name])
    print "pre function_name: ", function_name, arg_template.function_name
    function_name = ArgDefault.select_value([arg_template.function_name, function_name])
    print "function_name: ", function_name
    print "output_file: ", arg_template.output_file, output_file 
    output_file = ArgDefault.select_value([arg_template.output_file, output_file])
    print output_file
    # Specification
    io_precisions = ArgDefault.select_value([io_precisions])
    abs_accuracy = ArgDefault.select_value([abs_accuracy])
    libm_compliant = ArgDefault.select_value([arg_template.libm_compliant, libm_compliant])
    # Optimization parameters
    processor = ArgDefault.select_value([arg_template.target, processor])
    fuse_fma = ArgDefault.select_value([arg_template.fuse_fma, fuse_fma])
    fast_path_extract = ArgDefault.select_value([arg_template.fast_path_extract, fast_path_extract])
    # Debug verbosity
    debug_flag    = ArgDefault.select_value([arg_template.debug, debug_flag])
    vector_size   = ArgDefault.select_value([arg_template.vector_size, vector_size])
    language      = ArgDefault.select_value([arg_template.language, language])
    auto_test     = ArgDefault.select_value([arg_template.auto_test, arg_template.auto_test_execute, auto_test])
    auto_test_std = ArgDefault.select_value([arg_template.auto_test_std, auto_test_std])

    self.pre_gen_passes = arg_template.pre_gen_passes

    # io_precisions must be a list
    #     -> with a single element
    # XOR -> with as many elements as function arity (input + output arities)
    self.io_precisions = io_precisions

    ## enable the generation of numeric/functionnal auto-test
    self.auto_test_enable = (auto_test != False or auto_test_std != False)
    self.auto_test_number = auto_test
    self.auto_test_execute = ArgDefault.select_value([arg_template.auto_test_execute])
    self.auto_test_range = ArgDefault.select_value([arg_template.auto_test_range, auto_test_range])
    self.auto_test_std   = auto_test_std 

    self.language = language

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

    self.C_code_generator = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not self.debug_flag, libm_compliant = self.libm_compliant, language = self.language)
    uniquifier = self.function_name
    self.main_code_object = NestedCode(self.C_code_generator, static_cst = True, uniquifier = "{0}_".format(self.function_name))

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
    raise NotImplementedError


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
  #  @param copy  dict(optree -> optree) copy map to be used while duplicating
  #               pre_scheme (if None disable copy)
  #  @param enable_subexpr_sharing boolean flag, enables sub-expression sharing
  #         optimization
  #  @param verbose boolean flag, enable verbose mode
  #  @return optimizated scheme 
  def optimise_scheme(self, pre_scheme, copy = None,
                      enable_subexpr_sharing = True, verbose = True):
    """ default scheme optimization """
    # copying when required
    scheme = pre_scheme if copy is None else pre_scheme.copy(copy)
    # fusing FMA
    if self.fuse_fma:
      if verbose: print "MDL fusing FMA"
      scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

    if verbose: print "MDL abstract scheme"
    self.opt_engine.instantiate_abstract_precision(scheme,
                                                   default_precision = None)

    if verbose: print "MDL instantiated scheme"
    self.opt_engine.instantiate_precision(scheme, default_precision = None)

    if enable_subexpr_sharing:
      if verbose: print "subexpression sharing"
      self.opt_engine.subexpression_sharing(scheme)

    if verbose: print "silencing operation"
    self.opt_engine.silence_fp_operations(scheme)

    if verbose: print "checking processor support"
    self.opt_engine.check_processor_support(scheme, language = self.language)

    return scheme


  ## 
  #  @return main code object associted with function implementation
  def get_main_code_object(self):
    return self.main_code_object


  def generate_C(self, code_function_list):
    return self.generate_code(code_function_list, language = C_Code)

  ## generate C code for function implenetation 
  #  Code is generated within the main code object
  #  and dumped to a file named after implementation's name
  #  @param code_function_list list of CodeFunction to be generated (as sub-function )
  #  @return void
  def generate_code(self, code_function_list, language = C_Code):
    """ Final C generation, once the evaluation scheme has been optimized"""
    # registering scheme as function implementation
    #self.implementation.set_scheme(scheme)
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object
    for code_function in code_function_list:
      self.result = code_function.add_definition(self.C_code_generator,
                                                 language, code_object,
                                                 static_cst = True)

    # adding headers
    self.result.add_header("support_lib/ml_special_values.h")
    self.result.add_header("math.h")
    self.result.add_header("stdio.h")
    self.result.add_header("inttypes.h")

    Log.report(Log.Info, "Generating C code in " + self.output_file)
    output_stream = open(self.output_file, "w")
    output_stream.write(self.result.get(self.C_code_generator))
    output_stream.close()

  def gen_implementation(self, display_after_gen = False,
                         display_after_opt = False,
                         enable_subexpr_sharing = True):
    # generate scheme
    code_function_list = self.generate_function_list()
    if self.get_vector_size() != 1:
      scalar_scheme = self.implementation.get_scheme()
      scalar_arg_list = self.implementation.get_arg_list()
      self.implementation.clear_arg_list()

      code_function_list = self.generate_vector_implementation(
          scalar_scheme, scalar_arg_list, self.get_vector_size()
          )

    if self.auto_test_enable:
      code_function_list += self.generate_auto_test(
          test_num = self.auto_test_number if self.auto_test_number else 0,
          test_range = self.auto_test_range
          )


    for code_function in code_function_list:
      scheme = code_function.get_scheme()
      if display_after_gen:
        print "function %s, after gen " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True,
                             memoization_map = {})

      # optimize scheme
      opt_scheme = self.optimise_scheme(
          scheme, enable_subexpr_sharing = enable_subexpr_sharing
          )

      if display_after_opt:
        print "function %s, after opt " % code_function.get_name()
        print opt_scheme.get_str(depth = None, display_precision = True, memoization_map = {})

      # pre-generation optimization
      for pass_tag in self.pre_gen_passes:
        pass_class = Pass.get_pass_by_tag(pass_tag)
        pass_object = pass_class(self.processor)
        Log.report(Log.Info, "executing opt pass: {}".format(pass_tag))
        opt_scheme = pass_object.execute(opt_scheme)
        print "post pass scheme"
        print opt_scheme.get_str(depth = None, display_precision = True, memoization_map = {}, display_id = True)
      code_function.set_scheme(opt_scheme)

    # generate C code to implement scheme
    self.generate_code(code_function_list, language = self.language)

    if self.auto_test_enable:
      compiler = self.processor.get_compiler()
      test_file = "./test_%s.bin" % self.function_name
      test_command =  "%s -O2 -DML_DEBUG -I $ML_SRC_DIR/metalibm_core $ML_SRC_DIR/metalibm_core/support_lib/ml_libm_compatibility.c %s -o %s -lm " % (compiler, self.output_file, test_file) 
      test_command += " && %s " % self.processor.get_execution_command(test_file)
      if self.auto_test_execute:
        print "VALIDATION %s " % self.get_name()
        print test_command
        test_result = subprocess.call(test_command, shell = True)
        if not test_result:
          print "VALIDATION SUCCESS"
        else:
          print "VALIDATION FAILURE"
          sys.exit(1)
      else:
        print "VALIDATION %s command line:" % self.get_name()
        print test_command



  ## externalized an optree: generate a CodeFunction which compute the 
  #  given optree inside a sub-function and returns it as a result
  # @param optree ML_Operation object to be externalized
  # @param arg_list list of ML_Operation objects to be used as arguments
  # @return pair ML_Operation, CodeFunction
  def externalize_call(self, optree, arg_list, tag = "foo", result_format = None, name_factory = None):
    ext_function = self.call_externalizer.externalize_call(optree, arg_list, tag, result_format)
    return ext_function.get_function_object()(*arg_list), ext_function


  def generate_vector_implementation(self, scalar_scheme, scalar_arg_list,
                                     vector_size = 2):
    # declaring optimizer
    self.opt_engine.set_boolean_format(ML_Bool)
    self.vectorizer = StaticVectorizer(self.opt_engine)

    callback_name = self.uniquify_name("scalar_callback")

    scalar_callback_function = self.call_externalizer.externalize_call(scalar_scheme, scalar_arg_list, callback_name, self.precision)

    print "[SV] optimizing Scalar scheme"
    scalar_scheme = self.optimise_scheme(scalar_scheme)

    scalar_callback          = scalar_callback_function.get_function_object()

    print "[SV] vectorizing scheme"
    vec_arg_list, vector_scheme, vector_mask = \
        self.vectorizer.vectorize_scheme(scalar_scheme, scalar_arg_list,
                                         vector_size, self.call_externalizer,
                                         self.get_output_precision())

    vector_output_format = self.vectorizer.vectorize_format(self.precision,
                                                            vector_size)


    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    vec_res = Variable("vec_res", precision = vector_output_format,
                       var_type = Variable.Local)

    vec_elt_arg_tuple = tuple(
        VectorElementSelection(vec_arg, vi, precision = self.precision)
        for vec_arg in vec_arg_list
        )

    vector_mask.set_attributes(tag = "vector_mask", debug = debug_multi)

    print "[SV] building vectorized main statement"
    if self.language is OpenCL_Code:
      unrolled_cond_allocation = Statement()
      for i in xrange(vector_size):
        elt_index = Constant(i)
        vec_elt_arg_tuple = tuple(VectorElementSelection(vec_arg, elt_index, precision = self.precision) for vec_arg in vec_arg_list)
        unrolled_cond_allocation.add(
          ConditionBlock(
            Likely(VectorElementSelection(vector_mask, elt_index, precision = ML_Bool), None),
            ReferenceAssign(VectorElementSelection(vec_res, elt_index, precision = self.precision), scalar_callback(*vec_elt_arg_tuple)),
            # ReferenceAssign(VectorElementSelection(vec_res, elt_index, precision = self.precision), VectorElementSelection(vector_scheme, elt_index, precision = self.precision))
          )
        ) 


      function_scheme = Statement(
        vector_scheme,
        ConditionBlock(
          Test(vector_mask, specifier = Test.IsMaskNotAnyZero, precision = ML_Bool, likely = True, debug = debug_multi),
          Return(vector_scheme),
          Statement(
            ReferenceAssign(vec_res, vector_scheme),
            unrolled_cond_allocation,
            Return(vec_res)
          )
        )
      )

    else:
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

    # print "vectorized_scheme: ", function_scheme.get_str(depth = None, display_precision = True, memoization_map = {})

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

  ## provide numeric evaluation of the main function on @p input_value
  def numeric_emulate(self, input_value):
    raise NotImplementedError

  def generate_auto_test(self, test_num = 10, test_range = Interval(-1.0, 1.0), debug = False):
    low_input = inf(test_range)
    high_input = sup(test_range)
    auto_test = CodeFunction("main", output_format = ML_Int32)

    test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

    tested_function    = self.implementation.get_function_object()
    function_name      = self.implementation.get_name()

    failure_report_op       = FunctionOperator("report_failure")
    failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)

    printf_op = FunctionOperator("printf", arg_map = {0: "\"error[%%d]: %s(%%f/%%a)=%%a vs expected = %%a \\n\"" % function_name, 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3), 5: FO_Arg(4)}, void_function = True) 
    printf_function = FunctionObject("printf", [ML_Int32] + [self.precision] * 4, ML_Void, printf_op)

    printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True) 
    printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)


    num_std_case = len(self.standard_test_cases)
    test_total   = test_num 
    if self.auto_test_std:
      test_total += num_std_case

    diff = self.get_vector_size() - (test_total % self.get_vector_size())
    test_total += diff
    test_num   += diff

    sollya_precision = self.precision.get_sollya_object()
    interval_size = high_input - low_input 

    input_table = ML_Table(dimensions = [test_total], storage_precision = self.precision, tag = self.uniquify_name("input_table"))
    ## (low, high) are store in output table
    output_table = ML_Table(dimensions = [test_total, 2], storage_precision = self.precision, tag = self.uniquify_name("output_table"))

    # general index for input/output tables
    table_index = 0


    if self.auto_test_std:
      # standard test cases
      for i in range(num_std_case):
        input_value = round(self.standard_test_cases[i], sollya_precision, RN)

        input_table[table_index] = input_value
        # FIXME only valid for faithful evaluation
        output_table[table_index][0] = round(self.numeric_emulate(input_value), sollya_precision, RD)
        output_table[table_index][1] = round(self.numeric_emulate(input_value), sollya_precision, RU)

        table_index += 1

    # random test cases
    for i in range(test_num):
      input_value = random.uniform(low_input, high_input)
      input_value = round(input_value, sollya_precision, RN)
      #input_value = round(low_input + (random.randrange(2**32 + 1) / float(2**32)) * interval_size, sollya_precision, RN) 
      input_table[table_index] = input_value
      # FIXME only valid for faithful evaluation
      output_table[table_index][0] = round(self.numeric_emulate(input_value), sollya_precision, RD)
      output_table[table_index][1] = round(self.numeric_emulate(input_value), sollya_precision, RU)
      table_index += 1


    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)

    assignation_statement = Statement()

    if self.implementation.get_output_format().is_vector_format():
      # vector implementation
      vector_format = self.implementation.get_output_format()

      # building inputs
      local_input = Variable("vec_x", precision = vector_format, var_type = Variable.Local) 
      assignation_statement.push(local_input)
      for k in xrange(self.get_vector_size()):
        elt_assign = ReferenceAssign(VectorElementSelection(local_input, k), TableLoad(input_table, vi + k))
        assignation_statement.push(elt_assign)

      # computing results
      local_result = tested_function(local_input)
      loop_increment = self.get_vector_size()

      comp_statement = Statement()

      # comparison with expected
      for k in xrange(self.get_vector_size()):
        elt_input  = VectorElementSelection(local_input, k)
        elt_result = VectorElementSelection(local_result, k)
        low_bound    = TableLoad(output_table, vi + k, 0)
        high_bound   = TableLoad(output_table, vi + k, 1)

        failure_test = LogicalOr(
          Comparison(elt_result, low_bound, specifier = Comparison.Less),
          Comparison(elt_result, high_bound, specifier = Comparison.Greater)
        )

        comp_statement.push(
          ConditionBlock(
            failure_test,
            Statement(
              printf_function(vi + k, elt_input, elt_input, elt_result, high_bound), 
              Return(Constant(1, precision = ML_Int32))
         )))


      test_loop = Loop(
        ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
        vi < test_num_cst,
        Statement(
          assignation_statement,
          comp_statement,
          ReferenceAssign(vi, vi + loop_increment)
        ),
      )
    else: 
      # scalar function
      local_input  = TableLoad(input_table, vi)
      local_result = tested_function(local_input)
      low_bound    = TableLoad(output_table, vi, 0)
      high_bound   = TableLoad(output_table, vi, 1)

      failure_test = LogicalOr(
        Comparison(local_result, low_bound, specifier = Comparison.Less),
        Comparison(local_result, high_bound, specifier = Comparison.Greater)
      )

      loop_increment = self.get_vector_size()

      test_loop = Loop(
        ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
        vi < test_num_cst,
        Statement(
          assignation_statement,
          ConditionBlock(
            failure_test,
            Statement(
              printf_function(vi, local_input, local_input, local_result, high_bound), 
              Return(Constant(1, precision = ML_Int32))
            ),
          ),
          ReferenceAssign(vi, vi + loop_increment)
        ),
      )
    # common test scheme between scalar and vector functions
    test_scheme = Statement(
      test_loop,
      printf_success_function(),
      Return(Constant(0, precision = ML_Int32))
    )
    auto_test.set_scheme(test_scheme)
    return [auto_test]

  @staticmethod
  def get_name():
    return ML_FunctionBasis.function_name

  # list of input to be used for standard test validation
  standard_test_cases = []


        
def ML_Function(name):
  new_class = type(name, (ML_FunctionBasis,), {"function_name": name})
  new_class.get_name = staticmethod(lambda: name) 
  return new_class

# end of Doxygen's ml_function group
## @}

