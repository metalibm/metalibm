# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

###############################################################################
# created:
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import os
import random
import subprocess

from sollya import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.ml_call_externalizer import CallExternalizer
from metalibm_core.core.ml_vectorizer import StaticVectorizer
from metalibm_core.core.precisions import *

from metalibm_core.code_generation.code_object import (
    NestedCode, CodeObject, LLVMCodeObject, MultiSymbolTable
)
from metalibm_core.code_generation.code_function import (
    CodeFunction, FunctionGroup
)
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.llvm_ir_code_generator import LLVMIRCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
#from metalibm_core.code_generation.generator_utility import *
from metalibm_core.core.passes import (
    Pass, PassScheduler, PassDependency, AfterPassById
)

from metalibm_core.opt.p_function_std import (
    PassCheckProcessorSupport, PassSubExpressionSharing, PassFuseFMA
)
from metalibm_core.opt.p_function_typing import (
    PassInstantiateAbstractPrecision, PassInstantiatePrecision,
)

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.ml_template import DefaultArgTemplate


## \defgroup ml_function ml_function
## @{


class BuildError(Exception):
    """ Exception to indicate that a build stage failed """
    pass
class ValidError(Exception):
    """ Exception to indicate that a validation stage failed """
    pass


def build_code_function(src_list, bin_file, processor, link_trigger=False):
    """ Build the code function for processor
        Args:
            src_list(list): list of source file (string)
            bin_file(str): name of the binary file (build result)
            processor: target
            link_trigger: enable/disable binary link
        Return:
            bool, str (error, stdout) """
    compiler = processor.get_compiler()
    test_file = bin_file
    DEFAULT_OPTIONS = ["-O2", "-DML_DEBUG"]
    compiler_options = " ".join(DEFAULT_OPTIONS + processor.get_compilation_options())
    if not(link_trigger):
        # build only, disable link
        compiler_options += " -c  "
    else:
        src_list += [
            "%s/metalibm_core/support_lib/ml_libm_compatibility.c" % (os.environ["ML_SRC_DIR"]),
            "%s/metalibm_core/support_lib/ml_multi_prec_lib.c" % (os.environ["ML_SRC_DIR"]),
        ]
    Log.report(Log.Info, "Compiler options: \"{}\"".format(compiler_options))

    build_command = "{compiler} {options} -I{ML_SRC_DIR}/metalibm_core \
    {src_files} -o {test_file} -lm ".format(
        compiler=compiler,
        src_files = (" ".join(src_list)),
        test_file=test_file,
        options=compiler_options,
        ML_SRC_DIR=os.environ["ML_SRC_DIR"])

    Log.report(Log.Info, "Building source with command: {}".format(build_command))
    build_result, build_stdout = get_cmd_stdout(build_command)
    return build_result, build_stdout

def get_cmd_stdout(cmd):
    """ execute cmd on a subprocess and return return-code and stdout
        message """
    cmd_process = subprocess.Popen(
        filter(None, cmd.split(" ")), stdout=subprocess.PIPE, env=os.environ.copy())
    returncode = cmd_process.wait()
    return returncode, cmd_process.stdout.read()


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
  #   @param all arguments are transmittaed throughs @p arguments object which 
  #          should inherit from DefaultArgTemplate
  def __init__(self, args=DefaultArgTemplate):
    # selecting argument values among defaults
    self.display_after_opt = args.display_after_opt

    # enable/disable check_processor_support pass run
    self.check_processor_support = args.check_processor_support

    self.arity = args.arity
    self.precision = args.precision
    # io_precisions must be:
    #     -> a list
    # XOR -> None to select [se;f.precision] * self.get_arity()
    self.input_precisions = [self.precision] * self.get_arity() if args.input_precisions is None else args.input_precisions

    # enable the generation of numeric/functionnal auto-test
    self.auto_test_enable = (args.auto_test != False or args.auto_test_std != False)
    self.auto_test_number = args.auto_test
    self.auto_test_range = args.auto_test_range
    self.auto_test_std   = args.auto_test_std 

    # enable the computation of maximal error during functional testing
    self.compute_max_error = args.compute_max_error
    self.break_error = args.break_error

    # enable and configure the generation of a performance bench
    self.bench_enabled = args.bench_test_number 
    self.bench_test_number = args.bench_test_number
    self.bench_test_range = args.bench_test_range

    # source building
    self.build_enable = args.build_enable
    # binary execution
    self.execute_trigger = args.execute_trigger

    self.language = args.language

    Log.report(Log.Info, "auto test: {}, {}, {}".format(self.auto_test_enable, self.auto_test_number,  self.auto_test_range))

    # Naming logic, using provided information if available, otherwise deriving from base_name
    # base_name is e.g. exp
    # function_name is e.g. expf or expd or whatever 
    self.function_name = args.function_name if args.function_name else libc_naming(args.base_name, [self.precision] + self.input_precisions)

    self.output_file = args.output_file if args.output_file else self.function_name + ".c"

    self.debug_flag = args.debug

    self.vector_size = args.vector_size
    self.sub_vector_size = args.sub_vector_size

    # TODO: FIX which i/o precision to select
    # TODO: incompatible with fixed-point formats
    # self.sollya_precision = self.get_output_precision().get_sollya_object()

    # self.abs_accuracy = args.abs_accuracy if args.abs_accuracy else S2**(-self.get_output_precision().get_precision())
    self.libm_compliant = args.libm_compliant
    self.accuracy_class = args.accuracy
    self.accuracy = args.accuracy(self.get_output_precision())
    
    self.processor = args.target

    self.fuse_fma = args.fuse_fma
    self.dot_product_enabled = args.dot_product_enabled
    self.fast_path_extract = args.fast_path_extract

    # instance of CodeFunction containing the function implementation
    self.implementation = CodeFunction(self.function_name, output_format=self.get_output_precision())
    # instance of OptimizationEngine
    self.opt_engine = OptimizationEngine(self.processor, dot_product_enabled=self.dot_product_enabled)
    # instance of GappaCodeGenerator to perform inline proofs
    self.gappa_engine = GappaCodeGenerator(self.processor, declare_cst=True, disable_debug=True)
    # instance of Code Generation to generate source code
    CODE_GENERATOR_CLASS = self.get_codegen_class(self.language)
    CODE_OBJECT_CLASS = self.get_codeobject_ctor(self.language)

    self.main_code_generator = CODE_GENERATOR_CLASS(
        self.processor, declare_cst=False, disable_debug=not self.debug_flag,
        libm_compliant=self.libm_compliant, language=self.language
    )
    uniquifier = self.function_name
    shared_symbol_list = [
        MultiSymbolTable.ConstantSymbol,
        MultiSymbolTable.TableSymbol,
        MultiSymbolTable.FunctionSymbol,
        MultiSymbolTable.EntitySymbol
    ] 
    if self.language is LLVM_IR_Code:
        shared_symbol_list.append(MultiSymbolTable.VariableSymbol)
        shared_symbol_list.append(MultiSymbolTable.LabelSymbol)
    # main code object
    self.main_code_object = NestedCode(
        self.main_code_generator, static_cst=True,
        uniquifier="{0}_".format(self.function_name),
        code_ctor=CODE_OBJECT_CLASS,
        shared_symbol_list=shared_symbol_list)

    # pass scheduler
    # pass scheduler instanciation
    self.pass_scheduler = PassScheduler(
        pass_tag_list=[
            PassScheduler.Start,
            PassScheduler.Typing,
            PassScheduler.Optimization, 
            PassScheduler.JustBeforeCodeGen
        ]
    )


    Log.report(Log.Info, "inserting sub-expr sharing pass\n")
    pass_SES_id = self.pass_scheduler.register_pass(
        PassSubExpressionSharing(self.processor),
        pass_slot=PassScheduler.Optimization
    )
    #Log.report(Log.Info, "inserting fused fma pass\n")
    #self.pass_scheduler.register_pass(
    #    PassFuseFMA(self.processor, dot_product_enabled=self.dot_product_enabled),
    #    pass_slot=PassScheduler.Optimization
    #    )
    Log.report(Log.Info, "inserting instantiate abstract precision pass\n")
    pass_inst_abstract_prec = PassInstantiateAbstractPrecision(self.processor)
    pass_IAP_id = self.pass_scheduler.register_pass(
        pass_inst_abstract_prec,
        pass_slot=PassScheduler.Typing
        )
    Log.report(Log.Info, "inserting instantiate precision pass\n")
    pass_inst_prec = PassInstantiatePrecision(self.processor, default_precision=None)
    pass_IP_id = self.pass_scheduler.register_pass(
        pass_inst_prec,
        pass_dep = AfterPassById(pass_IAP_id),
        pass_slot=PassScheduler.Typing
        )
    # register the id of the last pass for each slot
    pass_slot_deps = {
        PassScheduler.Optimization: AfterPassById(pass_SES_id),
        PassScheduler.Typing: AfterPassById(pass_IP_id),
        PassScheduler.JustBeforeCodeGen: PassDependency(),
    }

    # empty pass dependency
    for pass_uplet in args.passes:
      pass_slot_tag, pass_tag = pass_uplet.split(":")
      pass_slot = PassScheduler.get_tag_class(pass_slot_tag)
      pass_class  = Pass.get_pass_by_tag(pass_tag)
      pass_object = pass_class(self.processor)
      if not pass_slot in pass_slot_deps:
        pass_slot_deps[pass_slot_dep] = PassDependency()
      pass_dep = pass_slot_deps[pass_slot]
      custom_pass_id = self.pass_scheduler.register_pass(pass_object, pass_dep=pass_dep, pass_slot=pass_slot)
      # linearly linking pass in the order they appear
      pass_slot_deps[pass_slot] = AfterPassById(custom_pass_id)

    # appending check_processor_support pass after custom passes
    Log.report(Log.Info, "inserting target support check pass\n")
    self.pass_scheduler.register_pass(
        PassCheckProcessorSupport(self.processor, self.language),
        pass_slot=PassScheduler.JustBeforeCodeGen,
        pass_dep=pass_slot_deps[PassScheduler.JustBeforeCodeGen],
    )

  def get_codegen_class(self, language):
    """ return the code generator class associated with a given language """
    return {
        C_Code: CCodeGenerator,
        OpenCL_Code: CCodeGenerator,
        LLVM_IR_Code: LLVMIRCodeGenerator
    }[language]

  def get_codeobject_ctor(self, language):
    """ return the basic code object class associated with a given language """
    return {
        C_Code: CodeObject,
        OpenCL_Code: CodeObject,
        LLVM_IR_Code: LLVMCodeObject,
    }[language]

  def get_vector_size(self):
    return self.vector_size


  ## generate a default argument template
  #  may be overloaded by sub-class to provide
  #  a meta-function specific default argument structure
  @staticmethod
  def get_default_args(**args):
    return DefaultArgTemplate(**args)

  ## Return function's arity (number of input arguments)
  #  Default to 1
  def get_arity(self):
    return self.arity

  ## compute the evaluation error of an ML_Operation node
  #  @param optree ML_Operation object whose evaluation error is computed
  #  @param variable_copy_map dict(optree -> optree) used to delimit the
  #         bound of optree
  #  @param goal_precision ML_Format object, precision used for evaluation goal
  #  @param gappa_filename string, name of the file where the gappa proof
  #         of the evaluation error will be dumped
  #  @return numerical value of the evaluation error
  def get_eval_error(
        self, optree, variable_copy_map = {}, goal_precision = ML_Exact,
        gappa_filename = "gappa_eval_error.g", relative_error = False
    ):
    """ wrapper for GappaCodeGenerator get_eval_error_v2 function """
    copy_map = {}
    for leaf in variable_copy_map:
      copy_map[leaf] = variable_copy_map[leaf]
    opt_optree = self.optimise_scheme(optree, copy = copy_map, verbose = False)
    new_variable_copy_map = {}
    for leaf in variable_copy_map:
      new_variable_copy_map[leaf.get_handle().get_node()] = variable_copy_map[leaf]
    return self.gappa_engine.get_eval_error_v2(
        self.opt_engine, opt_optree,
        new_variable_copy_map if variable_copy_map != None else {},
        goal_precision, gappa_filename, relative_error = relative_error
    )

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
    return self.precision
  def get_input_precision(self, index = 0):
    return self.input_precisions[index]
  def get_input_precisions(self):
    return self.input_precisions

  def get_sollya_precision(self):
    """ return the main precision use for sollya calls """
    return self.sollya_precision

  def generate_scheme(self):
    """ generate MDL scheme for function implementation """
    Log.report(Log.Error, "generate_scheme must be overloaded by ML_FunctionBasis child")

  ## Return the list of CodeFunction objects (main function
  #  and sub-functions) to be used to build @p self implementation
  #  This function may be overloaded by child class to define
  #  a specific way to build CodeFunction objects
  #
  # @return main_scheme, [list of sub-CodeFunction object]
  def generate_function_list(self):
    self.implementation.set_scheme(self.generate_scheme())
    return FunctionGroup([self.implementation])

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
      Log.report(Log.Verbose, "MDL fusing FMA")
      scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

    Log.report(Log.Verbose, "MDL abstract scheme")
    self.opt_engine.instantiate_abstract_precision(scheme,
                                                   default_precision = None)

    Log.report(Log.Verbose, "MDL instantiated scheme")
    self.opt_engine.instantiate_precision(scheme, default_precision = None)

    if enable_subexpr_sharing:
      Log.report(Log.Verbose, "subexpression sharing")
      self.opt_engine.subexpression_sharing(scheme)

    Log.report(Log.Verbose, "silencing operation")
    self.opt_engine.silence_fp_operations(scheme)

    return scheme


  ##
  #  @return main code object associted with function implementation
  def get_main_code_object(self):
    return self.main_code_object


  def generate_code(self, function_group, language):
    if self.language == C_Code:
        return self.generate_C_code(function_group, language=language)
    elif self.language == LLVM_IR_Code:
        return self.generate_LLVM_code(function_group, language=language)

  def generate_LLVM_code(self, function_group, language = LLVM_IR_Code):
    """ Final LLVM-IR generation, once the evaluation scheme has been optimized"""
    Log.report(Log.Info, "Generating Source Code ")
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object

    def gen_code_function_code(fct_group, fct):
        self.result = fct.add_definition(
            self.main_code_generator,
            language, code_object, static_cst=True)

    function_group.apply_to_all_functions(gen_code_function_code)

    #for code_function in code_function_list:
    #  self.result = code_function.add_definition(self.main_code_generator,
    #                                             language, code_object,
    #                                             static_cst = True)

    # adding headers
    Log.report(Log.Info, "Generating LLVM-IR code in " + self.output_file)
    output_stream = open(self.output_file, "w")
    output_stream.write(self.result.get(self.main_code_generator))
    output_stream.close()

  ## generate C code for function implenetation
  #  Code is generated within the main code object
  #  and dumped to a file named after implementation's name
  #  @param code_function_list list of CodeFunction to be generated (as sub-function )
  #  @return void
  def generate_C_code(self, function_group, language = C_Code):
    """ Final C generation, once the evaluation scheme has been optimized"""
    Log.report(Log.Info, "Generating Source Code ")
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object

    def gen_code_function_code(fct_group, fct):
        self.result = fct.add_definition(
            self.main_code_generator,
            language, code_object, static_cst=True)

    function_group.apply_to_all_functions(gen_code_function_code)

    #for code_function in code_function_list:
    #  self.result = code_function.add_definition(self.main_code_generator,
    #                                             language, code_object,
    #                                             static_cst = True)

    # adding headers
    self.result.add_header("support_lib/ml_special_values.h")
    self.result.add_header("math.h")
    self.result.add_header("stdio.h")
    self.result.add_header("inttypes.h")

    Log.report(Log.Info, "Generating C code in " + self.output_file)
    output_stream = open(self.output_file, "w")
    output_stream.write(self.result.get(self.main_code_generator))
    output_stream.close()

  def gen_implementation(self, display_after_gen=False,
                         display_after_opt=False,
                         enable_subexpr_sharing=True):
    """ generate implementation

        Args:
            display_after_gen enable (bool): I.R dump after generation
            display_after_opt enable (bool): I.R dump after optimization
            enable_subexpr_sharing (bool): I.R enable sub-expression sharing
               optimization

        """
    # generate scheme
    function_group = self.generate_function_list()

    ## apply @p pass_object optimization pass
    #  to the scheme of each entity in code_entity_list
    def execute_pass_on_fct_group(scheduler, pass_object, function_group):
        """ execute an optimization pass on a function_group """
        return pass_object.execute_on_fct_group(function_group)

    Log.report(Log.Info, "Applying <Start> stage passes")
    _ = self.pass_scheduler.get_full_execute_from_slot(
      function_group,
      PassScheduler.Start,
      execute_pass_on_fct_group
    )

    # generate vector size
    if self.get_vector_size() != 1:
        scalar_scheme = self.implementation.get_scheme()
        scalar_arg_list = self.implementation.get_arg_list()
        self.implementation.clear_arg_list()

        function_group = self.generate_vector_implementation(
            scalar_scheme, scalar_arg_list, self.get_vector_size()
        )

    # format instantiation
    Log.report(Log.Info, "Applying <Typing> stage passes")
    _ = self.pass_scheduler.get_full_execute_from_slot(
        function_group,
        PassScheduler.Typing,
        execute_pass_on_fct_group
    )

    # format instantiation
    Log.report(Log.Info, "Applying <Optimization> stage passes")
    _ = self.pass_scheduler.get_full_execute_from_slot(
        function_group,
        PassScheduler.Optimization,
        execute_pass_on_fct_group
    )

    # format instantiation
    Log.report(Log.Info, "Applying <JustBeforeCodeGen> stage passes")
    _ = self.pass_scheduler.get_full_execute_from_slot(
        function_group,
        PassScheduler.JustBeforeCodeGen,
        execute_pass_on_fct_group
    )

    main_pre_statement = Statement()
    main_statement = Statement()

    CstError = Constant(1, precision=ML_Int32)
    CstSuccess = Constant(0, precision=ML_Int32)

    def add_fct_call_check_in_main(fct_group, code_function):
        """ adding call to code_function with return value check
            in main statement """
        scheme = code_function.get_scheme()
        opt_scheme = self.optimise_scheme(
            scheme, enable_subexpr_sharing = enable_subexpr_sharing
        )
        code_function.set_scheme(opt_scheme)
        fct_call = code_function.build_function_object()()
        main_pre_statement.add(fct_call)
        main_statement.add(
            ConditionBlock(
                fct_call,
                Return(CstError)
            )
        )

    # generate auto-test wrapper
    if self.auto_test_enable:
        auto_test_function_group = self.generate_test_wrapper(
            test_num = self.auto_test_number if self.auto_test_number else 0,
            test_range = self.auto_test_range
        )
        auto_test_function_group.apply_to_all_functions(add_fct_call_check_in_main)
        # appending auto-test wrapper to general code_function_list
        function_group.merge_with_group(auto_test_function_group)

    if self.bench_enabled:
        bench_function_group = self.generate_bench_wrapper(
            test_num = self.bench_test_number if self.bench_test_number else 1000,
            test_range = self.bench_test_range
        )

        bench_function_group.apply_to_all_functions(add_fct_call_check_in_main)
        # appending bench wrapper to general code_function_list
        function_group.merge_with_group(bench_function_group)

    # adding main function
    if self.bench_enabled or self.auto_test_enable:
        main_function = CodeFunction("main", output_format=ML_Int32)
        main_function.set_scheme(
            Statement(
                main_pre_statement,
                main_statement,
                Return(CstSuccess)
            )
        )
        function_group.add_core_function(main_function)

    # generate C code to implement scheme
    self.generate_code(function_group, language = self.language)

    build_trigger = self.build_enable or self.execute_trigger
    link_trigger = self.execute_trigger

    if build_trigger:
        test_file = "./test_%s.bin" % self.function_name
        build_result, build_stdout = build_code_function(
            [self.output_file],
            test_file, 
            self.processor,
            link_trigger)

        if build_result:
            Log.report(
                Log.Error, "build failed: \n {}".format(build_stdout),
                error=BuildError()
            )
        else:
            Log.report(Log.Info, "build result: {}\n{}".format(build_result, build_stdout))

        # only executing if build was successful
        if not(build_result) and self.execute_trigger:
            test_command = " %s " % self.processor.get_execution_command(test_file)
            Log.report(Log.Info, "VALIDATION {} command line: {}".format(
                self.get_name(), test_command
            ))
            # executing test command
            test_result, test_stdout = get_cmd_stdout(test_command)
            if not test_result:
                print(test_stdout)
                Log.report(Log.Info, "VALIDATION SUCCESS")
            else:
                 Log.report(
                    Log.Error, "VALIDATION FAILURE [{}]\n{}".format(test_result, test_stdout),
                    error=ValidError()
                )



  ## externalized an optree: generate a CodeFunction which compute the 
  #  given optree inside a sub-function and returns it as a result
  # @param optree ML_Operation object to be externalized
  # @param arg_list list of ML_Operation objects to be used as arguments
  # @return pair ML_Operation, CodeFunction
  def externalize_call(self, optree, arg_list, tag = "foo", result_format = None, name_factory = None):
    # Call externalizer engine
    call_externalizer = CallExternalizer(self.get_main_code_object())
    ext_function = call_externalizer.externalize_call(optree, arg_list, tag, result_format)
    return ext_function.get_function_object()(*arg_list), ext_function


  ## Generate an OpenCL-compatible wrapper for a vectorized scheme 
  #  @p vector_scheme by testing vector mask element and branching
  #  to scalar callback when necessary
  def generate_opencl_vector_wrapper(self, vector_size, vec_arg_list, vector_scheme, vector_mask, vec_res, scalar_callback):
    unrolled_cond_allocation = Statement()
    for i in range(vector_size):
      elt_index = Constant(i)
      vec_elt_arg_tuple = tuple(VectorElementSelection(vec_arg, elt_index, precision = self.precision) for vec_arg in vec_arg_list)
      unrolled_cond_allocation.add(
        ConditionBlock(
          Likely(
            LogicalNot(
              VectorElementSelection(
                vector_mask, 
                elt_index, 
                precision = ML_Bool
              ),
              precision = ML_Bool
            ),
            None
          ),
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
    return function_scheme

  ## Generate a C-compatible wrapper for a vectorized scheme 
  #  @p vector_scheme by testing vector mask element and branching
  #  to scalar callback when necessary
  #
  #  @param vector_size number of element in a vector
  #  @param vector_arg_list
  #  @param vector_scheme
  #  @param vector_mask
  def generate_c_vector_wrapper(self, vector_size, vec_arg_list, vector_scheme, vector_mask, vec_res, scalar_callback):

    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    vec_elt_arg_tuple = tuple(
      VectorElementSelection(vec_arg, vi, precision = self.precision)
      for vec_arg in vec_arg_list
    )

    function_scheme = Statement(
      vector_scheme,
      ConditionBlock(
        # if there is not any zero in the mask. then
        # the vector result may be returned
        Test(
          vector_mask,
          specifier = Test.IsMaskNotAnyZero,
          precision = ML_Bool,
          likely = True,
          debug = debug_multi
        ),
        Return(vector_scheme, precision=vector_scheme.get_precision()),
        Statement(
          ReferenceAssign(vec_res, vector_scheme),
          Loop(
            ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
            vi < Constant(vector_size, precision = ML_Int32),
            Statement(
              ConditionBlock(
                LogicalNot(
                  Likely(
                    VectorElementSelection(
                      vector_mask, vi, precision = ML_Bool
                    ),
                    None
                  ),
                  precision = ML_Bool
                ),
                ReferenceAssign(
                  VectorElementSelection(
                    vec_res, vi, precision = self.precision
                  ),
                  scalar_callback(*vec_elt_arg_tuple)
                )
              ),
              ReferenceAssign(vi, vi + 1)
            ),
          ),
          Return(vec_res, precision=vec_res.get_precision())
        )
      )
    )
    return function_scheme

  def generate_vector_implementation(self, scalar_scheme, scalar_arg_list,
                                     vector_size = 2):
    # declaring optimizer
    self.opt_engine.set_boolean_format(ML_Bool)
    self.vectorizer = StaticVectorizer(self.opt_engine)

    callback_name = self.uniquify_name("scalar_callback")

    # Call externalizer engine
    call_externalizer = CallExternalizer(self.get_main_code_object())
    scalar_callback_function = call_externalizer.externalize_call(scalar_scheme, scalar_arg_list, callback_name, self.precision)

    Log.report(Log.Info, "[SV] optimizing Scalar scheme")
    scalar_scheme = self.optimise_scheme(scalar_scheme)

    scalar_callback          = scalar_callback_function.get_function_object()

    Log.report(Log.Info, "[SV] vectorizing scheme")
    vec_arg_list, vector_scheme, vector_mask = \
        self.vectorizer.vectorize_scheme(scalar_scheme, scalar_arg_list,
                                         vector_size, call_externalizer,
                                         self.get_output_precision(), self.sub_vector_size)

    vector_output_format = self.vectorizer.vectorize_format(self.precision,
                                                            vector_size)


    Log.report(Log.Info, "vector_output_format is {}".format(vector_output_format))
    vec_res = Variable("vec_res", precision=vector_output_format,
                       var_type = Variable.Local)


    vector_mask.set_attributes(tag = "vector_mask", debug = debug_multi)

    ## Test whether a vector-mask is fully set to True
    #  in order to disable scalar fallback generation
    def no_scalar_fallback_required(mask):
      return isinstance(mask, Constant) and \
            reduce(lambda v, acc: (v and acc), mask.get_value(), True)

    if self.language in [C_Code, OpenCL_Code]:
        self.get_main_code_object().add_header("support_lib/ml_vector_format.h")

    Log.report(Log.Info, "[SV] building vectorized main statement")
    if no_scalar_fallback_required(vector_mask):
      function_scheme = Statement(
        Return(vector_scheme, precision=vector_output_format)
      )
    elif self.language is OpenCL_Code:
      function_scheme = self.generate_opencl_vector_wrapper(vector_size, vec_arg_list, vector_scheme, vector_mask, vec_res, scalar_callback)

    else:
      function_scheme = self.generate_c_vector_wrapper(vector_size, vec_arg_list, vector_scheme, vector_mask, vec_res, scalar_callback)

    # print "vectorized_scheme: ", function_scheme.get_str(depth = None, display_precision = True, memoization_map = {})

    for vec_arg in vec_arg_list:
      self.implementation.register_new_input_variable(vec_arg)
    self.implementation.set_output_format(vector_output_format)

    # dummy scheme to make functionnal code generation
    self.implementation.set_scheme(function_scheme)

    Log.report(Log.Info, "[SV] end of generate_vector_implementation")
    return FunctionGroup([self.implementation], [scalar_callback_function])


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
  #  @param input_value SollyaObject numeric input value
  #  @return SollyaObject numeric output value corresponding
  #          to emulation of @p self function on @p input_value
  def numeric_emulate(self, input_value):
    raise NotImplementedError


  ## Generate a test wrapper for the @p self function 
  #  @param test_num   number of test to perform
  #  @param test_range numeric range for test's inputs
  #  @param debug enable debug mode
  def generate_test_wrapper(self, test_num = 10, test_range=Interval(-1.0, 1.0), debug=False):
    low_input = inf(test_range)
    high_input = sup(test_range)
    auto_test = CodeFunction("test_wrapper", output_format = ML_Int32)

    tested_function    = self.implementation.get_function_object()
    function_name      = self.implementation.get_name()

    failure_report_op       = FunctionOperator("report_failure")
    failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)

    printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True) 
    printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

    test_total   = test_num 
    # compute the number of standard test cases
    num_std_case = len(self.standard_test_cases)
    # add them to the total if standard test enabled
    if self.auto_test_std:
      test_total += num_std_case
    # round up the number of tests to the implementation vector-size
    diff = (self.get_vector_size() - (test_total % self.get_vector_size())) % self.get_vector_size()
    assert diff >= 0
    test_total += diff
    test_num   += diff

    Log.report(Log.Info, "test test_total, test_num, diff: {} {} {}".format(test_total, test_num, diff))

    sollya_precision = self.precision.get_sollya_object()
    interval_size = high_input - low_input 

    input_tables = [
      ML_NewTable(
        dimensions = [test_total], 
        storage_precision = self.get_input_precision(i), 
        tag = self.uniquify_name("input_table_arg%d" % i)
      ) for i in range(self.get_arity())
    ]
    ## output values required to check results are stored in output table
    num_output_value = self.accuracy.get_num_output_value()
    output_table = ML_NewTable(dimensions = [test_total, num_output_value], storage_precision = self.precision, tag = self.uniquify_name("output_table"))

    # general index for input/output tables
    table_index = 0

    test_case_list = []

    if self.auto_test_std:
      # standard test cases
      for i in range(num_std_case):
        input_list = []
        for in_id in range(self.get_arity()):
          input_value = self.get_input_precision(in_id).round_sollya_object(self.standard_test_cases[i][0], RN)
          input_list.append(input_value)
        test_case_list.append(tuple(input_list))


    # random test cases
    for i in range(test_num):
      input_list = []
      for in_id in range(self.get_arity()):
        input_value = random.uniform(low_input, high_input)
        input_value = self.precision.round_sollya_object(input_value, RN)
        input_list.append(input_value)
      test_case_list.append(tuple(input_list))

    # generating output from the concatenated list
    # of all inputs
    for table_index, input_tuple in enumerate(test_case_list):
      # storing inputs
      for in_id in range(self.get_arity()):
        input_tables[in_id][table_index] = input_tuple[in_id]
      # computing and storing output values
      output_values = self.accuracy.get_output_check_value(self, input_tuple)
      for o in range(num_output_value):
        output_table[table_index][o] = output_values[o]

    if self.implementation.get_output_format().is_vector_format():
      # vector implementation test
      test_loop = self.get_vector_test_wrapper(test_total, tested_function, input_tables, output_table)
    else: 
      # scalar implemetation test
      test_loop = self.get_scalar_test_wrapper(test_total, tested_function, input_tables, output_table)

    # common test scheme between scalar and vector functions
    test_scheme = Statement(
      test_loop,
      printf_success_function(),
      Return(Constant(0, precision = ML_Int32))
    )
    auto_test.set_scheme(test_scheme)
    return FunctionGroup([auto_test])

  ## return a FunctionObject display
  #  an error index, a list of argument values
  #  and a result value
  def get_printf_input_function(self):
    input_display_formats = ", ".join(prec.get_display_format() for prec in self.get_input_precisions())
    printf_arg_mapping = dict([
      (0, "\"error[%%d]: %s(%s), result is %s vs expected \"" % (self.function_name, input_display_formats, self.precision.get_display_format())), 
      (1, FO_Arg(0)), # error index
      (2 + self.get_arity(), FO_Arg(1 + self.get_arity())) # output
    ] + 
    [(2 + i, FO_Arg(1 + i)) for i in range(self.get_arity())] # arguments
    )
    printf_op = FunctionOperator("printf", arg_map = printf_arg_mapping, void_function = True) 
    printf_input_function = FunctionObject("printf", [ML_Int32] + self.get_input_precisions() + [self.precision], ML_Void, printf_op)
    return printf_input_function

  ## generate a test loop for vector tests
  #  @param test_num number of elementary tests to be executed
  #  @param tested_function FunctionObject to be tested
  #  @param input_tables list of ML_NewTable object containing test inputs
  #  @param output_table ML_NewTable object containing test outputs
  def get_vector_test_wrapper(self, test_num, tested_function, input_tables, output_table):
    vector_format = self.implementation.get_output_format()
    assignation_statement = Statement()
    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

    # building inputs
    local_inputs = [
      Variable(
        "vec_x_{}".format(i) , 
        precision = vector_format, 
        var_type = Variable.Local
      ) for i in range(self.get_arity())
    ]
    for input_index, local_input in enumerate(local_inputs):
      assignation_statement.push(local_input)
      for k in range(self.get_vector_size()):
        elt_assign = ReferenceAssign(VectorElementSelection(local_input, k), TableLoad(input_tables[input_index], vi + k))
        assignation_statement.push(elt_assign)

    # computing results
    local_result = tested_function(*local_inputs)
    loop_increment = self.get_vector_size()

    comp_statement = Statement()
    
    printf_input_function = self.get_printf_input_function()

    # comparison with expected
    for k in range(self.get_vector_size()):
      elt_inputs  = [VectorElementSelection(local_inputs[input_id], k) for input_id in range(self.get_arity())]
      elt_result = VectorElementSelection(local_result, k)

      output_values = [TableLoad(output_table, vi + k, i) for i in range(self.accuracy.get_num_output_value())]

      failure_test = self.accuracy.get_output_check_test(elt_result, output_values)

      comp_statement.push(
        ConditionBlock(
          failure_test,
          Statement(
            printf_input_function(*tuple([vi + k] + elt_inputs + [elt_result])), 
            self.accuracy.get_output_print_call(self.function_name, output_values),
            Return(Constant(1, precision = ML_Int32))
          )
        )
      )
    
    # common test Statement
    test_statement = Statement()

    test_loop = Loop(
      ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
      vi < test_num_cst,
      Statement(
        assignation_statement,
        comp_statement,
        ReferenceAssign(vi, vi + loop_increment)
      ),
    )

    # computing maximal error
    if self.compute_max_error:
      eval_error = Variable("max_error", precision = self.precision, var_type = Variable.Local)

      printf_error_op = FunctionOperator("printf", arg_map = {0: "\"max %s error is %s \\n \"" % (self.function_name, self.precision.get_display_format()), 1: FO_Arg(0)}, void_function = True) 
      printf_error_function = FunctionObject("printf", [self.precision], ML_Void, printf_error_op)

      local_inputs = [
        Variable(
          "vec_x_{}".format(i) , 
          precision = vector_format, 
          var_type = Variable.Local
        ) for i in range(self.get_arity())
      ]
      assignation_statement = Statement()
      for input_index, local_input in enumerate(local_inputs):
        assignation_statement.push(local_input)
        for k in range(self.get_vector_size()):
          elt_assign = ReferenceAssign(VectorElementSelection(local_input, k), TableLoad(input_tables[input_index], vi + k))
          assignation_statement.push(elt_assign)

      # computing results
      local_result = tested_function(*local_inputs)

      comp_statement = Statement()
      for k in range(self.get_vector_size()):
        elt_inputs = [VectorElementSelection(local_inputs[input_id], k) for input_id in range(self.get_arity())]
        elt_result = VectorElementSelection(local_result, Constant(k, precision = ML_Integer))

        output_values = [TableLoad(output_table, vi + k, i) for i in range(self.accuracy.get_num_output_value())]

        local_error = self.accuracy.compute_error(elt_result, output_values, relative = True)

        comp_statement.push(
          ReferenceAssign(
            eval_error,
            Max(
              local_error,
              eval_error,
              precision = self.precision
            )
          )
        )

      error_loop = Loop(
        ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
        vi < test_num_cst,
        Statement(
          assignation_statement,
          comp_statement,
          ReferenceAssign(vi, vi + loop_increment)
        ),
      )
      test_statement.add(
        Statement(
          ReferenceAssign(eval_error, Constant(0, precision = self.precision)),
          error_loop,
          printf_error_function(eval_error)
        )
      )

    # adding functional test_loop to test statement
    test_statement.add(test_loop)
    return test_statement

  ## generate a test loop for scalar tests
  #  @param test_num number of elementary tests to be executed
  #  @param tested_function FunctionObject to be tested
  #  @param input_table ML_NewTable object containing test inputs
  #  @param output_table ML_NewTable object containing test outputs
  #  @param printf_function FunctionObject to print error case
  def get_scalar_test_wrapper(self, test_num, tested_function, input_tables, output_table):
    assignation_statement = Statement()
    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")


    local_inputs  = tuple(TableLoad(input_tables[in_id], vi) for in_id in range(self.get_arity()))
    local_result = tested_function(*local_inputs)
    output_values = [TableLoad(output_table, vi, i) for i in range(self.accuracy.get_num_output_value())]

    failure_test = self.accuracy.get_output_check_test(local_result, output_values)

    printf_input_function = self.get_printf_input_function()

    printf_error_op = FunctionOperator("printf", arg_map = {0: "\"max %s error is %s \\n \"" % (self.function_name, self.precision.get_display_format()), 1: FO_Arg(0)}, void_function = True) 
    printf_error_function = FunctionObject("printf", [self.precision], ML_Void, printf_error_op)
    
    printf_max_op = FunctionOperator("printf", arg_map = {0: "\"max %s error is reached at input number %s \\n \"" % (self.function_name, "%d"), 1: FO_Arg(0)}, void_function = True) 
    printf_max_function = FunctionObject("printf", [self.precision], ML_Void, printf_max_op)

    loop_increment = self.get_vector_size()
    
    if self.break_error:
        return_statement_break = Statement(
            printf_input_function(*((vi,) + local_inputs + (local_result,))), 
            self.accuracy.get_output_print_call(self.function_name, output_values)
        )
    else:
        return_statement_break = Statement(
            printf_input_function(*((vi,) + local_inputs + (local_result,))), 
            self.accuracy.get_output_print_call(self.function_name, output_values),
            Return(Constant(1, precision = ML_Int32))
        )
    
    test_loop = Loop(
      ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
      vi < test_num_cst,
      Statement(
        assignation_statement,
        ConditionBlock(
          failure_test,
          return_statement_break,
        ),
        ReferenceAssign(vi, vi + loop_increment)
      ),
    )

    test_statement = Statement() 

    if self.compute_max_error:
      eval_error = Variable("max_error", precision = self.precision, var_type = Variable.Local)
      max_input = Variable("max_input", precision = ML_Int32, var_type = Variable.Local)
      max_result = Variable("max_result", precision = self.precision, var_type = Variable.Local)
      max_vi = Variable("max_vi", precision = ML_Int32, var_type = Variable.Local)
      local_inputs  = tuple(TableLoad(input_tables[in_id], vi) for in_id in range(self.get_arity()))

      local_result  = tested_function(*local_inputs)
      stored_values = [TableLoad(output_table, vi, i) for i in range(self.accuracy.get_num_output_value())]
      local_error = self.accuracy.compute_error(local_result, stored_values, relative = True)
      error_comp = Comparison(local_error, eval_error, specifier = Comparison.Greater, precision = ML_Bool)
      error_loop = Loop(
        ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
        vi < test_num_cst,
        Statement(
          assignation_statement,
          ConditionBlock(
            error_comp,
            Statement(
              ReferenceAssign(eval_error, local_error),
              ReferenceAssign(max_input, vi/loop_increment),
              ReferenceAssign(max_vi, vi),
              ReferenceAssign(max_result, local_result)              
            ),
            Statement()),
          ReferenceAssign(vi, vi + loop_increment)
        ),
      )
      test_statement.add(Statement(
        ReferenceAssign(eval_error, Constant(0, precision = self.precision)),
        ReferenceAssign(max_input, Constant(0, precision = ML_Int32)),
        error_loop,
        printf_error_function(eval_error),
        printf_max_function(max_input),
      ))

    # adding functional test_loop to test statement
    test_statement.add(test_loop)

    return test_statement


  ## Generate a test wrapper for the @p self function 
  #  @param test_num   number of test to perform
  #  @param test_range numeric range for test's inputs
  #  @param debug enable debug mode
  def generate_bench_wrapper(self, test_num = 10, loop_num=100000, test_range = Interval(-1.0, 1.0), debug = False):
    low_input = inf(test_range)
    high_input = sup(test_range)
    auto_test = CodeFunction("bench_wrapper", output_format = ML_Int32)

    tested_function    = self.implementation.get_function_object()
    function_name      = self.implementation.get_name()

    failure_report_op       = FunctionOperator("report_failure")
    failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)


    printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True) 
    printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

    test_total   = test_num 
    # compute the number of standard test cases
    num_std_case = len(self.standard_test_cases)
    # add them to the total if standard test enabled
    if self.auto_test_std:
      test_total += num_std_case
    # round up the number of tests to the implementation vector-size
    diff        = self.get_vector_size() - (test_total % self.get_vector_size())
    test_total += diff
    test_num   += diff

    sollya_precision = self.precision.get_sollya_object()
    interval_size = high_input - low_input 

    input_tables = [
      ML_NewTable(
        dimensions = [test_total], 
        storage_precision = self.get_input_precision(i), 
        tag = self.uniquify_name("input_table_arg%d" %i)
      )
      for i in range(self.get_arity())
    ]
    ## (low, high) are store in output table
    output_table = ML_NewTable(dimensions = [test_total], storage_precision = self.precision, tag = self.uniquify_name("output_table"), empty = True)

    # random test cases
    for i in range(test_num):
      for in_id in range(self.get_arity()):
        input_value = random.uniform(low_input, high_input)
        input_value = self.precision.round_sollya_object(input_value, RN)
        input_tables[in_id][i] = input_value

    if self.implementation.get_output_format().is_vector_format():
      # vector implementation bench
      test_loop = self.get_vector_bench_wrapper(test_num, tested_function, input_tables, output_table)
    else: 
      # scalar implemetation bench
      test_loop = self.get_scalar_bench_wrapper(test_num, tested_function, input_tables, output_table)

    timer = Variable("timer", precision = ML_Int64, var_type = Variable.Local)
    printf_timing_op = FunctionOperator(
        "printf",
        arg_map = {
            0: "\"%s %%\"PRIi64\" elts computed in %%\"PRIi64\" cycles => %%.3f CPE \\n\"" % function_name,
            1: FO_Arg(0), 2: FO_Arg(1),
            3: FO_Arg(2)
        }, void_function = True
    )
    printf_timing_function = FunctionObject("printf", [ML_Int64, ML_Int64, ML_Binary64], ML_Void, printf_timing_op)

    vj = Variable("j", precision=ML_Int32, var_type=Variable.Local)
    loop_num_cst = Constant(loop_num, precision=ML_Int32, tag="loop_num")
    loop_increment = 1

    # common test scheme between scalar and vector functions
    test_scheme = Statement(
      ReferenceAssign(timer, self.processor.get_current_timestamp()),
      Loop(
          ReferenceAssign(vj, Constant(0, precision=ML_Int32)),
          vj < loop_num_cst,
          Statement(
              test_loop,
              ReferenceAssign(vj, vj + loop_increment)
          )
      ),
      ReferenceAssign(timer,
        Subtraction(
          self.processor.get_current_timestamp(),
          timer,
          precision = ML_Int64
        )
      ),
      printf_timing_function(
        Constant(test_num * loop_num, precision = ML_Int64),
        timer,
        Division(
          Conversion(timer, precision = ML_Binary64),
          Constant(test_num * loop_num, precision = ML_Binary64),
          precision = ML_Binary64
        )
      ),
      Return(Constant(0, precision = ML_Int32))
    )
    auto_test.set_scheme(test_scheme)
    return FunctionGroup([auto_test])


  ## generate a test loop for vector tests
  #  @param test_num number of elementary tests to be executed
  #  @param tested_function FunctionObject to be tested
  #  @param input_table ML_NewTable object containing test inputs
  #  @param output_table ML_NewTable object containing test outputs
  def get_vector_bench_wrapper(self, test_num, tested_function, input_tables, output_table):
    vector_format = self.implementation.get_output_format()
    assignation_statement = Statement()
    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

    # building inputs
    local_inputs = [
      Variable(
        "vec_x_{}".format(i) , 
        precision = vector_format, 
        var_type = Variable.Local
      ) for i in range(self.get_arity())
    ]
    for input_index, local_input in enumerate(local_inputs):
      assignation_statement.push(local_input)
      for k in range(self.get_vector_size()):
        elt_assign = ReferenceAssign(VectorElementSelection(local_input, k), TableLoad(input_tables[input_index], vi + k))
        assignation_statement.push(elt_assign)

    # computing results
    local_result = tested_function(*local_inputs)
    loop_increment = self.get_vector_size()

    store_statement = Statement()

    # comparison with expected
    for k in range(self.get_vector_size()):
      elt_result = VectorElementSelection(local_result, k)

      # TODO: change to use aligned linear vector store
      store_statement.push(
        TableStore(elt_result, output_table, vi + k, precision = ML_Void) 
      )

    test_loop = Loop(
      ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
      vi < test_num_cst,
      Statement(
        assignation_statement,
        store_statement,
        ReferenceAssign(vi, vi + loop_increment)
      ),
    )
    return test_loop

  ## generate a bench loop for scalar tests
  #  @param test_num number of elementary tests to be executed
  #  @param tested_function FunctionObject to be tested
  #  @param input_tables list of ML_NewTable object containing test inputs
  #  @param output_table ML_NewTable object containing test outputs
  def get_scalar_bench_wrapper(self, test_num, tested_function, input_tables, output_table):
    assignation_statement = Statement()
    vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
    test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

    local_inputs  = tuple(TableLoad(input_tables[in_id], vi) for in_id in range(self.get_arity()))
    local_result = tested_function(*local_inputs)

    loop_increment = 1

    test_loop = Loop(
      ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
      vi < test_num_cst,
      Statement(
        TableStore(local_result, output_table, vi, precision = ML_Void),
        ReferenceAssign(vi, vi + loop_increment)
      ),
    )
    return test_loop

  #@staticmethod
  def get_name(self):
    return self.function_name

  # list of input to be used for standard test validation
  standard_test_cases = []


## Function class builder to build ML_FunctionBasis
#  child class with specific function_name value
def ML_Function(name):
  new_class = type(name, (ML_FunctionBasis,), {"function_name": name})
  return new_class

# end of Doxygen's ml_function group
## @}

