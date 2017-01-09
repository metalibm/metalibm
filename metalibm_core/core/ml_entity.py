# -*- coding: utf-8 -*-

###############################################################################
# This file is part of New Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Nov 17th, 2016    
# last-modified:    Nov 17th, 2016
#
# author(s):   Nicolas Brunie (nibrunie@gmail.com)
# decription:  Declare and implement a class to manage
#              hardware (vhdl like) entities
###############################################################################

from sollya import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import *  
from metalibm_core.core.ml_hdl_operations import *  
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.ml_call_externalizer import CallExternalizer
from metalibm_core.core.ml_vectorizer import StaticVectorizer

from metalibm_core.code_generation.code_object import NestedCode, VHDLCodeObject, CodeObject
from metalibm_core.code_generation.code_entity import CodeEntity
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.code_generation.vhdl_code_generator import VHDLCodeGenerator
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.code_generation.generator_utility import *

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.ml_template import ArgDefault

import random
import subprocess

## \defgroup ml_function ml_function
## @{


# return a random value in the given @p interval
# Samplin is done uniformly on value exponent, 
# not on the value itself
def random_log_sample(interval):
  lo = inf(interval)
  hi = sup(interval)


  
## default argument template to be used when no specific value
#  are given for a specific parameter
class DefaultEntityArgTemplate:
  def __init__(self, 
                base_name = "unknown_entity",
                entity_name = None,
                output_file = None,
                debug_file  = None,
                # Specification,
                precision = ML_Binary32,
                accuracy = ML_Faithful,
                io_precisions = [ML_Binary32],
                abs_accuracy   = None,
                libm_compliant = True,
                # Optimization parameters,
                backend = VHDLBackend(),
                fuse_fma = True,
                fast_path_extract = True,
                # Debug verbosity,
                debug= False,
                language = VHDL_Code,
                auto_test = False,
                auto_test_execute = False,
                auto_test_range = None,
                auto_test_std   = False,
                **kw # extra arguments
              ):
    self.base_name  = base_name
    self.entity_name  = entity_name
    self.output_file  = output_file
    self.debug_file   = debug_file
    # Specification,
    self.precision  = precision
    self.io_precisions  = io_precisions
    self.abs_accuracy  = abs_accuracy
    self.accuracy      = accuracy
    self.libm_compliant  = libm_compliant
    # Optimization parameters,
    self.backend  = backend
    self.fuse_fma  = fuse_fma
    self.fast_path_extract  = fast_path_extract
    # Debug verbosity,
    self.debug = debug
    self.language  = language
    self.auto_test  = auto_test
    self.auto_test_execute  = auto_test_execute
    self.auto_test_range  = auto_test_range
    self.auto_test_std  = auto_test_std
    # registering extra arguments
    for attr in kw:
      print "initializing: ", attr, kw[attr]
      setattr(self, attr, kw[attr])

class RetimeMap:
  def __init__(self):
    # map (op_key, stage) -> stage's op
    self.stage_map = {}
    # map of stage_index -> list of pipelined forward 
    # from <stage_index> -> <stage_index + 1>
    self.stage_forward = {}
    # list of nodes already retimed
    self.processed = []
    # 
    self.pre_statement = set()

  def get_op_key(self, op):
    op_key = op.attributes.init_op if not op.attributes.init_op is None else op
    return op_key

  def hasBeenProcessed(self, op):
    return self.get_op_key(op)  in self.processed
  def addToProcessed(self, op):
    op_key = self.get_op_key(op)
    return self.processed.append(op_key)

  def contains(self, op, stage):
    return (self.get_op_key(op), stage) in self.stage_map

  def get(self, op, stage):
    return self.stage_map[(self.get_op_key(op), stage)]
  def set(self, op, stage):
    op_key = self.get_op_key(op)
    self.stage_map[(op_key, stage)] = op

  def add_stage_forward(self, op_dst, op_src, stage):
    Log.report(Log.Verbose, " adding stage forward {op_src} to {op_dst} @ stage {stage}".format(op_src = op_src, op_dst = op_dst, stage = stage))
    if not stage in self.stage_forward:
      self.stage_forward[stage] = []
    self.stage_forward[stage].append(
      ReferenceAssign(op_dst, op_src)
    )
    self.pre_statement.add(op_src)

## Base class for all metalibm function (metafunction)
class ML_EntityBasis(object):
  name = "entity_basis"

  ## constructor
  #  @param base_name string function name (without precision considerations)
  #  @param function_name 
  #  @param output_file string name of source code output file
  #  @param debug_file string name of debug script output file
  #  @param io_precisions input/output ML_Format list
  #  @param abs_accuracy absolute accuracy
  #  @param libm_compliant boolean flag indicating whether or not the function should be compliant with standard libm specification (wrt exception, error ...)
  #  @param processor GenericProcessor instance, target of the implementation
  #  @param fuse_fma boolean flag indicating whether or not fusing Multiply+Add optimization must be applied
  #  @param fast_path_extract boolean flag indicating whether or not fast path extraction optimization must be applied
  #  @param debug_flag boolean flag, indicating whether or not debug code must be generated 
  def __init__(self,
             # Naming
             base_name = ArgDefault("unknown_entity", 2),
             entity_name= ArgDefault(None, 2),
             output_file = ArgDefault(None, 2),
             debug_file  = ArgDefault(None, 2),
             # Specification
             io_precisions = ArgDefault([ML_Binary32], 2), 
             abs_accuracy = ArgDefault(None, 2),
             libm_compliant = ArgDefault(True, 2),
             # Optimization parameters
             backend = ArgDefault(VHDLBackend(), 2),
             fuse_fma = ArgDefault(True, 2), 
             fast_path_extract = ArgDefault(True, 2),
             # Debug verbosity
             debug_flag = ArgDefault(False, 2),
             language = ArgDefault(VHDL_Code, 2),
             auto_test = ArgDefault(False, 2),
             auto_test_range = ArgDefault(Interval(-1, 1), 2),
             auto_test_std = ArgDefault(False, 2),
             arg_template = DefaultEntityArgTemplate 
         ):
    # selecting argument values among defaults
    base_name = ArgDefault.select_value([base_name])
    print "pre entity_name: ", entity_name, arg_template.entity_name
    entity_name = ArgDefault.select_value([arg_template.entity_name, entity_name])
    print "entity_name: ", entity_name
    print "output_file: ", arg_template.output_file, output_file 
    print "debug_file:  ", arg_template.debug_file, debug_file 
    output_file = ArgDefault.select_value([arg_template.output_file, output_file])
    debug_file  = ArgDefault.select_value([arg_template.debug_file, debug_file])
    # Specification
    io_precisions = ArgDefault.select_value([io_precisions])
    abs_accuracy = ArgDefault.select_value([abs_accuracy])
    # Optimization parameters
    backend = ArgDefault.select_value([arg_template.backend, backend])
    fuse_fma = ArgDefault.select_value([arg_template.fuse_fma, fuse_fma])
    fast_path_extract = ArgDefault.select_value([arg_template.fast_path_extract, fast_path_extract])
    # Debug verbosit
    debug_flag    = ArgDefault.select_value([arg_template.debug, debug_flag])
    language      = ArgDefault.select_value([arg_template.language, language])
    auto_test     = ArgDefault.select_value([arg_template.auto_test, arg_template.auto_test_execute, auto_test])
    auto_test_std = ArgDefault.select_value([arg_template.auto_test_std, auto_test_std])


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
    # entity_name is e.g. expf or expd or whatever 
    self.entity_name = entity_name if entity_name else generic_naming(base_name, self.io_precisions)

    self.output_file = output_file if output_file else self.entity_name + ".vhd"
    self.debug_file  = debug_file  if debug_file  else "{}_dbg.do".format(self.entity_name)

    self.debug_flag = debug_flag

    # TODO: FIX which i/o precision to select
    # TODO: incompatible with fixed-point formats
    # self.sollya_precision = self.get_output_precision().get_sollya_object()

    self.abs_accuracy = abs_accuracy if abs_accuracy else S2**(-self.get_output_precision().get_precision())

    self.backend = backend

    self.fuse_fma = fuse_fma
    self.fast_path_extract = fast_path_extract

    self.implementation = CodeEntity(self.entity_name)

    self.vhdl_code_generator = VHDLCodeGenerator(self.backend, declare_cst = False, disable_debug = not self.debug_flag, language = self.language)
    uniquifier = self.entity_name
    self.main_code_object = NestedCode(self.vhdl_code_generator, static_cst = True, uniquifier = "{0}_".format(self.entity_name), code_ctor = VHDLCodeObject)
    if self.debug_flag:
      self.debug_code_object = CodeObject(self.language)
      self.vhdl_code_generator.set_debug_code_object(self.debug_code_object)


  def get_implementation(self):
    return self.implementation

  ## name generation
  #  @param base_name string, name to be extended for unifiquation
  def uniquify_name(self, base_name):
    """ return a unique identifier, combining base_name + function_name """
    return "%s_%s" % (self.function_name, base_name)

  ## emulation code generation
  def generate_emulate(self):
    raise NotImplementedError


  ## propagate forward @p op until it is defined
  #  in @p stage
  def propagate_op(self, op, stage, retime_map):
    op_key = retime_map.get_op_key(op)
    Log.report(Log.Verbose, " propagating {op} (key={op_key}) to stage {stage}".format(op = op, op_key = op_key, stage = stage))
    # look for the latest stage where op is defined
    current_stage = op_key.attributes.init_stage
    while retime_map.contains(op_key, current_stage + 1):
      current_stage += 1
    op_src = retime_map.get(op_key, current_stage)
    while current_stage != stage:  
      # create op instance for <current_stage+1>
      op_dst = Signal(tag = "{tag}_S{stage}".format(tag = op_key.get_tag(), stage = (current_stage + 1)), init_stage = current_stage + 1, init_op = op_key, precision = op_key.get_precision()) 
      retime_map.add_stage_forward(op_dst, op_src, current_stage)
      retime_map.set(op_dst, current_stage + 1)
      # update values for next iteration
      current_stage += 1
      op_src = op_dst
      

  # process op's inputs and if necessary
  # propagate them to op's stage
  def retime_op(self, op, retime_map):
    Log.report(Log.Verbose, "retiming op %s " % (op))
    if retime_map.hasBeenProcessed(op):
      return
    op_stage = op.attributes.init_stage
    if not isinstance(op, ML_LeafNode):
      for in_id in range(op.get_input_num()):
        in_op = op.get_input(in_id)
        in_stage = in_op.attributes.init_stage
        Log.report(Log.Verbose, "retiming input {inp} of {op} stage {in_stage} -> {op_stage}".format(inp = in_op, op = op, in_stage = in_stage, op_stage = op_stage))
        if not retime_map.hasBeenProcessed(in_op):
          self.retime_op(in_op, retime_map)
        if in_stage < op_stage:
          if not retime_map.contains(in_op, op_stage):
            self.propagate_op(in_op, op_stage, retime_map)
          new_in = retime_map.get(in_op, op_stage)
          Log.report(Log.Verbose, "new version of input {inp} for {op} is {new_in}".format(inp = in_op, op = op, new_in = new_in))
          op.set_input(in_id, new_in)
        elif in_stage > op_stage:
          Log.report(Log.Error, "input {inp} of {op} is defined at a later stage".format(inp = in_op, op = op))
    retime_map.set(op, op_stage)
    retime_map.addToProcessed(op)
        
  # try to extract 'clk' input or create it if 
  # it does not exist
  def get_clk_input(self):
    clk_in = self.implementation.get_input_by_tag("clk")
    if not clk_in is None:
      return clk_in
    else:
      return self.implementation.add_input_signal('clk', ML_StdLogic)


  def generate_pipeline_stage(self):
    retiming_map = {}
    retime_map = RetimeMap()
    output_list = self.implementation.get_output_list()
    for output in output_list:
      Log.report(Log.Verbose, "generating pipeline from output %s " % (output))
      self.retime_op(output, retime_map)
    process_statement = Statement()

    # adding stage forward process
    clk = self.get_clk_input()
    for stage_id in sorted(retime_map.stage_forward.keys()):
      stage_block = ConditionBlock(
        Event(clk, precision = ML_Bool),
        Statement(*tuple(assign for assign in retime_map.stage_forward[stage_id]))
      )
      process_statement.add(stage_block)
    pipeline_process = Process(process_statement, sensibility_list = [clk])
    for op in retime_map.pre_statement:
      pipeline_process.add_to_pre_statement(op)
    self.implementation.add_process(pipeline_process)
      

  def get_output_precision(self):
    return self.io_precisions[0]

  def get_input_precision(self):
    return self.io_precisions[-1]


  def get_sollya_precision(self):
    """ return the main precision use for sollya calls """
    return self.sollya_precision


  def generate_scheme(self):
    """ generate MDL scheme for function implementation """
    Log.report(Log.Error, "generate_scheme must be overloaded by ML_EntityBasis child")

  ## 
  # @return main_scheme, [list of sub-CodeFunction object]
  def generate_entity_list(self):
    return self.generate_scheme()

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
    return scheme


  ## 
  #  @return main code object associted with function implementation
  def get_main_code_object(self):
    return self.main_code_object


  ## generate VHDL code for entity implenetation 
  #  Code is generated within the main code object
  #  and dumped to a file named after implementation's name
  #  @param code_function_list list of CodeFunction to be generated (as sub-function )
  #  @return void
  def generate_code(self, code_entity_list, language = VHDL_Code):
    """ Final VHDL generation, once the evaluation scheme has been optimized"""
    # registering scheme as function implementation
    #self.implementation.set_scheme(scheme)
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object
    code_str = ""
    for code_entity in code_entity_list:
      entity_code_object = NestedCode(self.vhdl_code_generator, static_cst = True, uniquifier = "{0}_".format(self.entity_name), code_ctor = VHDLCodeObject)
      result = code_entity.add_definition(self.vhdl_code_generator, language, entity_code_object, static_cst = True)
      result.add_library("ieee")
      result.add_header("ieee.std_logic_1164.all")
      result.add_header("ieee.std_logic_unsigned.all")
      result.add_header("ieee.numeric_std.all")
      #result.push_into_parent_code(self.result, self.vhdl_code_generator, headers = True)
      code_str += result.get(self.vhdl_code_generator, headers = True)

    # adding headers
    #self.result.add_header("ieee.std_logic_1164.all")
    #self.result.add_header("ieee.std_logic_unsigned.all")
    #self.result.add_header("ieee.numeric_std.all")

    Log.report(Log.Verbose, "Generating VHDL code in " + self.output_file)
    output_stream = open(self.output_file, "w")
    output_stream.write(code_str)
    output_stream.close()
    if self.debug_flag:
      Log.report(Log.Verbose, "Generating Debug code in {}".format(self.debug_file))
      debug_code_str = self.debug_code_object.get(None)
      debug_stream = open(self.debug_file, "w")
      debug_stream.write(debug_code_str)
      debug_stream.close()


  def gen_implementation(self, display_after_gen = False, display_after_opt = False, enable_subexpr_sharing = True):
    # generate scheme
    code_entity_list = self.generate_entity_list()
    
    self.generate_pipeline_stage()

    if self.auto_test_enable:
      code_entity_list += self.generate_auto_test(test_num = self.auto_test_number if self.auto_test_number else 0, test_range = self.auto_test_range)
      

    for code_entity in code_entity_list:
      scheme = code_entity.get_scheme()
      if display_after_gen:
        print "function %s, after gen " % code_entity.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

      # optimize scheme
      opt_scheme = self.optimise_scheme(scheme, enable_subexpr_sharing = enable_subexpr_sharing)

      if display_after_opt:
        print "function %s, after opt " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})


    # generate VHDL code to implement scheme
    self.generate_code(code_entity_list, language = self.language)

    if self.auto_test_enable:
      pass


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
    # extracting test interval boundaries
    low_input = inf(test_range)
    high_input = sup(test_range)
    # instanciating tested component
    io_map = {}
    input_signals = {}
    output_signals = {}
    # excluding clock and reset signals from argument list
    # reduced_arg_list = [input_port for input_port in self.implementation.get_arg_list() if not input_port.get_tag() in ["clk", "reset"]]
    reduced_arg_list = self.implementation.get_arg_list()
    for input_port in reduced_arg_list:
      input_tag = input_port.get_tag()
      input_signal = Signal(input_tag + "_i", precision = input_port.get_precision(), var_type = Signal.Local)
      io_map[input_tag] = input_signal
      if not input_tag in ["clk", "reset"]:
        input_signals[input_tag] = input_signal
    for output_port in self.implementation.get_output_port():
      output_tag = output_port.get_tag()
      output_signal = Signal(output_tag + "_o", precision = output_port.get_precision(), var_type = Signal.Local)
      io_map[output_tag] = output_signal
      output_signals[output_tag] = output_signal

    self_component = self.implementation.get_component_object()
    self_instance = self_component(io_map = io_map, tag = "tested_entity")

    test_statement = Statement()


    # building list of test cases
    tc_list = []
    for i in range(test_num):
      input_values = {}
      for input_tag in input_signals:
        input_signal = io_map[input_tag]
        # FIXME: correct value generation depending on signal precision
        input_precision = input_signal.get_precision().get_base_format()
        input_size = input_precision.get_bit_size()
        # input_value = random.uniform(low_input, high_input)
        low_input_exp = int(floor(log2(abs(low_input))))
        high_input_exp = int(floor(log2(abs(high_input))))
        if isinstance(input_precision, ML_FP_Format):
          input_value = random.uniform(0.5, 1.0) * S2**random.randrange(input_precision.get_emin_normal(), 1) * (high_input - low_input) + low_input
          input_value = round(input_value, input_precision.get_sollya_object(), RN)
        else: 
          input_value = random.randrange(2**input_precision.get_bit_size())
        print("input_value %e" % input_value)
        input_values[input_tag] = input_value
      tc_list.append((input_values,None))

    # Appending standard test cases if required
    if self.auto_test_std:
      tc_list += self.standard_test_cases 

    for input_values, output_values in tc_list:
      # Adding input setting
      for input_tag in input_values:
        input_signal = io_map[input_tag]
        # FIXME: correct value generation depending on signal precision
        input_value = input_values[input_tag]
        test_statement.add(ReferenceAssign(input_signal, Constant(input_value, precision = input_signal.get_precision())))
      test_statement.add(Wait(10))
      # Computin output values when necessary
      if output_values is None:
        output_values = self.numeric_emulate(input_values)
      # Adding output value comparison
      for output_tag in output_signals:
        output_signal = output_signals[output_tag]
        output_value  = Constant(output_values[output_tag], precision = output_signal.get_precision())
        value_msg = output_signal.get_precision().get_cst(output_values[output_tag], language = VHDL_Code).replace('"',"'")
        test_statement.add(
          Assert(
            Comparison(
              output_signal, 
              output_value, 
              specifier = Comparison.Equal, 
              precision = ML_Bool
            ),
            "unexpected value for output %s, expecting %s " % (output_tag, value_msg),
            severity = Assert.Failure
          )
        )


    testbench = CodeEntity("testbench") 
    test_process = Process(
      test_statement,
      # end of test
      Assert(
        Constant(0, precision = ML_Bool),
        " end of test, no error encountered",
        severity = Assert.Failure
      )
    )

    testbench_scheme = Statement(
      self_instance,
      test_process
    )

    testbench.add_process(testbench_scheme)

    return [testbench]

  @staticmethod
  def get_name():
    return ML_EntityBasis.function_name

  # list of input to be used for standard test validation
  standard_test_cases = []


def ML_Entity(name):
  new_class = type(name, (ML_EntityBasis,), {"entity_name": name})
  new_class.get_name = staticmethod(lambda: name) 
  return new_class

# end of Doxygen's ml_function group
## @}

