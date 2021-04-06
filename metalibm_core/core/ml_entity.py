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
# created:          Nov 17th, 2016
# last-modified:    Mar  8th, 2018
#
# author(s):   Nicolas Brunie (nbrunie@kalray.eu)
# decription:  Declare and implement a class to manage
#              hardware (vhdl like) entities
###############################################################################


import sollya

from sollya import Interval

S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import (
    Variable,
    Statement, ReferenceAssign, Constant, Comparison, ConditionBlock,
    WhileLoop,
    LogicalNot, Conversion, TypeCast,
    FunctionObject,
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Signal, Wait, Report, Concatenation, Assert,
    multi_Concatenation,
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Bool, ML_String, ML_FP_Format, ML_Fixed_Format,
    ML_Void,
)

from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat, ML_StdLogic,
    is_fixed_point,
    HDL_FILE, HDL_LINE, HDL_OPEN_FILE_STATUS,
)

from metalibm_core.code_generation.code_object import (
    NestedCode, VHDLCodeObject, CodeObject, MultiSymbolTable
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Arg
)
from metalibm_core.code_generation.code_entity import CodeEntity
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.code_generation.vhdl_code_generator import VHDLCodeGenerator
from metalibm_core.code_generation.code_constant import VHDL_Code

from metalibm_core.core.passes import (
    PassScheduler, PassDependency, Pass, AfterPassById
)

from metalibm_core.code_generation.gappa_code_generator import (
    GappaCodeGenerator
)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.rtl_debug_utils import rtl_debug_multi
from metalibm_core.utility.ml_template import (
    ArgDefault, DefaultEntityArgTemplate
)

from metalibm_core.opt.p_pipelining import generate_pipeline_stage
from metalibm_core.opt.p_tag_node import Pass_DebugTaggedNode

import random
import subprocess

from metalibm_core.core.random_gen import get_precision_rng

def generate_random_fp_value(precision, inf, sup):
    """ Generate a random floating-point value of format precision """
    assert isinstance(precision, ML_FP_Format)
    random_exp = S2**random.randrange(precision.get_emin_normal(), 1)
    value = random.uniform(0.5, 1.0) * random_exp * (sup - inf) + inf
    rounded_value = precision.round_sollya_object(value, sollya.RN)
    return rounded_value

def generate_random_fixed_value(precision):
    """ Generate a random fixed-point value of format precision """
    assert is_fixed_point(precision)
    # fixed point format
    lo_value = precision.get_min_value()
    hi_value = precision.get_max_value()
    value = random.uniform( lo_value, hi_value)
    rounded_value = precision.round_sollya_object(value)
    return rounded_value

class RawLogicVectorRandomGen:
    """ Random number generator for raw logic/logic_vector
        datum """
    def __init__(self, size, min_value=0, max_value=None):
        self.size = size
        self.min_value = min_value
        self.max_value = 2**size-1 if max_value is None else max_value
    def get_new_value(self):
        # use size to generate a logarithmic distribution
        sub_size = random.randrange(0, self.size+1)
        return max(min(random.randrange(2**sub_size), self.max_value), self.min_value)


def get_hdl_precision_rng(input_precision, input_range=None):
    """ build a random number generator object for format <input_precision>
        with values within interval <input_range> """
    if isinstance(input_precision, ML_StdLogicVectorFormat):
        # default min_value for RawLogicVectorRandomGen
        # must be zero
        min_value = 0
        max_value = None
        if not input_range is None:
            min_value = int(inf(input_range))
            max_value = int(sup(input_range))
        return RawLogicVectorRandomGen(input_precision.get_bit_size(), min_value=min_value, max_value=max_value) 
    else:
        # after processing hdl-specific format we can fallback
        # on the generic get_precision_rng method 
        return get_precision_rng(input_precision, input_range)

## \defgroup ml_entity ml_entity
## @{



# Helper function for test case implementations
def get_input_assign(input_signal, input_value):
    """ Get input assignation statement """
    input_assign = ReferenceAssign(
        input_signal,
        Constant(input_value, precision=input_signal.get_precision())
    )
    return input_assign


def get_input_msg(input_tag, input_signal, input_value):
    """ generate input debug message """
    value_msg = input_signal.get_precision().get_cst(
      input_value, language = VHDL_Code
    ).replace('"',"'")
    value_msg += " / " + hex(
      input_signal.get_precision().get_base_format().get_integer_coding(input_value)
    )
    return " {}={} ".format(input_tag, value_msg)


def get_output_check_statement(output_signal, output_tag, output_value):
    """ Generate output value check statement """
    test_pass_cond = Comparison(
        output_signal,
        output_value,
        specifier=Comparison.Equal,
        precision=ML_Bool
    )

    check_statement = ConditionBlock(
        LogicalNot(
            test_pass_cond,
            precision = ML_Bool
        ),
        Report(
            Concatenation(
                " result for {}: ".format(output_tag),
                Conversion(
                    output_signal if output_signal.get_precision() is ML_StdLogic else
                    TypeCast(
                        output_signal,
                        precision=ML_StdLogicVectorFormat(
                            output_signal.get_precision().get_bit_size()
                        )
                     ),
                    precision = ML_String
                    ),
                precision = ML_String
            )
        )
    )
    return test_pass_cond, check_statement

def get_output_value_msg(output_signal, output_value):
    """ generate message describing expected output value """
    output_precision = output_signal.get_precision()
    expected_dec = output_precision.get_cst(output_value, language=VHDL_Code).replace('"',"'")
    expected_hex = " / " + hex(output_precision.get_base_format().get_integer_coding(output_value))
    value_msg = "{} / {}".format(expected_dec, expected_hex)
    return value_msg



def signal_str_conversion(optree, op_format):
    """ converision of @p optree from op_format to ML_String """
    return Conversion(
        optree if op_format is ML_StdLogic else
        TypeCast(
            optree,
            precision=ML_StdLogicVectorFormat(
                op_format.get_bit_size()
            )
         ),
        precision=ML_String
    )



# utility tcl function to display fixed-point value
debug_utils_lib = """proc get_fixed_value {value weight} {
  return [expr $value * pow(2.0, $weight)]
}\n"""


class HDLSimulation:
    def __init__(self, source_file_list):
        self.source_file_list = source_file_list

    def elab(self, entity_name=None):
        raise NotImplementedError

    def run(self, simulation_time, debug=False, exit_after_test=True):
        raise NotImplementedError

    def elab_and_run(self, simulation_time, debug=False, exit_after_test=True):
        elab_result = self.elab()
        if elab_result:
            return elab_result
        sim_result = self.run(simulation_time, debug=debug, exit_after_test=exit_after_test)
        return sim_result

class ModelsimSimulation(HDLSimulation):
    def __init__(self, source_file_list, debug_file=None):
        HDLSimulation.__init__(self, source_file_list)
        self.debug_file = debug_file
    def elab(self, entity_name=None):
        # TODO/FIXME: a single file supported for now
        output_file = self.source_file_list[0]
        modelsim_elab_cmd = "vlib work && vcom -2008 {}".format(output_file)
        Log.report(Log.Info, "elaboration command:\n{}".format(modelsim_elab_cmd))
        elab_result = subprocess.call(modelsim_elab_cmd, shell=True)
        Log.report(Log.Info, "elaboration result:{}".format(elab_result))
        return elab_result

    def run(self, simulation_time, debug=False, exit_after_test=True):
        debug_cmd = "do {debug_file};".format(debug_file=self.debug_file) if debug else "" 
        debug_cmd += " exit;" if exit_after_test else ""
        # simulation
        modelsim_run_cmd = "vsim -c work.testbench -do \"run {test_delay} ns; {debug_cmd}\"".format(
            debug_cmd=debug_cmd, test_delay=simulation_time)
        Log.report(Log.Info, "simulation command:\n{}".format(modelsim_run_cmd))
        sim_result = subprocess.call(modelsim_run_cmd, shell=True)
        Log.report(Log.Info, "simulation result:{}".format(sim_result))
        return sim_result

class GHDLSimulation(HDLSimulation):
    def elab(self, entity_name):
        # TODO/FIXME: a single file supported for now
        output_file = self.source_file_list[0]
        ghdl_elab_cmd = "ghdl -c --ieee=synopsys --std=08 {} -e {} ".format(output_file, entity_name)
        Log.report(Log.Info, "elaboration command:\n{}".format(ghdl_elab_cmd))
        elab_result = subprocess.call(ghdl_elab_cmd, shell=True)
        Log.report(Log.Info, "elaboration result:{}".format(elab_result))
        return elab_result


    def elab_and_run(self, simulation_time, debug=False, exit_after_test=True):
        output_file = self.source_file_list[0]
        ghdl_elab_run_cmd = "ghdl -c --ieee=synopsys --std=08 {} -r testbench --stop-time={}ns ".format(output_file, simulation_time)
        Log.report(Log.Info, "simulation command:\n{}".format(ghdl_elab_run_cmd))
        sim_result = subprocess.call(ghdl_elab_run_cmd, shell=True)
        Log.report(Log.Info, "simulation result:{}".format(sim_result))
        return sim_result



## Base class for all metalibm function (metafunction)
class ML_EntityBasis(object):
  name = "entity_basis"

  ## constructor
  #  @param base_name string function name (without precision considerations)
  #  @param function_name
  #  @param output_file string name of source code output file
  #  @param debug_file string name of debug script output file
  #  @param io_precisions input/output ML_Format list
  #  @param libm_compliant boolean flag indicating whether or not the function
  #                        should be compliant with standard libm specification
  #                        (wrt exception, error ...)
  #  @param fast_path_extract boolean flag indicating whether or not fast
  #                           path extraction optimization must be applied
  #  @param debug_flag boolean flag, indicating whether or not debug code
  #                    must be generated
  def __init__(self,
             # Naming
             base_name = ArgDefault("unknown_entity", 2),
             entity_name= ArgDefault(None, 2),
             output_file = ArgDefault(None, 2),
             # Specification
             io_precisions = ArgDefault([ML_Binary32], 2),
             libm_compliant = ArgDefault(True, 2),
             # Optimization parameters
             backend = ArgDefault(VHDLBackend(), 2),
             fast_path_extract = ArgDefault(True, 2),
             # Debug verbosity
             debug_flag = ArgDefault(False, 2),
             language = ArgDefault(VHDL_Code, 2),
             arg_template = DefaultEntityArgTemplate
         ):
    # selecting argument values among defaults
    base_name = ArgDefault.select_value([base_name])
    Log.report(Log.Info, "pre entity_name: %s %s " % (entity_name, arg_template.entity_name))
    entity_name = ArgDefault.select_value([arg_template.entity_name, entity_name])
    Log.report(Log.Info, "entity_name: %s " % entity_name)
    Log.report(Log.Info, "output_file: %s %s " % (arg_template.output_file, output_file))
    Log.report(Log.Info, "debug_file:  %s "% arg_template.debug_file)
    output_file = ArgDefault.select_value([arg_template.output_file, output_file])
    debug_file  = arg_template.debug_file
    # Specification
    io_precisions = ArgDefault.select_value([io_precisions])
    # Optimization parameters
    backend = ArgDefault.select_value([arg_template.backend, backend])
    fast_path_extract = ArgDefault.select_value([arg_template.fast_path_extract, fast_path_extract])
    # Debug verbosity
    debug_flag    = ArgDefault.select_value([arg_template.debug, debug_flag])
    language      = ArgDefault.select_value([arg_template.language, language])
    auto_test     = arg_template.auto_test
    auto_test_std = arg_template.auto_test_std

    self.precision = arg_template.precision
    self.io_formats = arg_template.io_formats
    self.pipelined = arg_template.pipelined

    # io_precisions must be a list
    #     -> with a single element
    # XOR -> with as many elements as function arity (input + output arities)
    self.io_precisions = io_precisions


    ## enable the generation of numeric/functionnal auto-test
    self.auto_test_enable  = (auto_test != False or auto_test_std != False)
    self.auto_test_number  = auto_test
    self.auto_test_range   = arg_template.auto_test_range
    self.auto_test_std     = auto_test_std
    # embedded test in behavior or externalize inputs/expected in data file
    self.externalized_test_data = arg_template.externalized_test_data

    # enable/disable automatic exit once functional test is finished
    self.exit_after_test   = arg_template.exit_after_test

    # enable post-generation RTL elaboration
    self.build_enable = arg_template.build_enable
    # enable post-elaboration simulation
    self.execute_trigger = arg_template.execute_trigger

    self.simulator = arg_template.simulator

    self.language = language

    # Naming logic, using provided information if available, otherwise deriving from base_name
    # base_name is e.g. exp
    # entity_name is e.g. expf or expd or whatever
    self.entity_name = entity_name if entity_name else generic_naming(base_name, self.io_precisions)

    self.output_file = output_file if output_file else self.entity_name + ".vhd"
    self.debug_file  = debug_file  if debug_file  else "{}_dbg.do".format(self.entity_name)

    # debug version
    self.debug_flag = debug_flag
    # debug display
    self.display_after_gen = arg_template.display_after_gen
    self.display_after_opt = arg_template.display_after_opt

    self.decorate_code=arg_template.decorate_code


    # target selection
    self.backend = backend

    # register control
    self.reset_pipeline, self.synchronous_reset = arg_template.reset_pipeline
    self.negate_reset = arg_template.negate_reset
    self.reset_name = arg_template.reset_name
    self.recirculate_pipeline = arg_template.recirculate_pipeline


    # optimization parameters
    self.fast_path_extract = fast_path_extract

    self.implementation = CodeEntity(self.entity_name)

    self.vhdl_code_generator = VHDLCodeGenerator(self.backend, declare_cst = False, disable_debug = not self.debug_flag, language = self.language, decorate_code=self.decorate_code)
    uniquifier = self.entity_name
    self.main_code_object = NestedCode(
        self.vhdl_code_generator, static_cst=False,
        uniquifier="{0}_".format(self.entity_name),
        code_ctor=VHDLCodeObject,
        shared_symbol_list=[MultiSymbolTable.EntitySymbol, MultiSymbolTable.ProtectedSymbol]
    )
    if self.debug_flag:
      self.debug_code_object = CodeObject(self.language)
      self.debug_code_object << debug_utils_lib
      self.vhdl_code_generator.set_debug_code_object(self.debug_code_object)

    # pass scheduler instanciation
    self.pass_scheduler = PassScheduler()
    # recursive pass dependency
    pass_dep = PassDependency()
    Log.report(Log.Verbose, "extra_passes: {}".format(arg_template.extra_passes))
    for pass_uplet in arg_template.passes + arg_template.extra_passes:
      pass_slot_tag, pass_tag = pass_uplet.split(":")
      pass_slot = PassScheduler.get_tag_class(pass_slot_tag)
      pass_class  = Pass.get_pass_by_tag(pass_tag)
      pass_object = pass_class(self.backend)
      self.pass_scheduler.register_pass(pass_object, pass_dep = pass_dep, pass_slot = pass_slot)
      # linearly linking pass in the order they appear
      pass_dep = AfterPassById(pass_object.get_pass_id())

    Log.report(Log.LogLevel("DumpPassInfo"), self.pass_scheduler.dump_pass_info())

    # TODO/FIXME: can be overloaded
    if  self.reset_pipeline:
        self.reset_signal = self.implementation.add_input_signal(self.reset_name, ML_StdLogic)
    self.recirculate_signal_map = dict((index, self.implementation.add_input_signal(v, ML_StdLogic)) for index, v in arg_template.recirculate_signal_map.items())

  def get_pass_scheduler(self):
    return self.pass_scheduler


  ## Class method to generate a structure containing default arguments
  #  which must be overloaded by @p kw
  @staticmethod
  def get_default_args(**kw):
    return DefaultEntityArgTemplate(**kw)

  def get_implementation(self):
    """ return meta-entity CodeEntity object """
    return self.implementation

  ## name generation
  #  @param base_name string, name to be extended for unifiquation
  def uniquify_name(self, base_name):
    """ return a unique identifier, combining base_name + function_name """
    return "%s_%s" % (self.function_name, base_name)

  ## emulation code generation
  def generate_emulate(self):
    raise NotImplementedError


  # try to extract 'clk' input or create it if
  # it does not exist
  def get_clk_input(self):
    clk_in = self.implementation.get_input_by_tag("clk")
    if not clk_in is None:
      return clk_in
    else:
      return self.implementation.add_input_signal('clk', ML_StdLogic)


  def get_output_precision(self):
    return self.io_precisions[0]

  def get_input_precision(self):
    return self.io_precisions[-1]

  def get_io_format(self, tag):
    return self.io_formats[tag]


  def get_sollya_precision(self):
    """ return the main precision use for sollya calls """
    return self.sollya_precision

  def generate_interfaces(self):
    """ Generate entity interfaces """
    raise NotImplementedError

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

  def get_recirculate_signal(self, stage_id):
    """ generate / retrieve the signal used to recirculate
        register at pipeline stage <stage_id> """
    try:
        return self.recirculate_signal_map[stage_id]
    except KeyError as e:
        Log.report(Log.Error, "stage {} has no associated recirculation signal in recirculate_signal_map", stage_id, error=e)


  def is_main_entity(self, entity):
    """ Overloadable predicate to determine which sub entities should be
        considered principal """
    return True

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

    # list of ComponentObject which are entities on which self depends
    common_entity_list = []
    generated_entity = []

    self.result = code_object
    code_str = ""
    while len(code_entity_list) > 0:
      code_entity = code_entity_list.pop(0)
      if code_entity in generated_entity:
        continue
      entity_code_object = NestedCode(
        self.vhdl_code_generator, static_cst=False,
        uniquifier="{0}_".format(self.entity_name),
        code_ctor=VHDLCodeObject,
        shared_symbol_list=[MultiSymbolTable.EntitySymbol, MultiSymbolTable.ProtectedSymbol],
      )
      if self.is_main_entity(code_entity):
        self.vhdl_code_generator.disable_debug = not self.debug_flag
      else:
        self.vhdl_code_generator.disable_debug = True
      result = code_entity.add_definition(self.vhdl_code_generator, language, entity_code_object, static_cst = False)
      result.add_library("ieee")
      result.add_header("ieee.std_logic_1164.all")
      result.add_header("ieee.std_logic_arith.all")
      result.add_header("ieee.std_logic_misc.all")
      result.add_header("STD.textio.all")
      result.add_header("ieee.std_logic_textio.all")
      code_str += result.get(self.vhdl_code_generator, headers = True)

      generated_entity.append(code_entity)

      # adding the entities encountered during code generation
      # for future generation
      # TODO: document
      extra_entity_list = [
            comp_object.get_code_entity() for comp_object in
                entity_code_object.get_entity_list()
      ]
      Log.report(Log.Info, "appending {} extra entit(y/ies)\n".format(len(extra_entity_list)))
      code_entity_list += extra_entity_list

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


  def gen_implementation(self,
			display_after_gen = False,
			display_after_opt = False,
			enable_subexpr_sharing = True
		):
    ## apply @p pass_object optimization pass
    #  to the scheme of each entity in code_entity_list
    def entity_execute_pass(scheduler, pass_object, code_entity_list):
      for code_entity in code_entity_list:
        entity_scheme = code_entity.get_scheme()
        processed_scheme = pass_object.execute(entity_scheme)
        # todo check pass effect
        # code_entity.set_scheme(processed_scheme)
      return code_entity_list

    # generate scheme
    code_entity_list = self.generate_entity_list()

    Log.report(Log.Info, "Applying passes at start of generation")
    code_entity_list = self.pass_scheduler.get_full_execute_from_slot(
      code_entity_list,
      PassScheduler.Start,
      entity_execute_pass
    )

    # defaulting pipeline stage to None
    self.implementation.set_current_stage(None)

    Log.report(Log.Info, "Applying passes just before pipelining")
    code_entity_list = self.pass_scheduler.get_full_execute_from_slot(
      code_entity_list,
      PassScheduler.BeforePipelining,
      entity_execute_pass
    )

    if self.pipelined:
        self.stage_num = generate_pipeline_stage(
            self, reset=self.reset_pipeline,
            recirculate=self.recirculate_pipeline,
            synchronous_reset=self.synchronous_reset,
            negate_reset=self.negate_reset)
    else:
        self.stage_num = 1
    Log.report(Log.Info, "there is/are {} pipeline stage(s)".format(self.stage_num))

    Log.report(Log.Info, "Applying passes just after pipelining")
    code_entity_list = self.pass_scheduler.get_full_execute_from_slot(
      code_entity_list,
      PassScheduler.AfterPipelining,
      entity_execute_pass
    )

    # debug instrumentation pass: enable debug for all nodes whose tag is listed
    # in self.debug_flag
    debug_pass = Pass_DebugTaggedNode(self.backend, self.debug_flag, debug_mapping=rtl_debug_multi)
    _ = self.pass_scheduler.execute_pass_list(
       [debug_pass],
       code_entity_list,
       entity_execute_pass)

    # stage duration (in ns)
    time_step = 10

    if self.auto_test_enable:
      code_entity_list += self.generate_auto_test(
				test_num = self.auto_test_number if self.auto_test_number else 0,
				test_range = self.auto_test_range,
                time_step = time_step
			)


    for code_entity in code_entity_list:
      scheme = code_entity.get_scheme()
      if display_after_gen or self.display_after_gen:
        print("function %s, after gen " % code_entity.get_name())
        print(scheme.get_str(depth = None, display_precision = True, memoization_map = {}))

      # optimize scheme
      opt_scheme = self.optimise_scheme(scheme, enable_subexpr_sharing = enable_subexpr_sharing)

      if display_after_opt or self.display_after_opt:
        print("function %s, after opt " % code_entity.get_name())
        print(scheme.get_str(depth = None, display_precision = True, memoization_map = {}))


    Log.report(Log.Info, "Applying passes just before codegen")
    code_entity_list = self.pass_scheduler.get_full_execute_from_slot(
      code_entity_list,
      PassScheduler.JustBeforeCodeGen,
      entity_execute_pass
    )

    # generate VHDL code to implement scheme
    self.generate_code(code_entity_list, language = self.language)

    if self.simulator == "vsim":
        simulation = ModelsimSimulation([self.output_file], self.debug_file)
    elif self.simulator == "ghdl":
        simulation = GHDLSimulation([self.output_file])
    else:
        Log.report(Log.Error, "unknown RTL elab and simulation tool: {}", self.simulator)

    if self.execute_trigger:
      test_delay = time_step * (self.stage_num + 2) * (self.auto_test_number + (len(self.standard_test_cases) if self.auto_test_std else 0) + 100) 
      sim_result = simulation.elab_and_run(test_delay, debug=self.debug_flag, exit_after_test=self.exit_after_test)
      # rtl elaboration
      if sim_result:
        Log.report(Log.Error, "simulation failed [{}]".format(sim_result))
      else:
        Log.report(Log.Info, "simulation success")

    elif self.build_enable:
      elab_result = simulation.elab(self.entity_name)
      if elab_result:
        Log.report(Log.Error, "failed to elaborate [{}]".format(elab_result))
      else:
        Log.report(Log.Info, "elaboration success")


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

  def generate_test_case(self, input_signals, io_map, index, test_range=None):
    """ generic test case generation: generate a random input
        with index @p index

        Args:
            index (int): integer index of the test case

        Returns:
            dict: mapping (input tag -> numeric value)
    """
    # extracting test interval boundaries
    input_values = {}
    for input_tag in input_signals:
        input_value = self.input_generators[input_tag].get_new_value()
        # registering input value
        input_values[input_tag] = input_value
    return input_values

  def init_test_generator(self, io_map, test_range=None):
    """ Generic initialization of test case generator """
    # reset input generators map
    self.input_generators = {}
    # the following map is a copy of the input subset of io_map
    # with direct input Signal (no copies)
    input_signals = self.extract_input_signal_map(io_map)
    for input_tag in input_signals:
        input_signal = io_map[input_tag]
        input_precision = input_signal.get_precision().get_base_format()
        input_test_range = None if (test_range is None or not input_tag in test_range) else test_range[input_tag]
        input_generator = get_hdl_precision_rng(input_precision, input_test_range)
        self.input_generators[input_tag] = input_generator


  def implement_test_case(self, io_map, input_values, output_signals,
                          output_values, time_step, index=None):
      """ Implement the test case check and assertion whose I/Os values
          are described in input_values and output_values dict

          :param index: index of the corresponding test case (to be displayed as info) 
      """
      test_statement = Statement()
      input_msg = ""
      # Adding input setting
      for input_tag in input_values:
        input_signal = io_map[input_tag]
        # FIXME: correct value generation depending on signal precision
        input_value = input_values[input_tag]
        test_statement.add(get_input_assign(input_signal, input_value))
        input_msg += get_input_msg(input_tag, input_signal, input_value)

      test_statement.add(Wait(time_step * (self.stage_num + 2)))

      # Adding output value comparison
      for output_tag in output_signals:
        output_signal = output_signals[output_tag]
        output_value = output_values[output_tag]
        output_cst_value  = Constant(output_value, precision=output_signal.get_precision())

        value_msg = get_output_value_msg(output_signal, output_value)
        test_pass_cond, check_statement = get_output_check_statement(output_signal, output_tag, output_cst_value)

        test_statement.add(check_statement)
        assert_statement = Assert(
          test_pass_cond,
          "\"unexpected value for {test_id}, inputs {input_msg}, output {output_tag}, expecting {value_msg}, got: \"".format(input_msg = input_msg, output_tag = output_tag, value_msg = value_msg, test_id=("" if index is None else "test #{}".format(index))),
          severity = Assert.Failure
        )
        # pretty print: generate a message that can be used as a debug use-case
        test_statement.add(Report("standard test-case: ({}, None)".format(input_values)))
        test_statement.add(assert_statement)
      return test_statement

  def generate_input_signal_map(self, io_map):
    # map of input_tag -> input_signal, excludind commodity signals
    # (e.g. clock and reset)
    input_signals = {}
    reduced_arg_list = self.implementation.get_arg_list()
    for input_port in reduced_arg_list:
      input_tag = input_port.get_tag()
      input_signal = Signal(input_tag + "_i", precision = input_port.get_precision(), var_type = Signal.Local)
      io_map[input_tag] = input_signal
      # excluding clk, reset and recirculate signals
      if not input_tag in ["clk", self.reset_name] + [sig.get_tag() for sig in self.recirculate_signal_map.values()]:
        input_signals[input_tag] = input_signal
    return input_signals

  def extract_input_signal_map(self, io_map):
    """ extract input map from I/O map """
    input_signals = {}
    reduced_arg_list = self.implementation.get_arg_list()
    for input_port in reduced_arg_list:
      input_tag = input_port.get_tag()
      # excluding clk, reset and recirculate signals
      if not input_tag in ["clk", self.reset_name] + [sig.get_tag() for sig in self.recirculate_signal_map.values()]:
        input_signals[input_tag] = io_map[input_tag]
    return input_signals

  def generate_output_signal_map(self, io_map):
    """ generate map of output signals """
    # map of output_tag -> output_signal
    output_signals = {}
    # excluding clock and reset signals from argument list
    for output_port in self.implementation.get_output_port():
      output_tag = output_port.get_tag()
      output_signal = Signal(
        output_tag + "_o",
        precision=output_port.get_precision(),
        var_type=Signal.Local
      )
      io_map[output_tag] = output_signal
      output_signals[output_tag] = output_signal
    return output_signals


  def generate_datafile_testbench(self, tc_list, io_map, input_signals, output_signals, time_step, test_fname="test.input"):
    """ Generate testbench with input and output data externalized in
        a data file """
    # textio function to read hexadecimal text
    def FCT_HexaRead_gen(input_format):
        legalized_input_format = input_format
        FCT_HexaRead = FunctionObject("hread", [HDL_LINE, legalized_input_format], ML_Void, FunctionOperator("hread", void_function=True, arity=2))
        return FCT_HexaRead
    # textio function to read binary text
    FCT_Read = FunctionObject("read", [HDL_LINE, ML_StdLogic], ML_Void, FunctionOperator("read", void_function=True, arity=2))
    input_line = Variable("input_line", precision=HDL_LINE, var_type=Variable.Local)

    # building ordered list of input and output signal names
    input_signal_list = [sname for sname in input_signals.keys()]
    input_statement = Statement()
    for input_name in input_signal_list:
        input_format = input_signals[input_name].precision
        input_var = Variable(
            "v_" + input_name,
            precision=input_format,
            var_type=Variable.Local)
        if input_format is ML_StdLogic:
            input_statement.add(FCT_Read(input_line, input_var))
        else:
            input_statement.add(FCT_HexaRead_gen(input_format)(input_line, input_var))
        input_statement.add(ReferenceAssign(input_signals[input_name], input_var))

    output_signal_list = [sname for sname in output_signals.keys()]
    output_statement = Statement()
    for output_name in output_signal_list:
        output_format = output_signals[output_name].precision
        output_var = Variable(
            "v_" + output_name,
            precision=output_format,
            var_type=Variable.Local)
        if output_format is ML_StdLogic:
            output_statement.add(FCT_Read(input_line, output_var))
        else:
            output_statement.add(FCT_HexaRead_gen(output_format)(input_line, output_var))

        output_signal = output_signals[output_name]
        #value_msg = get_output_value_msg(output_signal, output_value)
        test_pass_cond, check_statement = get_output_check_statement(output_signal, output_name, output_var)

        input_msg = multi_Concatenation(*tuple(sum([[" %s=" % input_tag, signal_str_conversion(input_signals[input_tag], input_signals[input_tag].precision)] for input_tag in input_signal_list], [])))

        output_statement.add(check_statement)
        assert_statement = Assert(
            test_pass_cond,
            multi_Concatenation(
                "unexpected value for inputs ",
                input_msg,
                " expecting :",
                signal_str_conversion(output_var, output_format),
                " got :",
                signal_str_conversion(output_signal, output_format),
               precision = ML_String
            ),
            severity=Assert.Failure
        )
        output_statement.add(assert_statement)

    self_component = self.implementation.get_component_object()
    self_instance = self_component(io_map = io_map, tag = "tested_entity")
    test_statement = Statement()

    DATA_FILE_NAME = test_fname

    with open(DATA_FILE_NAME, "w") as data_file:
        # dumping column tags
        data_file.write("# " + " ".join(input_signal_list + output_signal_list) + "\n")

        def get_raw_cst_string(cst_format, cst_value):
            size = int((cst_format.get_bit_size() + 3) / 4)
            return ("{:x}").format(cst_format.get_base_format().get_integer_coding(cst_value)).zfill(size)

        for input_values, output_values in tc_list:
            # TODO; generate test data file
            cst_list = []
            for input_name in input_signal_list:
                input_value = input_values[input_name]
                input_format = input_signals[input_name].get_precision()
                cst_list.append(get_raw_cst_string(input_format, input_value))

            for output_name in output_signal_list:
                output_value = output_values[output_name]
                output_format = output_signals[output_name].get_precision()
                cst_list.append(get_raw_cst_string(output_format, output_value))
            # dumping line into file
            data_file.write(" ".join(cst_list) + "\n")

    input_stream = Variable("data_file", precision=HDL_FILE, var_type=Variable.Local)
    file_status = Variable("file_status", precision=HDL_OPEN_FILE_STATUS, var_type=Variable.Local)
    FCT_EndFile = FunctionObject("endfile", [HDL_FILE], ML_Bool, FunctionOperator("endfile", arity=1)) 
    FCT_OpenFile = FunctionObject(
        "FILE_OPEN", [HDL_OPEN_FILE_STATUS, HDL_FILE, ML_String], ML_Void,
        FunctionOperator(
            "FILE_OPEN",
            arg_map={0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2), 3: "READ_MODE"},
            void_function=True))
    FCT_ReadLine =  FunctionObject(
        "readline", [HDL_FILE, HDL_LINE], ML_Void,
        FunctionOperator("readline", void_function=True, arity=2))

    reset_statement = self.get_reset_statement(io_map, time_step)
    OPEN_OK = Constant("OPEN_OK", precision=HDL_OPEN_FILE_STATUS)

    testbench = CodeEntity("testbench")
    test_process = Process(
        reset_statement,
        FCT_OpenFile(file_status, input_stream, DATA_FILE_NAME),
        ConditionBlock(
            Comparison(file_status, OPEN_OK, specifier=Comparison.NotEqual, precision=ML_Bool),
          Assert(
            Constant(0, precision=ML_Bool),
            " \"failed to open file {}\"".format(DATA_FILE_NAME),
            severity=Assert.Failure
          )
        ),
        # consume legend line
        FCT_ReadLine(input_stream, input_line),
        WhileLoop(
            LogicalNot(FCT_EndFile(input_stream)),
            Statement(
                FCT_ReadLine(input_stream, input_line),
                input_statement,
                Wait(time_step * (self.stage_num + 2)),
                output_statement,
            ),
        ),
      # end of test
      Assert(
        Constant(0, precision = ML_Bool),
        " \"end of test, no error encountered \"",
        severity = Assert.Warning
      ),
      # infinite end loop
        WhileLoop(
            Constant(1, precision=ML_Bool),
            Statement(
                Wait(time_step * (self.stage_num + 2)),
            )
        )
    )

    testbench_scheme = Statement(
      self_instance,
      test_process
    )

    if self.pipelined:
        half_time_step = time_step / 2
        assert (half_time_step * 2) == time_step
        # adding clock process for pipelined bench
        clk_process = Process(
            Statement(
                ReferenceAssign(
                    io_map["clk"],
                    Constant(1, precision = ML_StdLogic)
                ),
                Wait(half_time_step),
                ReferenceAssign(
                    io_map["clk"],
                    Constant(0, precision = ML_StdLogic)
                ),
                Wait(half_time_step),
            )
        )
        testbench_scheme.push(clk_process)

    testbench.add_process(testbench_scheme)

    return [testbench]

  def generate_auto_test(self, test_num=10, test_range=None, debug=False, time_step=10):
    """ time_step: duration of a stage (in ns) """
    # instanciating tested component
    # map of input_tag -> input_signal and output_tag -> output_signal
    io_map = {}

    # map of input_tag -> input_signal, excludind commodity signals
    # (e.g. clock and reset)
    input_signals = self.generate_input_signal_map(io_map)
    # map of output_tag -> output_signal
    output_signals = self.generate_output_signal_map(io_map)

    # building list of test cases
    tc_list = []


    # initializing random test case generator
    self.init_test_generator(io_map, test_range)

    # Appending standard test cases if required
    if self.auto_test_std:
      tc_list += self.standard_test_cases

    for i in range(test_num):
      input_values = self.generate_test_case(input_signals, io_map, i)
      tc_list.append((input_values,None))

    def compute_results(tc):
        """ update test case with output values if required """
        input_values, output_values = tc
        if output_values is None:
            return input_values, self.numeric_emulate(input_values)
        else:
            return tc

    # filling output values
    tc_list = [compute_results(tc) for tc in tc_list]
    if self.externalized_test_data:
        return self.generate_datafile_testbench(tc_list, io_map, input_signals, output_signals, time_step, test_fname=self.externalized_test_data)
    else:
        return self.generate_embedded_testbench(tc_list, io_map, input_signals, output_signals, time_step)

  def get_reset_statement(self, io_map, time_step):
    reset_statement = Statement()
    if self.reset_pipeline:
        # TODO: fix pipeline register reset
        reset_value = 0 if self.negate_reset else 1
        unreset_value = 1 - reset_value
        reset_signal = io_map[self.reset_name]
        reset_statement.add(ReferenceAssign(reset_signal, Constant(reset_value, precision=ML_StdLogic)))
        # to account for synchronous reset
        reset_statement.add(Wait(time_step * 3))
        reset_statement.add(ReferenceAssign(reset_signal, Constant(unreset_value, precision=ML_StdLogic)))
        reset_statement.add(Wait(time_step * 3))
        for recirculate_signal in self.recirculate_signal_map.values():
            reset_statement.add(ReferenceAssign(io_map[recirculate_signal.get_tag()], Constant(0, precision=ML_StdLogic)))
    return reset_statement

  def generate_embedded_testbench(self, tc_list, io_map, input_signals, output_signals, time_step, test_fname="test.input"):
    """ Generate testbench with embedded input and output data """
    self_component = self.implementation.get_component_object()
    self_instance = self_component(io_map = io_map, tag = "tested_entity")
    test_statement = Statement()

    for index, (input_values, output_values) in enumerate(tc_list):
      test_statement.add(
          self.implement_test_case(io_map, input_values, output_signals, output_values, time_step, index=index)
      )

    reset_statement = self.get_reset_statement(io_map, time_step)

    testbench = CodeEntity("testbench")
    test_process = Process(
      reset_statement,
      test_statement,
      # end of test
      Assert(
        Constant(0, precision = ML_Bool),
        " \"end of test, no error encountered \"",
        severity = Assert.Warning
      ),
      # infinite end loop
        WhileLoop(
            Constant(1, precision=ML_Bool),
            Statement(
                Wait(time_step * (self.stage_num + 2)),
            )
        )
    )

    testbench_scheme = Statement(
      self_instance,
      test_process
    )

    if self.pipelined:
        half_time_step = time_step / 2
        assert (half_time_step * 2) == time_step
        # adding clock process for pipelined bench
        clk_process = Process(
            Statement(
                ReferenceAssign(
                    io_map["clk"],
                    Constant(1, precision = ML_StdLogic)
                ),
                Wait(half_time_step),
                ReferenceAssign(
                    io_map["clk"],
                    Constant(0, precision = ML_StdLogic)
                ),
                Wait(half_time_step),
            )
        )
        testbench_scheme.push(clk_process)

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

# end of Doxygen's ml_entity group
## @}

