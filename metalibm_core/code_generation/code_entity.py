# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Nov 17th, 2016
# last-modified:    Nov 17th, 2016
#
# author(s):   Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: Implementation of the class for wrapping
#              the content and interfaces (code) of an architectural 
#              entity
###############################################################################

from ..core.ml_operations import AbstractVariable, Variable, FunctionObject, Statement, ReferenceAssign
from ..core.ml_hdl_operations import Signal, ComponentObject
from .code_object import NestedCode
from .generator_utility import FunctionOperator, FO_Arg
from .code_constant import *
from ..core.attributes import Attributes, AttributeCtor


class CodeEntity(object):
  """ function code object """
  def __init__(self, name, arg_list = None, output_list = None, code_object = None, language = VHDL_Code):
    """ code function initialization """
    self.name = name
    self.arg_list = arg_list if arg_list else []
    self.arg_map = dict((arg.get_tag(), arg) for arg in self.arg_list)
    self.output_list = output_list if output_list else []
    self.code_object = code_object
    self.entity_object   = None
    self.entity_operator = None
    self.language = language
    self.process_list = []
    self.current_stage = 0
    # component object to generate external instance of entity
    self.component_object = None
    # attribute to contain thestage where the pipelined
    # signal was originally created
    self.init_stage_attribute = AttributeCtor("init_stage", default_value = 0)
    # attribute to contain the original operation for the pipelined signals
    self.init_op_attribute    = AttributeCtor("init_op", default_value = None) 
    Attributes.add_dyn_attribute(self.init_stage_attribute)
    Attributes.add_dyn_attribute(self.init_op_attribute)

  def get_name(self):
    return self.name

  def add_input_variable(self, name, vartype):
    input_var = Variable(name, precision = vartype) 
    self.arg_list.append(input_var)
    self.arg_map[name] = input_var
    return input_var
  def add_output_variable(self, name, output_node):
    output_var = Variable(name, precision = output_node.get_precision(), var_type = Variable.Output)
    output_assign = ReferenceAssign(output_var, output_node)
    self.output_list.append(output_assign)

  def add_input_signal(self, name, signaltype):
    input_signal = Signal(name, precision = signaltype) 
    self.arg_list.append(input_signal)
    self.arg_map[name] = input_signal
    return input_signal
  def add_output_signal(self, name, output_node):
    output_var = Signal(name, precision = output_node.get_precision(), var_type = Signal.Output)
    output_assign = ReferenceAssign(output_var, output_node)
    self.output_list.append(output_assign)

  def get_input_by_tag(self, tag):
    if tag in self.arg_map:
      return self.arg_map[tag]
    else:
      return None

  def get_output_list(self):
    return [op.get_input(1) for op in self.output_list]

  def start_new_stage(self):
    self.set_current_stage(self.current_stage + 1)

  def set_current_stage(self, stage_id = 0):
    self.current_stage = stage_id
    self.init_stage_attribute.default_value = self.current_stage

  def add_process(self, new_process):
    self.process_list.append(new_process)
  def register_new_input_variable(self, new_input):
    self.arg_list.append(new_input)

  def get_arg_list(self):
    return self.arg_list
  def clear_arg_list(self):
    self.arg_list = []

  def get_component_object(self):
    if self.component_object is None:
      self.component_object = self.build_component_object()
    return self.component_object

  def build_component_object(self):
    io_map = {}
    for arg_input in self.arg_list:
      io_map[arg_input] = AbstractVariable.Input 
    for arg_output in self.get_output_list():
      io_map[arg_output] = AbstractVariable.Output 
    return ComponentObject(self.name, io_map, self)
    
  def get_declaration(self, final = True, language = None):
    language = self.language if language is None else language
    # input signal declaration
    input_port_list = ["%s : in %s" % (inp.get_tag(), inp.get_precision().get_code_name(language = language)) for inp in self.arg_list]
    output_port_list = ["%s : out %s" % (out.get_input(0).get_tag(), out.get_input(0).get_precision().get_code_name(language = language)) for out in self.output_list]
    port_format_list = ";\n  ".join(input_port_list + output_port_list)
    # FIXME: add suport for inout and generic
    return "entity {entity_name} is \nport (\n  {port_list}\n);\nend {entity_name};\n\n".format(entity_name = self.name, port_list = port_format_list)

  def get_component_declaration(self, final = True, language = None):
    language = self.language if language is None else language
    # input signal declaration
    input_port_list = ["%s : in %s" % (inp.get_tag(), inp.get_precision().get_code_name(language = language)) for inp in self.arg_list]
    output_port_list = ["%s : out %s" % (out.get_input(0).get_tag(), out.get_input(0).get_precision().get_code_name(language = language)) for out in self.output_list]
    port_format_list = ";\n  ".join(input_port_list + output_port_list)
    # FIXME: add suport for inout and generic
    return "component {entity_name} \nport (\n  {port_list}\n);\nend component;\n\n".format(entity_name = self.name, port_list = port_format_list)

  ## @return function implementation (ML_Operation DAG)
  def get_scheme(self):
    return Statement(*tuple(self.process_list + self.output_list))

  def get_definition(self, code_generator, language, folded = True, static_cst = False):
    code_object = NestedCode(code_generator, static_cst = static_cst, code_ctor = VHDLCodeObject)
    code_object << self.get_declaration(final = False, language = language)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = False, language = language)
    code_object.close_level()
    return code_object

  def add_definition(self, code_generator, language, code_object, folded = True, static_cst = False):
    code_object << self.get_declaration(final = False, language = language)
    code_object << "architecture rtl of {entity_name} is\n".format(entity_name = self.name)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.get_scheme(), folded = folded, initial = False, language = language)
    code_object.close_level()
    code_object << "end architecture;\n"
    return code_object
