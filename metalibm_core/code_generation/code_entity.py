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

from ..core.ml_operations import Variable, FunctionObject, Statement, ReferenceAssign
from ..core.ml_hdl_operations import Signal
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
    self.output_list = output_list if output_list else []
    self.code_object = code_object
    self.entity_object   = None
    self.entity_operator = None
    self.language = language
    self.process_list = []
    self.current_stage = 0
    self.init_stage_attribute = AttributeCtor("init_stage", default_value = 0)
    Attributes.add_dyn_attribute(self.init_stage_attribute)

  def get_name(self):
    return self.name

  def add_input_variable(self, name, vartype):
    input_var = Variable(name, precision = vartype) 
    self.arg_list.append(input_var)
    return input_var
  def add_output_variable(self, name, output_node):
    output_var = Variable(name, precision = output_node.get_precision(), var_type = Variable.Output)
    output_assign = ReferenceAssign(output_var, output_node)
    self.output_list.append(output_assign)

  def add_input_signal(self, name, signaltype):
    input_signal = Signal(name, precision = signaltype) 
    self.arg_list.append(input_signal)
    return input_signal
  def add_output_signal(self, name, output_node):
    output_var = Signal(name, precision = output_node.get_precision(), var_type = Signal.Output)
    output_assign = ReferenceAssign(output_var, output_node)
    self.output_list.append(output_assign)

  def start_new_stage(self):
    self.set_current_stage(self.current_stage + 1)

  def set_current_stage(self, stage_id = 0):
    self.current_stage = stage_id
    self.init_stage_attribute.default_value = self.current_stage
    print "current_stage: ", self.current_stage, self.init_stage_attribute, self.init_stage_attribute.default_value

  def add_process(self, new_process):
    self.process_list.append(new_process)
  def register_new_input_variable(self, new_input):
    self.arg_list.append(new_input)

  def get_arg_list(self):
    return self.arg_list
  def clear_arg_list(self):
    self.arg_list = []

  def get_function_object(self):
    # if None, build it
    if self.function_object is None:
      self.function_object = self.build_function_object()
    return self.function_object

  def build_function_object(self):
    arg_list_precision = [arg.get_precision() for arg in self.arg_list]
    return FunctionObject(self.name, arg_list_precision, self.output_format, self.get_function_operator())

  def get_function_operator(self):
    return self.build_function_operator()

  def build_function_operator(self):
    function_arg_map = {}
    for i in xrange(len(self.arg_list)):
      function_arg_map[i] = FO_Arg(i)
    return FunctionOperator(self.name, arg_map = function_arg_map)

    
  def get_declaration(self, final = True, language = None):
    language = self.language if language is None else language
    # input signal declaration
    input_port_list = ["%s : in %s" % (inp.get_tag(), inp.get_precision().get_name(language = language)) for inp in self.arg_list]
    output_port_list = ["%s : out %s" % (out.get_input(0).get_tag(), out.get_input(0).get_precision().get_name(language = language)) for out in self.output_list]
    port_format_list = ";\n  ".join(input_port_list + output_port_list)
    # FIXME: add suport for inout and generic
    return "entity {entity_name} is \nport (\n  {port_list}\n);\nend {entity_name};\n\n".format(entity_name = self.name, port_list = port_format_list)

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
