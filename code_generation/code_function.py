# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Feb  1st, 2016
# last-modified:    Feb  5th, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..core.ml_operations import Variable, FunctionObject
from .code_object import NestedCode
from .generator_utility import FunctionOperator, FO_Arg


class CodeFunction:
  """ function code object """
  def __init__(self, name, arg_list = None, output_format = None, code_object = None):
    """ code function initialization """
    self.name = name
    self.arg_list = arg_list if arg_list else []
    self.code_object = code_object
    self.output_format = output_format 
    self.function_object   = None
    self.function_operator = None

  def get_name(self):
    return self.name

  def add_input_variable(self, name, vartype):
    input_var = Variable(name, precision = vartype) 
    self.arg_list.append(input_var)
    return input_var

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

  ## retrieve format of the result(s) returned by the function
  #  @return ML_Format object
  def get_output_format(self):
    return self.output_format
  ## define a new format for the function return value(s)
  #  @param new_output_format ML_Format object indicated which format is returned by the function
  def set_output_format(self, new_output_format):
    self.output_format = new_output_format
    

  def get_declaration(self, final = True):
    arg_format_list = ", ".join("%s %s" % (inp.get_precision().get_c_name(), inp.get_tag()) for inp in self.arg_list)
    final_symbol = ";" if final else ""
    return "%s %s(%s)%s" % (self.output_format.get_c_name(), self.name, arg_format_list, final_symbol)

  ## define function implementation
  #  @param scheme ML_Operation object to be defined as function implementation
  def set_scheme(self, scheme):
    self.scheme = scheme
  ## @return function implementation (ML_Operation DAG)
  def get_scheme(self):
    return self.scheme

  def get_definition(self, code_generator, language, folded = True, static_cst = False):
    code_object = NestedCode(code_generator, static_cst = static_cst)
    code_object << self.get_declaration(final = False)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = False)
    code_object.close_level()
    return code_object

  def add_definition(self, code_generator, language, code_object, folded = True, static_cst = False):
    code_object << self.get_declaration(final = False)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = False)
    code_object.close_level()
    return code_object
