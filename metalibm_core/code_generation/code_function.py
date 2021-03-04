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
# created:          Feb  1st, 2016
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..core.ml_operations import Variable, FunctionObject, FunctionType
from .code_object import NestedCode
from .generator_utility import FunctionOperator, FO_Arg
from .code_constant import *
from ..core.ml_formats import ML_Void


class CodeFunction(object):
  """ function code object """
  def __init__(self, name, arg_list=None, output_format=None, code_object=None, language=C_Code, attributes=None, external=False, vector_size=1, function_object=None, function_operator=None):
    """ code function initialization """
    self.arg_list = arg_list if arg_list else []
    arg_list_precision = [arg.get_precision() for arg in self.arg_list]
    self.function_type = FunctionType(name, arg_list_precision, output_format, attributes)
    self.code_object = code_object
    # FunctionObject and FunctionOperator (can be left by default to be built
    # automatically by build_function_object and build_function_operator)
    self.function_object   = function_object
    self.function_operator = function_operator
    self.language = language
    # is the function externally defined (does not contain any implementation)
    self.external = external
    self.scheme = None
    # is the function scalar of SIMD-vector
    self.vector_size = vector_size

  @property
  def name(self):
    return self.function_type.name
  @property
  def output_format(self):
    return self.function_type.output_format
  @property
  def attributes(self):
    return self.function_type.attributes

  def get_name(self):
    return self.name

  def add_input_variable(self, name, vartype, **kw):
    """ declares a new Variable with name @p name and format @p vartype
        and registers it as an input variable """
    input_var = Variable(name, precision = vartype, **kw)
    self.arg_list.append(input_var)
    # WARNING: self.function_type.arg_list_precision is not updated
    return input_var

  def register_new_input_variable(self, new_input):
    self.arg_list.append(new_input)
    # WARNING: self.function_type.arg_list_precision is not updated

  def clear_arg_list(self):
    """ remove all registered arguments """
    self.arg_list = []

  def get_function_object(self):
    # if None, build it
    if self.function_object is None:
      self.function_object = self.build_function_object()
    return self.function_object

  def build_function_object(self):
    arg_list_precision = [arg.get_precision() for arg in self.arg_list]
    return FunctionObject(self.name, arg_list_precision, self.output_format, self.get_function_operator(), self.attributes)

  def get_function_operator(self):
    return self.build_function_operator()

  def build_function_operator(self):
    function_arg_map = {}
    for i in range(len(self.arg_list)):
      function_arg_map[i] = FO_Arg(i)
    return FunctionOperator(self.name, arg_map = function_arg_map, void_function=(self.output_format is ML_Void))

  ## retrieve format of the result(s) returned by the function
  #  @return ML_Format object
  def get_output_format(self):
    return self.output_format
  ## define a new at for the function return value(s)
  #  @param new_output_format ML_Format object indicated which format is returned by the function
  def set_output_format(self, new_output_format):
    self.function_type.output_format = new_output_format

  def add_attribute(self, attribute):
    assert not attribute in self.attributes
    self.attributes.append(attribute)

  def get_attributes_dec(self, language=C_Code):
    """ generate function attribute string """
    if self.attributes:
        return " ".join(self.attributes)
    return ""

  def get_LLVM_definition(self, final=True, language=LLVM_IR_Code):
    # TODO: support attributes and metadata
    arg_format_list = ", ".join("%s %s" % (inp.get_precision().get_name(language = language), inp.get_tag()) for inp in self.arg_list)
    return "define %s @%s(%s)" % (self.output_format.get_name(language = language), self.name, arg_format_list)

  def update_arg_list_precisions(self):
    self.function_type.arg_list_precision = [arg.precision for arg in self.arg_list]

  def get_declaration(self, code_generator, final=True, language=None, named_arg_list=False, is_definition=False):
    """
        :param self:
        :param for_definition: indicate if the declaration is a definition prolog or a true declaration
        :type for_definition: bool
    """
    self.update_arg_list_precisions()
    language = self.language if language is None else language
    if is_definition:
        return code_generator.get_function_definition(self.function_type, final, language, arg_list=(self.arg_list if named_arg_list else None))
    else:
        # pure declaration
        return code_generator.get_function_declaration(self.function_type, final, language, arg_list=(self.arg_list if named_arg_list else None))
    #self.name, self.output_format, self.arg_list, final, language
    #)


  ## define function implementation
  #  @param scheme ML_Operation object to be defined as function implementation
  def set_scheme(self, scheme):
    self.scheme = scheme
  ## @return function implementation (ML_Operation DAG)
  def get_scheme(self):
    return self.scheme

  def get_definition(self, code_generator, language, folded = True, static_cst = False):
    code_object = NestedCode(code_generator, static_cst = static_cst)
    code_object << self.get_declaration(code_generator, final=False, language=language, named_arg_list=True, is_definition=True)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = True, language = language)
    code_object.close_level()
    return code_object

  def add_definition(self, code_generator, language, code_object, folded = True, static_cst = False):
    code_object << self.get_declaration(code_generator, final=False, language=language, named_arg_list=True, is_definition=True)
    code_object.open_level()
    code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = True, language = language)
    code_object.close_level()
    return code_object

  def add_declaration(self, code_generator, language, code_object):
    code_object << self.get_declaration(code_generator, final=True, language=language) +"\n"
    return code_object

class FunctionGroup(object):
    """ group of multiple functions """
    def __init__(self, core_function_list=None, sub_function_list=None):
        self.core_function_list = [] if not(core_function_list) else core_function_list
        self.sub_function_list = [] if not(sub_function_list) else sub_function_list

    def add_sub_function(self, sub_function):
        self.sub_function_list.append(sub_function)

    def add_core_function(self, sub_function):
        self.core_function_list.append(sub_function)

    def apply_to_core_functions(self, routine, inc_external=False):
        for fct in self.core_function_list:
            if inc_external or not fct.external:
                routine(self, fct)

    def apply_to_sub_functions(self, routine, inc_external=False):
        for fct in self.sub_function_list:
            if inc_external or not fct.external:
                routine(self, fct)

    def apply_to_all_functions(self, routine, inc_external=False):
        self.apply_to_sub_functions(routine, inc_external=inc_external)
        self.apply_to_core_functions(routine, inc_external=inc_external)
        return self


    def merge_with_group(self, subgroup, demote_sub_core=True):
        """ Merge two FunctionGroup-s together (if demote_sub_core
            is set, the argument core and sub function list are merged
            into self sub function list, it unset core list are merged
            together and sub list are merged together """
        for sub_fct in subgroup.sub_function_list:
            self.add_sub_function(sub_fct)
        for sub_fct in subgroup.core_function_list:
            if demote_sub_core:
                self.add_sub_function(sub_fct)
            else:
                self.add_core_function(sub_fct)
        return self


    def get_code_function_by_name(self, function_name):
        for fct in self.core_function_list + self.sub_function_list:
            if fct.name == function_name:
                return fct
        return None


