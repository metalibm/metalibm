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
# created:          Feb  5th, 2016
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import Variable, ReferenceAssign, Statement, Return, ML_ArithmeticOperation, ConditionBlock, LogicalAnd 

from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import FunctionOperator


from metalibm_core.utility.log_report import Log

class CallExternalizer(object):
  def __init__(self, name_factory):
    self.name_factory = name_factory


  ## 
  # @param optree ML_Operation object to be externalized
  # @param arg_list list of ML_Operation objects to be used as arguments
  # @return pair ML_Operation, ML_Funct
  def externalize_call(self, optree, arg_list, tag = "foo", result_format = None):
    # determining return format
    return_format = optree.get_precision() if result_format is None else result_format
    assert(not return_format is None and "external call result format must be defined")
    # function_name = self.main_code_object.declare_free_function_name(tag)
    function_name = self.name_factory.declare_free_function_name(tag)

    ext_function = CodeFunction(function_name, output_format = return_format)

    # creating argument copy
    arg_map = {}
    arg_index = 0
    for arg in arg_list:
      arg_tag = arg.get_tag(default = "arg_%d" % arg_index)
      arg_index += 1
      arg_map[arg] = ext_function.add_input_variable(arg_tag, arg.get_precision())

    # copying optree while swapping argument for variables
    optree_copy = optree.copy(copy_map = arg_map)
    # instanciating external function scheme
    if isinstance(optree, ML_ArithmeticOperation):
      function_optree = Statement(Return(optree_copy))
    else:
      function_optree = Statement(optree_copy)
    ext_function.set_scheme(function_optree)
    self.name_factory.declare_function(function_name, ext_function.get_function_object())

    return ext_function
