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
# created:          Oct  6th, 2015
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from ..utility.source_info import SourceInfo

from .generator_utility import ML_CG_Operator

LOG_LEVEL_COMPLEX_OPERATOR = Log.LogLevel("ComplexOperatorVerbose")

## complex generator for symbol-like operators, 
#  the generate expression modifies the optree before calling the 
#  code_generator.generate_expr
#
class ComplexOperator(ML_CG_Operator):

  ## constructor
  def __init__(self, optree_modifier, backup_operator = None, **kwords):
    ML_CG_Operator.__init__(self, **kwords)
    ## function used to modify optree before code generation
    self.optree_modifier = optree_modifier
    ## generator used when optree returned by modifier is None
    self.backup_operator = backup_operator

    self.sourceinfo = SourceInfo.retrieve_source_info(0)


  ## generate expression for operator
  # @param self current operator
  # @param code_generator CodeGenerator object used has helper for code generation services
  # @param code_object    CobeObject receiving generated code
  # @param optree         Operation object being generated
  # @param arg_tuple      tuple of optree's arguments
  # @param generate_pre_process  lambda function (None by default) used in preprocessing
  # @param kwords         generic keywords dictionnary (see ML_CG_Operator() class for list of supported arguments)
  def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
    new_optree = self.optree_modifier(optree)
    Log.report(LOG_LEVEL_COMPLEX_OPERATOR, "modified {} to {} ", optree, new_optree)
    if new_optree is None and self.backup_operator != None:
      return self.backup_operator.generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
    else:
      return code_generator.generate_expr(code_object, new_optree, **kwords)



## Dynamic operator which adapts some of its parameters according
#  to the optree being generated
class DynamicOperator(ML_CG_Operator):
  def __init__(self, dynamic_function, **kwords):
    ML_CG_Operator.__init__(self, **kwords)
    self.dynamic_function = dynamic_function

  ## generate expression for operator
  # @param self current operator
  # @param code_generator CodeGenerator object used has helper for code generation services
  # @param code_object    CobeObject receiving generated code
  # @param optree         Operation object being generated
  # @param arg_tuple      tuple of optree's arguments
  # @param generate_pre_process  lambda function (None by default) used in preprocessing
  # @param kwords         generic keywords dictionnary (see ML_CG_Operator() class for list of supported arguments)
  def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
    generated_operator = self.dynamic_function(optree)
    return generated_operator.generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)



