# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Oct 6th, 2015
# last-modified:    
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from ..utility.common import ML_NotImplemented, zip_index
from ..core.ml_formats import *
from .code_element import CodeVariable, CodeExpression, CodeFunction
from .code_constant import C_Code, Gappa_Code
from ..core.ml_operations import *
from .generator_utility import *



class ComplexOperator(ML_CG_Operator):
  """ complex generator for symbol-like operators, 
      the generate expression modifies the optree before calling the 
      code_generator.generate_expr
      """

  def __init__(self, optree_modifier, backup_operator = None, **kwords):
    ML_CG_Operator.__init__(self, **kwords)
    self.optree_modifier = optree_modifier
    self.backup_operator = backup_operator

  def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
    new_optree = self.optree_modifier(optree)
    if new_optree is None and self.backup_operator != None:
      return self.backup_operator.generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
    else:
      return code_generator.generate_expr(code_object, new_optree, **kwords)

