# -*- coding: utf-8 -*-

## @package ml_operations
#  Metalibm Description Language HDL Operations 

###############################################################################
# This file is part of New Metalibm tool
# Copyrights Nicolas Brunie (2016)
# All rights reserved
# created:          Nov 17th, 2016
# last-modified:    Nov 17th, 2016
#
# author(s): Nicolas Brunie (nibrunie)
###############################################################################


import sys, inspect

from sollya import Interval, SollyaObject, nearestint, floor, ceil

from ..utility.log_report import Log
from .attributes import Attributes, attr_init
from .ml_formats import *
from .ml_operations import *


class Process(AbstractOperationConstructor("Process")):
  def __init__(self, *args, **kwords):
    self.__class__.__base__.__init__(self, *args, **kwords)
    self.arity = len(args)
    # list of variables which trigger the process
    self.sensibility_list = attr_init(kwords, "sensibility_list", [])
    self.pre_statement = Statement()
  
  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.arity = self.arity

  def get_sensibility_list(self):
    return self.sensibility_list

  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.pre_statement = self.pre_statement.copy(copy_map)
    new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]

  def get_pre_statement(self):
    return self.pre_statement

  def add_to_pre_statement(self, optree):
    self.pre_statement.add(optree)

  def push_to_pre_statement(self, optree):
    self.pre_statement.push(optree)

class Event(AbstractOperationConstructor("Event", arity = 1)):
  def __init__(self, *args, **kwords):
    self.__class__.__base__.__init__(self, *args, **kwords)
    arg_precision = None if not "precision" in kwords else kwords["precision"]
    self.precision = ML_Bool if arg_precision is None else arg_precision
  def get_likely(self):
    return False

class ZeroExt(AbstractOperationConstructor("ZeroExt", arity = 1)):
  def __init__(self, op, ext_size, **kwords):
    self.__class__.__base__.__init__(self, op, **kwords)
    self.ext_size = ext_size

class Concatenation(AbstractOperationConstructor("Concatenation", arity = 2)): pass

class Replication(AbstractOperationConstructor("Replication", arity = 2)): pass

class Signal(AbstractVariable): pass 
