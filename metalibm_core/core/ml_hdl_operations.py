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
from .ml_hdl_format import *
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

## Build a larger value by concatenating two smaller values
# The first operand (Left hand side) is positioned as the new Value most significant bits
# and the second operand (Right hand side) is positionned
class Concatenation(AbstractOperationConstructor("Concatenation", arity = 2)): pass

## This operation replicates its operand as to completely 
#  populate its output format
class Replication(AbstractOperationConstructor("Replication", arity = 1)): pass

class Signal(AbstractVariable): pass 

## Truncate the operand to a smaller format
#  truncate parameters are derived from input and output format
class Truncate(AbstractOperationConstructor("Truncate", arity = 1)): pass

def force_size(optree, size):
  op_size = optree.get_precision().get_bit_size()
  out_precision = ML_StdLogicVectorFormat(size)
  if op_size == size:
    return optree
  elif op_size < size:
    return ZeroExt(optree, size - op_size, precision = out_precision)
  else:
    return Truncate(optree, precision = out_precision)

## Operation to assemble a float point value from
#  a sign, exponent, mantissa
#  the result = sign * 2^(exponent - bias) * (1.0 + mantissae * 2^(-precision))
def FloatBuild(sign_bit, exp_field, mantissa_field, precision = ML_Binary32):
  # assert exp_field has the right output format
  exp_field = force_size(exp_field, precision.get_exponent_size())
  # assert mantissa_field has the right output format
  mantissa_field = force_size(mantissa_field, precision.get_field_size())
  # build cast concatenation of operands
  result = TypeCast(
    Concatenation(
      Concatenation(
        sign_bit, 
        exp_field, 
        precision = ML_StdLogicVectorFormat(1 + precision.get_exponent_size())
      ),
      mantissa_field,
      precision = ML_StdLogicVectorFormat(precision.get_bit_size())
    ),
    precision = precision
  )
  return result

## Specific sub-class of loop, 
#  A single iterator is used to go through
#  the input interval 
class RangeLoop(Loop):
  ## Loop is iterating through its input interval 
  #  from lower bound to upper bound 
  class Increasing: pass
  ## Loop is iterating through its input interval 
  #  from upper bound down to lower bound 
  class Decreasing: pass

  def __init__(self, iterator, loop_range, loop_body, specifier = None, **kw):
    Loop.__init__(self, iterator, loop_body)
    self.specifier = specifier
    self.loop_range = loop_range

  def get_codegen_key(self):
    return self.specifier

  def get_specifier(self):
    return self.specifier

  def get_loop_range(self):
    return self.loop_range
