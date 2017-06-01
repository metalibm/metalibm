# -*- coding: utf-8 -*-

## @package ml_hdl_operations
#  Metalibm Description Language HDL Operations 

###############################################################################
# This file is part of New Metalibm tool
# Copyrights Nicolas Brunie (2016)
# All rights reserved
# created:          Nov 17th, 2016
# last-modified:    May  9th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################


import sys, inspect

from sollya import Interval, SollyaObject, nearestint, floor, ceil

from ..utility.log_report import Log
from .attributes import Attributes, attr_init
from .ml_formats import *
from .ml_hdl_format import *
from .ml_operations import *

## \defgroup ml_hdl_operations ml_hdl_operations
#  @{

## Process object (sequential imperative module)
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

class SignExt(AbstractOperationConstructor("SignExt", arity = 1)):
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
def FloatBuild(sign_bit, exp_field, mantissa_field, precision = ML_Binary32, **kw):
  float_precision = precision.get_base_format()
  # assert exp_field has the right output format
  exp_field = force_size(exp_field, float_precision.get_exponent_size())
  # assert mantissa_field has the right output format
  mantissa_field = force_size(mantissa_field, float_precision.get_field_size())
  # build cast concatenation of operands
  result = TypeCast(
    Concatenation(
      Concatenation(
        sign_bit, 
        exp_field, 
        precision = ML_StdLogicVectorFormat(1 + float_precision.get_exponent_size())
      ),
      mantissa_field,
      precision = ML_StdLogicVectorFormat(float_precision.get_bit_size())
    ),
    precision = precision,
    **kw
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

class ComponentInstance(AbstractOperationConstructor("ComponentInstance")):
  def __init__(self, component_object, *args, **kwords):
    ComponentInstance.__base__.__init__(self, *args, **kwords)
    self.component_object = component_object
    self.instance_id = self.component_object.get_new_instance_id()
    self.io_map = kwords["io_map"]
    io_inputs = []
    # mapping of input integer index to component input tag
    self.comp_map_id2tag = {}
    # populating input list
    for io_tag in self.io_map:
      if self.component_object.get_io_tag_specifier(io_tag) == AbstractVariable.Input:
        io_index = len(self.inputs) + len(io_inputs)
        self.comp_map_id2tag[io_index] = io_tag
        io_inputs.append(self.io_map[io_tag])
    self.inputs = self.inputs + tuple(io_inputs)

  def set_input(self, input_id, new_input):
    ComponentInstance.__base__.set_input(self, input_id, new_input)
    if input_id in self.comp_map_id2tag:
      self.io_map[self.comp_map_id2tag[input_id]] = new_input
      

  def get_instance_id(self):
    return self.instance_id

  def get_component_object(self):
    return self.component_object

  def get_io_map(self):
    return self.io_map

class ComponentObject(object):
  ##
  # @param io_map is a dict <Signal, Signal.Specifier>
  def __init__(self, name, io_map, generator_object):
    self.name = name
    self.io_map = io_map
    self.generator_object = generator_object
    self.instance_id = -1
    self.io_map_tag2specifier = {}
    for sig in self.io_map:
      self.io_map_tag2specifier[sig.get_tag()] = self.io_map[sig]

  def get_io_tag_specifier(self, io_tag):
    return self.io_map_tag2specifier[io_tag]

  def get_new_instance_id(self):
    self.instance_id += 1
    return self.instance_id

  def get_name(self):
    return self.name

  def __call__(self, *args, **kw):
    return ComponentInstance(self, *args, **kw)

  def get_declaration(self):
    return self.generator_object.get_component_declaration()

## This operation gets several inputs while only one is used
#  as effective output
class PlaceHolder(AbstractOperationConstructor("AbstractOperation")):
  def __init__(self, *args, **kw):
    PlaceHolder.__base__.__init__(self, *args, **kw)

  def get_main_input(self):
    return self.get_input(0)

  def get_precision(self):
    return self.get_main_input().get_precision()


## Boolean assertion
class Assert(AbstractOperationConstructor("Assert")):
  class Failure: 
    descriptor = "failure"
  class Warning: 
    descriptor = "warning"
  def __init__(self, cond, error_msg, severity, **kw):
    Assert.__base__.__init__(self, cond, **kw)
    self.severity = severity
    self.error_msg = error_msg

  def get_severity(self):
    return self.severity
  def get_error_msg(self):
    return self.error_msg

  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.error_msg = self.error_msg
    new_copy.severity = severity

## Timed wait routine
class Wait(AbstractOperationConstructor("Wait")):
  def __init__(self, time_ns, **kw):
    Wait.__base__.__init__(self, **kw)
    self.time_ns = time_ns

  def get_time_ns(self):
    return self.time_ns

  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.time_ns = self.time_ns

## TypeCast for signal values
class SignCast(TypeCast):
  name = "SignCast"
  class Signed: pass # 
  class Unsigned: pass
  def __init__(self, arg, specifier = None, **kw):
    TypeCast.__init__(self, arg, **kw)
    self.specifier = specifier
  def get_codegen_key(self):
    return self.specifier

  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.specifier = self.specifier

## extract a sub-signal from inputs
#  arguments are:
#  @param arg input signal
#  @param inf_index least significant index to start from in @p arg
#  @param sup_index most significant index to stop at in @p arg
#  @return sub-signal arg(inf_index to sup_index)
class SubSignalSelection(AbstractOperationConstructor("SubSignalSelection")):
  def __init__(self, arg, inf_index, sup_index, **kw):
    if not "precision" in kw:
      kw["precision"] = ML_StdLogicVectorFormat(sup_index - inf_index + 1)
    SubSignalSelection.__base__.__init__(self, arg, **kw)
    self.inf_index = inf_index
    self.sup_index = sup_index

  def get_inf_index(self):
    return self.inf_index
  def get_sup_index(self):
    return self.sup_index

  def finish_copy(self, new_copy, copy_map = {}):
    new_copy.inf_index = self.inf_index
    new_copy.sup_index = self.sup_index

## Wrapper for the generation of a bit selection operation
#  from a multi-bit signal
def BitSelection(optree, index, **kw):
  return VectorElementSelection(
    optree, 
    Constant(index, precision = ML_Integer),
    precision = ML_StdLogic,
    **kw
  )

## @} 
# end of metalibm's Doxygen ml_hdl_operations group
