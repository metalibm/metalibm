# -*- coding: utf-8 -*-

## @package ml_hdl_operations
#  Metalibm Description Language HDL Operations 

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
# created:          Nov 17th, 2016
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
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
  """ Instance of a sub-component """
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
  """ Type object for a sub-component """
  ##
  # name (str) name of the component
  # @param io_map is a dict <Signal, Signal.Specifier>
  # @param generator_object (CodeEntity): generator for the component
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

  def get_code_entity(self):
    return self.generator_object

  def get_declaration(self):
    return self.generator_object.get_component_declaration()


class PlaceHolder(AbstractOperationConstructor("PlaceHolder")):
    """ This operation has an arbitrary arity.
        For all purpose it is equal to its first input (main_input)
        but carries on several inputs """
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

## Message reporting operation
class Report(AbstractOperationConstructor("Report")):
    """ Message reporting operation """
    def __init__(self, *ops, **kw):
        Report.__base__.__init__(self, *ops, **kw)
        self.arity = len(ops)
        self.set_precision(ML_Void)

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

def cst_promotion(value, precision = ML_Integer):
    """ promote argument to Constant node with given precision if
        it is not already a ML_Operation node """
    if isinstance(value, ML_Operation):
        return value
    else:
        return Constant(value, precision = precision)

class StaticDelay(PlaceHolder):
    """ extract a signal and add a static delay on it """
    arity = 2
    name = "StaticDelay"

    def __init__(self, op, delay, relative=True, **kw):
        PlaceHolder.__init__(self, op, **kw)
        self.delay = delay
        # if set indicates that delay should be added to op's init stage
        # to determine  ouput stage
        # if unset indicates that delay is the absolute stage index for the output
        self.relative = True

    def get_name(self):
        """ return operation name for display """
        relative_label = "Rel" if self.relative else "Abs"
        return "StaticDelay[{}={}]".format(relative_label, self.delay)

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.relative = self.relative
        new_copy.delay = delay

## extract a sub-signal from inputs
#  arguments are:
#  @param arg input signal
#  @param inf_index least significant index to start from in @p arg
#  @param sup_index most significant index to stop at in @p arg
#  @return sub-signal arg(inf_index to sup_index)
class SubSignalSelection(AbstractOperationConstructor("SubSignalSelection", arity = 3)):
  def __init__(self, arg, inf_index, sup_index, **kw):
    if not "precision" in kw:
      kw["precision"] = ML_StdLogicVectorFormat(sup_index - inf_index + 1)
    inf_index = cst_promotion(inf_index, precision = ML_Integer)
    sup_index = cst_promotion(sup_index, precision = ML_Integer)
    SubSignalSelection.__base__.__init__(self, arg, inf_index, sup_index, **kw)
    #self.inf_index = inf_index
    #self.sup_index = sup_index

  def get_inf_index(self):
    return self.get_input(1) #self.inf_index
  def get_sup_index(self):
    return self.get_input(2) #self.sup_index

  #def finish_copy(self, new_copy, copy_map = {}):
  #  new_copy.inf_index = self.inf_index
  #  new_copy.sup_index = self.sup_index

## Wrapper for the generation of a bit selection operation
#  from a multi-bit signal
def BitSelection(optree, index, **kw):
  return VectorElementSelection(
    optree, 
    cst_promotion(index, precision = ML_Integer),
    precision = ML_StdLogic,
    **kw
  )

## Wrapper for single bit node equality
def equal_to(optree, cst_value):
    if not(isinstance(cst_value, int) or isinstance(cst_value, sollya.SollyaObject)):
        Log.report(Log.Error, "cst_value {} in equal_to MUST be a numeric constant".format(cst_value))

    return Equal(
        optree,
        Constant(cst_value, precision=ML_StdLogic),
        precision = ML_Bool
    )

def logical_reduce(op_list, op_ctor=LogicalOr, precision=ML_Bool, **kw):
    """ Logical/Boolean operand list reduction """
    local_list = [op for op in op_list]
    while len(local_list) > 1:
        op0 = local_list.pop(0)
        op1 = local_list.pop(0)
        local_list.append(
            op_ctor(op0, op1, precision=precision)
        )
    # assigning attributes to the resulting node
    result = local_list[0]
    result.set_attributes(**kw)
    return result

## Specialization of logical reduce to OR operation
logical_or_reduce  = lambda op_list, **kw: logical_reduce(op_list, LogicalOr, ML_Bool, **kw)
## Specialization of logical reduce to AND operation
logical_and_reduce = lambda op_list, **kw: logical_reduce(op_list, LogicalAnd, ML_Bool, **kw)


def UnsignedOperation(lhs, rhs, op_ctor, **kw):
  """ sign cast  @p lhs and @p rhs to implement a proper
      binary unsigned addition """
  return op_ctor(
    SignCast(lhs, precision=get_unsigned_precision(lhs.get_precision()), specifier = SignCast.Unsigned) if not lhs.get_precision() is ML_StdLogic else lhs,
    SignCast(rhs, precision=get_unsigned_precision(rhs.get_precision()), specifier = SignCast.Unsigned) if not rhs.get_precision() is ML_StdLogic else rhs,
    **kw
  )
def UnsignedAddition(lhs, rhs, **kw):
  return UnsignedOperation(lhs, rhs, Addition, **kw)
def UnsignedSubtraction(lhs, rhs, **kw):
  return UnsignedOperation(lhs, rhs, Subtraction, **kw)
def UnsignedMultiplication(lhs, rhs, **kw):
  return UnsignedOperation(lhs, rhs, Multiplication, **kw)

def multi_Concatenation(*args, **kw):
    """ multiple input concatenation """
    num_args = len(args)
    if num_args == 1:
        return args[0]
    else:
        half_num = int(num_args / 2)
        if not "precision" in kw:
            kw.update(precision=ML_String)
        return Concatenation(
            multi_Concatenation(*args[:half_num], **kw),
            multi_Concatenation(*args[half_num:], **kw),
            **kw
        )

## @} 
# end of metalibm's Doxygen ml_hdl_operations group
