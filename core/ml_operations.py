# -*- coding: utf-8 -*-

## @package ml_operations
#  Metalibm Description Language basic Operation 

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 23rd, 2013
# last-modified:    Mar 20th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


import sys, inspect

from pythonsollya import Interval, SollyaObject, PSI_is_range, nearestint

from ..utility.log_report import Log
from ..utility.common import Callable, ML_NotImplemented
from .attributes import Attributes, attr_init
from .ml_formats import * # FP_SpecialValue, ML_FloatingPointException, ML_FloatingPoint_RoundingMode, ML_FPRM_Type, ML_FPE_Type

## \defgroup ml_operations ml_operations
#  @{


## merge abstract format function
#  @return most generic abstract format to unify args formats 
def std_merge_abstract_format(*args):
    has_float = False
    has_integer = False
    has_bool = False
    for arg_type in args:
        if isinstance(arg_type, ML_FP_Format): has_float = True
        if isinstance(arg_type, ML_Fixed_Format): has_integer = True
        if isinstance(arg_type, ML_Bool_Format): has_bool = True

    if has_float: return ML_Float
    if has_integer: return ML_Integer
    if has_bool: return ML_Bool
    else:
        Log.report(Log.Error, "unknown formats while merging abstract format tuple")


## parent to Metalibm's operation 
#  @brief Every operation class must inherit from this class
class ML_Operation(object):
    pass


## implicit operation conversion (from number to Constant when required) 
#  @brief This function is called on every operations arguments
#         to legalize them
def implicit_op(op):

    if isinstance(op, ML_Operation):
        return op
    elif isinstance(op, SollyaObject) or isinstance(op, int) or isinstance(op, float) or isinstance(op, FP_SpecialValue):
        return Constant(op)
    elif isinstance(op, ML_FloatingPointException):
        return Constant(op, precision = ML_FPE_Type)
    elif isinstance(op, ML_FloatingPoint_RoundingMode):
        return Constant(op, precision = ML_FPRM_Type)
    else:
        print "ERROR: unsupported operand in implicit_op conversion ", op, op.__class__
        raise Exception()


## Parent for abstract operations 
#  @brief parent to Metalibm's abstrat operation 
class AbstractOperation(ML_Operation):
    name = "AbstractOperation"
    extra_inputs = []
    global_index = 0
    str_del = "| "

    ## init operation handle
    def __init__(self, **init_map):
        self.attributes = Attributes(**init_map)
        self.index = AbstractOperation.global_index; AbstractOperation.global_index += 1
        self.get_handle().set_node(self)

    ## extract the High part of the Node
    @property
    def hi(self):
        return ComponentSelection(self, specifier = ComponentSelection.Hi)

    ## extract the Low part of the Node
    @property
    def lo(self):
        return ComponentSelection(self, specifier = ComponentSelection.Lo)

    def __getitem__(self, index):
        return VectorElementSelection(self, index)
    def __setitem__(self, index, value):
        return ReferenceAssign(VectorElementSelection(self, index), value, precision = value.get_precision())


    ## Operator for boolean negation of operands 
    def __not__(self):
        return LogicalNot(self)

    ## Operator for boolean AND of operands 
    def __and__(self, op):
        return LogicalAnd(self, op)

    ## Operator for boolean OR of operands 
    def __or__(self, op):
        return LogicalOr(self, op)

    ## Unary operator for arithmetic negation 
    def __neg__(self):
        return Negation(self)

    ## implicit add operation between AbstractOperation 
    def __add__(self, op):
        return Addition(self, implicit_op(op))

    ## 2-Operand operator for arithmetic power 
    def __pow__(self, n):
        if n == 0: return Constant(1, precision = self.get_precision())
        elif n == 1: return self
        else:
          tmp = self**(n/2)
          if n % 2 == 1:
            return tmp * tmp * self
          else:
            return tmp * tmp

    ## 2-Operand implicit subtraction operator
    def __sub__(self, op):
        """ implicit add operation between AbstractOperation """
        return Subtraction(self, implicit_op(op))

    ## 2-Operand implicit commuted addition operator
    def __radd__(self, op):
        """ implicit reflexive add operation between AbstractOperation """
        return Addition(implicit_op(op), self)

    ## 2-Operand implicit multiplication operator
    def __mul__(self, op):
        """ implicit multiply operation between AbstractOperation """
        return Multiplication(self, implicit_op(op))

    ## 2-Operand implicit commuted subtraction operator
    def __rsub__(self, op):
        """ Commutation operator for 2-operand subtraction """
        return Subtraction(implicit_op(op), self)

    ## 2-Operand implicit commuted multiplication operator
    def __rmul__(self, op):
        """ implicit reflexive multiply operation between AbstractOperation """
        return Multiplication(implicit_op(op), self)

    ## 2-Operand implicit division operator
    def __div__(self, op):
        """ implicit division operation between AbstractOperation """
        return Division(self, implicit_op(op))
        
    ## 2-Operand implicit commuted division operator
    def __rdiv__(self, op):
        """ implicit reflexive division operation between AbstractOperation """
        return Division(implicit_op(op), self)
        
    ## 2-Operand implicit modulo operator
    def __mod__(self, op):
        """ implicit modulo operation between AbstractOperation """
        return Modulo(self, implicit_op(op))
        
    ## 2-Operand implicit commuted modulo operator
    def __rmod__(self, op):
        """ implicit reflexive modulo operation between AbstractOperation """
        return Modulo(self, implicit_op(op))

    ## implicit less than operation "
    def __lt__(self, op):
        return Comparison(self, implicit_op(op), specifier = Comparison.Less)

    ## implicit less or equal operation
    def __le__(self, op):
        return Comparison(self, implicit_op(op), specifier = Comparison.LessOrEqual)

    ## implicit greater or equal operation 
    def __ge__(self, op):
        return Comparison(self, implicit_op(op), specifier = Comparison.GreaterOrEqual)

    ## implicit greater than operation
    def __gt__(self, op):
        return Comparison(self, implicit_op(op), specifier = Comparison.Greater)

    ## precision getter
    #  @return the node output precision
    def get_precision(self):
        return self.attributes.get_precision()

    ## set the node output precision
    def set_precision(self, new_precision):
        self.attributes.set_precision(new_precision)


    def get_inputs(self):
        return self.inputs

    def get_input(self, index):
        return self.inputs[index]

    ##
    #  @return the node evaluated live-range (when available) 
    def get_interval(self):
        return self.attributes.get_interval()
    ## set the node live-range interval
    def set_interval(self, new_interval):
        return self.attributes.set_interval(new_interval)

    ## wrapper for getting the exact field of node's attributes
    #  @return the node exact flag value
    def get_exact(self):
        return self.attributes.get_exact()

    ## wrapper for setting the exact field of node's attributes
    def set_exact(self, new_exact_value):
        self.attributes.set_exact(new_exact_value)

    ## wrapper for getting the tag value within node's attributes
    #  @return the node's tag 
    def get_tag(self, default = None):
        """ tag getter (transmit to self.attributes field) """
        op_tag = self.attributes.get_tag()
        return default if op_tag == None else op_tag

    ## wrapper for setting the tag value within node's attributes
    def set_tag(self, new_tag):
        """ tag setter (transmit to self.attributes field) """
        return self.attributes.set_tag(new_tag)

    ## wrapper for getting the value of debug field from node's attributes
    def get_debug(self):
        return self.attributes.get_debug()
    ## wrapper for setting the debug field value within node's attributes
    def set_debug(self, new_debug):
        return self.attributes.set_debug(new_debug)

    ## wrapper for getting the value of silent field from node's attributes
    def get_silent(self):
    ## wrapper for setting the value of silent field within node's attributes
        return self.attributes.get_silent()
    def set_silent(self, silent_value):
        return self.attributes.set_silent(silent_value)

    ## wrapper for retrieving the handle field from node's attributes
    def get_handle(self):
        return self.attributes.get_handle()
    ## wrapper for changing the handle within node's attributes
    def set_handle(self, new_handle):
        self.attributes.set_handle(new_handle)

    ##  wrapper for getting the value of clearprevious field from node's attributes
    def get_clearprevious(self):
        return self.attributes.get_clearprevious()
    ## wrapper for setting the value of clearprevious field within node's attributes
    def set_clearprevious(self, new_clearprevious):
        return self.attributes.set_clearprevious(new_clearprevious)

    ##  wrapper for getting the value of unbreakable field from node's attributes
    def get_unbreakable(self):
        return self.attributes.get_unbreakable()
    ## wrapper for setting the value of unbreakable field within node's attributes
    def set_unbreakable(self, new_unbreakable):
        return self.attributes.set_unbreakable(new_unbreakable)

    ## wrapper to change some attributes values using dictionnary arguments
    def set_attributes(self, **kwords):
        if "likely" in kwords:
            self.set_likely(kwords["likely"])
            kwords.pop("likely")
        self.attributes.set_attr(**kwords)

    ## modify the values of some node's attributes and return the current node
    #  @return current node
    def modify_attributes(self, **kwords):
        self.attributes.set_attr(**kwords)
        return self

    ## 
    #  @return the node index
    def get_index(self):
        """ index getter function """
        return self.index
    ## set the node's index value
    #  
    def set_index(self, new_index):
        """ index setter function """
        self.index = new_index 

    ## wrapper for getting the rounding_mode field from node's attributes
    #  @return the node rounding mode field
    def get_rounding_mode(self):
        """ rounding mode getter function (attributes)"""
        return self.attributes.get_rounding_mode()
    ## wrapper for setting the rounding field within node's attributes
    def set_rounding_mode(self, new_rounding_mode):
        """ rounding mode setter function (attributes) """
        self.attributes.set_rounding_mode(new_rounding_mode)

    ## wrapper for getting the max_abs_error field of node's attributes
    #  @return the node's max_abs_error attribute value
    def get_max_abs_error(self):
        return self.attributes.get_max_abs_error()
    ## wrapper for setting the max_abs_error field value within node's attributes
    def set_max_abs_error(self, new_max_abs_error):
        self.attributes.set_max_abs_error(new_max_abs_error)

    def get_prevent_optimization(self):
        return self.attributes.get_prevent_optimization()
    def set_prevent_optimization(self, prevent_optimization):
        self.attributes.set_prevent_optimization(prevent_optimization)

    ## wrapper to access the class name field
    #  @return the node's name (generally node's class name)
    def get_name(self):
        """ return operation name (by default class name) """
        return self.name

    ##
    #  @return the list of node's extra inputs
    def get_extra_inputs(self):
        """ return list of non-standard inputs """
        return self.extra_inputs

    ## change the node to mirror optree
    # by copying class, attributes, arity and inputs from optree to self
    def change_to(self, optree):
        """ change <self> operation to match optree """
        self.__class__ = optree.__class__
        self.inputs = optree.inputs
        self.arity = optree.arity
        self.attributes = optree.attributes
        if isinstance(optree, SpecifierOperation):
            self.specifier = optree.specifier


    ## string conversion 
    #  @param  depth [integer/None] node depth where the display recursion stops
    #  @param  display_precision [boolean] enable/display node's precision display
    #  @param  tab_level number of tab to be inserted left to node's description
    #  @param  memoization_map [dict] hastable to store previously described node (already generated node tag will be use rather than copying the full description)
    #  @param  display_attribute [boolean] enable/disable display of node's attributes
    #  @param  display_id [boolean]  enable/disbale display of unique node identified
    #  @return a string describing the node
    def get_str(self, depth = 2, display_precision = False, tab_level = 0, memoization_map = {}, display_attribute = False, display_id = False):
        new_depth = None 
        if depth != None:
            if  depth < 0: 
                return "" 
        new_depth = (depth - 1) if depth != None else None
            
        tab_str = AbstractOperation.str_del * tab_level
        silent_str = "[S]" if self.get_silent() else ""
        dbg_str = "[DBG]" if self.get_debug() else ""
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        if self.arity == 1:
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            memoization_map[self] = str_tag
            return tab_str + "%s%s%s%s%s%s -------> %s\n%s" % (self.get_name(), precision_str, dbg_str, silent_str, id_str, attribute_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id) for inp in self.inputs))
        else:
            memoization_map[self] = str_tag
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            return tab_str + "%s%s%s%s%s%s ------> %s\n%s" % (self.get_name(), precision_str, dbg_str, silent_str, id_str, attribute_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id) for inp in self.inputs))


    ## virtual function, called after a node's copy
    #  overleaded by inheriter of AbstractOperation
    def finish_copy(self, new_copy, copy_map = {}):
        pass


    ## pure virtual copy  node function
    #  @param copy_map dictionnary of previously built copy, if a node is found within this table, table's value is returned as copy result
    #  @return node's copy (newly generated or memoized)
    def copy(self, copy_map = {}):
        print "Error: copy not implemented"
        print self, self.__class__
        raise ML_NotImplemented()

    ## propagate given precision
    #  @param precision
    #  @return None
    def propagate_precision(self, precision, boundary_list = []):
      self.set_precision(precision)
      if not isinstance(self, ML_LeafNode):
        for op in self.inputs:
          if op.get_precision() is None and not op in boundary_list: 
            op.propagate_precision(precision, boundary_list)


## base class for all arithmetic operation that may depend
#  on floating-point context (rounding mode for example) 
class ML_ArithmeticOperation(AbstractOperation):
    pass

## Parent for AbstractOperation with no expected input
class ML_LeafNode(AbstractOperation): 
    pass

## Constant node class
class Constant(ML_LeafNode):
    ## Initializer
    #  @param value numerical value of the constant
    #  @param init_map dictionnary for attributes initialization
    def __init__(self, value, **init_map):
        # value initialization
        AbstractOperation.__init__(self, **init_map)
        self.value = value
        # attribute fields initialization
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, SollyaObject):
            self.attributes.set_interval(Interval(value))


    ## accessor to the constat value
    #  @return the numerical constant value
    def get_value(self):
        return self.value
    def set_value(self, new_value):
        self.value = new_value

    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}, display_attribute = False, display_id = False):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        id_str        = ("[id=%x]" % id(self)) if display_id else ""
        return AbstractOperation.str_del * tab_level + "Cst(%s)%s%s%s\n" % (self.value, attribute_str, precision_str, id_str)


    def copy(self, copy_map = {}):
        """ return a new, free copy of <self> """
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # else define a new and free copy
        new_copy = Constant(self.value)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        return new_copy


## class for Variable node, which contains a temporary state of the operation DAG
##  which may have been defined outside the scope of the implementation (input variable)
class Variable(ML_LeafNode):
    ## Input type for Variable Node
    #  such node is not defined as an input to the function description
    class Input: pass
    class Local: pass

    ## Intermediary type for Variable Node
    #  such node is defined within the function description.
    #  It holds an intermediary state
    class Intermediary: pass

    ## constructor
    #  @param tag string name of the Variable object
    #  @param init_map standard ML_Operation attribute dictionnary initialization 
    def __init__(self, tag, **init_map):
        AbstractOperation.__init__(self, **init_map)
        self.attributes.set_tag(tag)
        # used to distinguish between input variables (without self.inputs) 
        # and intermediary variables 
        self.var_type = attr_init(init_map, "var_type", default_value = Variable.Input)  

    ## @return the type (Input or Intermediary0 of the Variable node
    def get_var_type(self):
        return self.var_type

    ## generate string description of the Variable node
    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}, display_attribute = False, display_id = False):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        id_str        = ("[id=%x]" % id(self)) if display_id else ""
        return AbstractOperation.str_del * tab_level + "Var(%s)%s%s%s\n" % (self.get_tag(), precision_str, attribute_str, id_str)


    def copy(self, copy_map = {}):
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # by default input variable are not copied
        # this behavior can be bypassed by manually 
        # defining a copy into copy_map
        if self.get_var_type() == Variable.Input: 
            copy_map[self] = self
            return self
        # else define a new and free copy
        new_copy = Variable(tag = self.get_tag(), var_type = self.var_type)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        return new_copy


class InstanciatedOperation(ML_Operation):
  """ parent to Metalibm's type-instanciated operation """
  pass


def AbstractOperation_init(self, *ops, **init_map):
  """ init function for abstract operation """
  AbstractOperation.__init__(self, **init_map)
  self.inputs = tuple(implicit_op(op) for op in ops)
  if self.get_interval() == None:
      self.set_interval(self.range_function(self.inputs))

def AbstractOperation_copy(self, copy_map = {}):
  """ base function to copy an abstract operation object,
      copy_map is a memoization hashtable which can be use to factorize
      copies """
  # test for previous definition in memoization map
  if self in copy_map: return copy_map[self]
  # else define a new and free copy
  new_copy = self.__class__(*tuple(op.copy(copy_map) for op in self.inputs), __copy = True)
  new_copy.attributes = self.attributes.get_copy()
  self.finish_copy(new_copy, copy_map)
  copy_map[self] = new_copy
  return new_copy


class InvalidInterval(Exception): 
  """ Invalid interval exception """
  pass

def interval_check(lrange):
    """ check if the argument <lrange> is a valid interval,
        if it is, returns it, else raises an InvalidInterval 
        exception """
    if isinstance(lrange, SollyaObject) and PSI_is_range(lrange):
        return lrange
    else:
        raise InvalidInterval()

def interval_wrapper(self, interval_op, ops):
    try:
        return interval_op(self, tuple(interval_check(op.get_interval()) for op in ops))
    except InvalidInterval:
        return None


def interval_func(interval_op):
    """ interval function builder for multi-arity interval operation """
    if interval_op == None:
        return lambda self, ops: None
    else:
        return lambda self, ops: interval_wrapper(self, interval_op, ops)

def AbstractOperation_get_codegen_key(self):
    return None



def GeneralOperationConstructor(name, arity = 2, range_function = None, error_function = None, inheritance = [], base_class = AbstractOperation):
    """ meta-class constructor for abstract operation """
    field_map = {
        # operation initialization function assignation
        "__init__": AbstractOperation_init,
        # operation copy
        "copy": AbstractOperation_copy,
        # operation name
        "name": name,
        # operation arity
        "arity": arity, 
        # interval function building
        "range_function": interval_func(range_function), 
        # error function building
        "error_function": error_function,
        # generation key building
        "get_codegen_key": AbstractOperation_get_codegen_key,
    }
    return type(name, (base_class,) + tuple(inheritance), field_map)


def AbstractOperationConstructor(name, arity = 2, range_function = None, error_function = None, inheritance = []):
    return GeneralOperationConstructor(name, arity = arity, range_function = range_function, error_function = error_function, inheritance = inheritance, base_class = AbstractOperation)


def ArithmeticOperationConstructor(name, arity = 2, range_function = None, error_function = None, inheritance = []):
    return GeneralOperationConstructor(name, arity = arity, range_function = range_function, error_function = error_function, inheritance = inheritance, base_class = ML_ArithmeticOperation)


class BitLogicAnd(ArithmeticOperationConstructor("BitLogicAnd")):
    pass
class BitLogicOr(ArithmeticOperationConstructor("BitLogicOr")):
    pass
class BitLogicXor(ArithmeticOperationConstructor("BitLogicXor")):
    pass
class BitLogicNegate(ArithmeticOperationConstructor("BitLogicNegate", arity = 1)):
    pass
class BitLogicRightShift(ArithmeticOperationConstructor("BitLogicRightShift", arity = 2)):
    pass
class BitLogicLeftShift(ArithmeticOperationConstructor("BitLogicLeftShift", arity = 2)):
    pass


class Abs(ArithmeticOperationConstructor("Abs", range_function = lambda self, ops: abs(ops[0]))):
    """ abstract absolute value operation """
    pass

class Negation(ArithmeticOperationConstructor("Negation", range_function = lambda self, ops: - ops[0])): 
    """ abstract negation """
    pass

class Addition(ArithmeticOperationConstructor("Addition", range_function = lambda self, ops: ops[0] + ops[1])): 
    """ abstract addition """
    pass

class Negate(ArithmeticOperationConstructor("Negate", range_function = lambda self, ops: -ops[0])): 
    """ abstract negation """
    pass


class SpecifierOperation: 
    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

class ComponentSelectionSpecifier: pass

class Split(ArithmeticOperationConstructor("Split", arity = 1)):
    pass
class ComponentSelection(ArithmeticOperationConstructor("ComponentSelection", inheritance = [SpecifierOperation], arity = 1)):
    class Hi(ComponentSelectionSpecifier): pass
    class Lo(ComponentSelectionSpecifier): pass

    implicit_arg_precision = {
        ML_DoubleDouble: ML_Binary64,
        ML_TripleDouble: ML_Binary64,
    }

    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def __init__(self, *args, **kwords):
        self.specifier = attr_init(kwords, "specifier", ComponentSelection.Hi)
        self.__class__.__base__.__init__(self, *args, **kwords)

        # setting implicit precision
        if self.get_precision() == None and len(args) > 0 and args[0].get_precision() != None:
            arg_precision = args[0].get_precision()
            if arg_precision in ComponentSelection.implicit_arg_precision:
                self.set_precision(ComponentSelection.implicit_arg_precision[arg_precision])


class FMASpecifier(object): 
    """ Common parent to all Test specifiers """
    pass

def FMASpecifier_Builder(name, arity, range_function = None): 
    """ Test Specifier constructor """
    return type(name, (FMASpecifier,), {"arity": arity, "name": name, "range_function": Callable(range_function)})

class FusedMultiplyAdd(ArithmeticOperationConstructor("FusedMultiplyAdd", inheritance = [SpecifierOperation], range_function = lambda optree, ops: optree.specifier.range_function(optree, ops))):
    """ abstract fused multiply and add operation op0 * op1 + op2 """
    class Standard(FMASpecifier_Builder("Standard", 3, lambda optree, ops: ops[0] * ops[1] + ops[2])):
        """ op0 * op1 + op2 """
        pass
    class Subtract(FMASpecifier_Builder("Subtract", 3, lambda optree, ops: ops[0] * ops[1] - ops[2])): 
        """ op0 * op1 - op2 """
        pass
    class Negate(FMASpecifier_Builder("Negate", 3, lambda _self, ops: - ops[0] * ops[1] - ops[2])): 
        """ -op0 * op1 - op2 """
        pass
    class SubtractNegate(FMASpecifier_Builder("SubtractNegate", 3, lambda _self, ops: - ops[0] * ops[1] + ops[2])):
        """ -op0 * op1 + op2 """
        pass
    class DotProduct(FMASpecifier_Builder("DotProduct", 4, lambda _self, ops: ops[0] * ops[1] + ops[2] * ops[3])):
        """ op0 * op1 + op2 * op3 """
        pass
    class DotProductNegate(FMASpecifier_Builder("DotProductNegate", 4, lambda _self, ops: ops[0] * ops[1] - ops[2] * ops[3])):
        """ op0 * op1 - op2 * op3 """
        pass
        
    def __init__(self, *args, **kwords):
        self.specifier = attr_init(kwords, "specifier", FusedMultiplyAdd.Standard)
        # indicates wheter a base operation commutation has been processed
        self.commutated = attr_init(kwords, "commutated", False)
        self.__class__.__base__.__init__(self, *args, **kwords)
        self.arity = self.specifier.arity

    def set_commutated(self, new_commutated_value):
        self.commutated = new_commutated_value
    def get_commutated(self):
        return self.commutated

    def get_name(self):
        """ return operation name (class.specifier) """
        com = "[C]" if self.commutated else ""
        return  "FusedMultiplyAdd.%s%s" % (self.specifier.name, com)

    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.specifier = self.specifier
        new_copy.arity = new_copy.specifier.arity
        new_copy.commutated = self.commutated

def FMA(op0, op1, op2, **kwords):
    kwords["specifier"] = FusedMultiplyAdd.Standard
    return FusedMultiplyAdd(op0, op1, op2, **kwords)

def FMS(op0, op1, op2, **kwords):
    kwords["specifier"] = FusedMultiplyAdd.Subtract
    return FusedMultiplyAdd(op0, op1, op2, **kwords)

def FMSN(op0, op1, op2, **kwords):
    kwords["specifier"] = FusedMultiplyAdd.SubtractNegate
    return FusedMultiplyAdd(op0, op1, op2, **kwords)



class Subtraction(ArithmeticOperationConstructor("Subtraction", range_function = lambda self, ops: ops[0] - ops[1])): 
    """ abstract addition """
    pass


class Multiplication(ArithmeticOperationConstructor("Multiplication", range_function = lambda self, ops: ops[0] * ops[1])): 
    """ abstract addition """
    pass


class Division(ArithmeticOperationConstructor("Division", range_function = lambda self, ops: ops[0] / ops[1])): 
    """ abstract addition """
    pass


class Modulo(ArithmeticOperationConstructor("Modulo", range_function = lambda self, ops: ops[0] % ops[1])):
    """ abstract modulo operation """
    pass


class NearestInteger(ArithmeticOperationConstructor("NearestInteger", arity = 1, range_function = lambda self, ops: nearestint(ops[0]))): 
    """ abstract addition """
    pass


class PowerOf2(ArithmeticOperationConstructor("PowerOf2", arity = 1, range_function = lambda self, ops: S2**ops[0])):
    """ abstract power of 2 operation """
    pass

class Return(AbstractOperationConstructor("Return", arity = 1, range_function = lambda self, ops: ops[0])):
    """ abstract return value operation """
    pass

class TableLoad(ArithmeticOperationConstructor("TableLoad", arity = 2, range_function = lambda self, ops: None)):
    """ abstract load from a table operation """
    pass


def interval_union(int0, int1):
    return Interval(min(inf(int0), inf(int1)), max(sup(int0), sup(int1)))

class Select(ArithmeticOperationConstructor("Select", arity = 3, range_function = lambda self, ops: interval_union(ops[1], ops[2]))):
    pass


def Max(op0, op1, **kwords):
    return Select(Comparison(op0, op1, specifier = Comparison.Greater), op0, op1, **kwords)

def Min(op0, op1, **kwords):
    return Select(Comparison(op0, op1, specifier = Comparison.Less), op0, op1, **kwords)

class Loop(AbstractOperationConstructor("Loop", arity = 3)):
  """ abstract loop constructor 
      loop (init_statement, exit_condition, loop_body)
  """


class ConditionBlock(AbstractOperationConstructor("ConditionBlock", arity = 3)):
    """ abstract if/then(/else) block """
    
    def __init__(self, *args, **kwords):
        """ condition block initialization """
        self.__class__.__base__.__init__(self, *args, **kwords)
        self.parent_list = []
        # statement being executed before the condition or either of the branch is executed 
        self.pre_statement = Statement()
        self.extra_inputs = [self.pre_statement]

    def set_extra_inputs(self, new_extra_inputs):
        self.extra_inputs = new_extra_inputs

    def set_parent_list(self, parent_list):
        self.parent_list = parent_list

    def get_parent_list(self):
        return self.parent_list

    def get_pre_statement(self):
        return self.pre_statement

    def add_to_pre_statement(self, optree):
        self.pre_statement.add(optree)

    def push_to_pre_statement(self, optree):
        self.pre_statement.push(optree)

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.pre_statement = self.pre_statement.copy(copy_map)
        new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]
        new_copy.parent_list = [op.copy(copy_map) for op in self.parent_list] 
  

class Conversion(ArithmeticOperationConstructor("Conversion", arity = 1)):
  """ abstract conversion operation """
  pass

class TypeCast(ArithmeticOperationConstructor("TypeCast", arity = 1)):
  """ abstract conversion operation """
  pass
    
class Dereference(ArithmeticOperationConstructor("Dereference", arity = 1)):
  """ abstract pointer derefence operation """
  pass

class ReferenceAssign(AbstractOperationConstructor("ReferenceAssign", arity = 1)):
  """ abstract assignation to reference operation """
  pass

class ExponentInsertion(ArithmeticOperationConstructor("ExponentInsertion", arity = 1, inheritance = [SpecifierOperation])):
  """ insertion of a number in the exponent field of a floating-point value """ 
  class Default: pass
  class NoOffset: pass

  def __init__(self, *args, **kwords):
    self.__class__.__base__.__init__(self, *args, **kwords)
    self.specifier = attr_init(kwords, "specifier", default_value = ExponentInsertion.Default)

  def get_codegen_key(self):
    """ return code generation specific key """
    return self.specifier

class MantissaExtraction(ArithmeticOperationConstructor("MantissaExtraction", arity = 1)):
  """ extraction of the mantissa field of a floating-point value """
  pass

class ExponentExtraction(ArithmeticOperationConstructor("ExponentExtraction", arity = 1)):
  """ extraction of the exponent field of a floating-point value """
  pass

class RawSignExpExtraction(ArithmeticOperationConstructor("RawSignExpExtraction", arity = 1)):
    pass

class CountLeadingZeros(ArithmeticOperationConstructor("CountLeadingZeros", arity = 1)):
    pass

class TestSpecifier(object): 
    """ Common parent to all Test specifiers """
    pass

def TestSpecifier_Builder(name, arity): 
    """ Test Specifier constructor """
    return type(name, (TestSpecifier,), {"arity": arity, "name": name})

class LikelyPossible: 
    """ likely true or false """
    pass

class BooleanOperation:
    """ Boolean operation parent """
    def __init__(self, likely):
        """ # likely indicate if the boolean operation is likely
            #  -> to be True (likely = True)
            #  -> to be False (likely = False)
            #  -> to be either True or False (likely = LikelyPossible)
            #  -> undertermined (likely = None)"""
        self.likely = likely

    def get_likely(self):
        """ return likely value """
        return self.likely

    def set_likely(self, likely):
        """ set likely value """
        self.likely = likely

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.likely = self.likely


def logic_operation_init(self, *args, **kwords):
    self.__class__.__base__.__init__(self, *args, **kwords)
    BooleanOperation.__init__(self, attr_init(kwords, "likely"))
    # if likely has not been overloaded
    # trying to determine it with respect to input likeliness
    if self.likely == None:
        self.likely = self.likely_function(*tuple(arg.get_likely() for arg in args))

def LogicOperationBuilder(op_name, arity = 2, likely_function = lambda self, *ops: None):
    field_map = {
        "__init__": logic_operation_init,
        "likely_function": likely_function,
    }
    return type(op_name, (ArithmeticOperationConstructor(op_name, inheritance = [BooleanOperation]),), field_map)



LogicalAnd = LogicOperationBuilder("LogicalAnd", arity = 2, likely_function = lambda self, *ops: ops[0] and ops[1])
LogicalOr  = LogicOperationBuilder("LogicalOr",  arity = 2, likely_function = lambda self, *ops: ops[0] or ops[1])
LogicalNot = LogicOperationBuilder("LogicalNot", arity = 1, likely_function = lambda self, *ops: not ops[0] )

        

class Test(ArithmeticOperationConstructor("Test", inheritance = [BooleanOperation, SpecifierOperation])):
    """ Abstract Test operation class """
    class IsNaN(TestSpecifier_Builder("IsNaN", 1)): pass
    class IsQuietNaN(TestSpecifier_Builder("IsQuietNaN", 1)): pass
    class IsSignalingNaN(TestSpecifier_Builder("IsSignalingNaN", 1)): pass
    class IsInfty(TestSpecifier_Builder("IsInfty", 1)): pass
    class IsPositiveInfty(TestSpecifier_Builder("IsPositiveInfty", 1)): pass
    class IsNegativeInfty(TestSpecifier_Builder("IsNegativeInfty", 1)): pass
    class IsIEEENormalPositive(TestSpecifier_Builder("IsIEEENormalPositive", 1)): pass
    class IsInfOrNaN(TestSpecifier_Builder("IsInfOrNaN", 1)): pass
    class IsZero(TestSpecifier_Builder("IsZero", 1)): pass
    class IsPositiveZero(TestSpecifier_Builder("IsPositiveZero", 1)): pass
    class IsNegativeZero(TestSpecifier_Builder("IsNegativeZero", 1)): pass
    class IsSubnormal(TestSpecifier_Builder("IsSubnormal", 1)): pass
    class CompSign(TestSpecifier_Builder("CompSign", 2)): pass
    class SpecialCases(TestSpecifier_Builder("SpecialCases", 1)): pass
    class IsInvalidInput(TestSpecifier_Builder("IsInvalidInput", 1)): pass
    class IsMaskAllZero(TestSpecifier_Builder("IsMaskAllZero", 1)): pass
    class IsMaskNotAllZero(TestSpecifier_Builder("IsMaskNotAllZero", 1)): pass
    class IsMaskAnyZero(TestSpecifier_Builder("IsMaskAnyZero", 1)): pass
    class IsMaskNotAnyZero(TestSpecifier_Builder("IsMaskNotAnyZero", 1)): pass

    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        BooleanOperation.__init__(self, attr_init(kwords, "likely"))
        self.specifier = attr_init(kwords, "specifier", required = True)
        self.arity = self.specifier.arity if not self.specifier is None else 1


    def get_name(self):
        """ return operation name (class.specifier) """
        return  "Test.%s" % self.specifier.name


    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.specifier = self.specifier
        BooleanOperation.finish_copy(self, new_copy, copy_map)
        new_copy.arity = self.arity

class ComparisonSpecifier(object): pass
def CompSpecBuilder(name, opcode):
    field_map = {
        # operation copy
        "opcode": opcode,
        # operation name
        "get_opcode": (lambda self: self.opcode),
    }
    return type(name, (ComparisonSpecifier,), field_map)

  

class Comparison(ArithmeticOperationConstructor("Comparison", arity = 2, inheritance = [BooleanOperation, SpecifierOperation])):
    """ Abstract Comparison operation """
    Equal          = CompSpecBuilder("Equal", "eq")
    NotEqual       = CompSpecBuilder("NotEqual", "ne")
    Less           = CompSpecBuilder("Less",  "lt")
    LessOrEqual    = CompSpecBuilder("LessOrEqual", "le")
    Greater        = CompSpecBuilder("Greater", "gt")
    GreaterOrEqual = CompSpecBuilder("GreaterOrEqual", "ge")


    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        BooleanOperation.__init__(self, attr_init(kwords, "likely"))
        self.specifier = attr_init(kwords, "specifier", required = True)


    def get_name(self):
        """ return operation name (class.specifier) """
        return  "Comparison.%s" % self.specifier.__name__


    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier


    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.specifier = self.specifier
        BooleanOperation.finish_copy(self, new_copy, copy_map)

def Equal(op0, op1, **kwords):
    """ syntaxic bypass for equality comparison """
    kwords["specifier"] = Comparison.Equal
    return Comparison(op0, op1, **kwords)

def NotEqual(op0, op1, **kwords):
    """ syntaxic bypass for non-equality comparison """
    kwords["specifier"] = Comparison.NotEqual
    return Comparison(op0, op1, **kwords)

class Statement(AbstractOperationConstructor("Statement")):
    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        self.arity = len(args)


    # add a new statement at the end of the inputs list 
    # @param optree ML_Operation object added at the end of inputs list
    def add(self, optree):
        self.inputs = self.inputs + (optree,)
        self.arity += 1

    # push a new statement at the end of the inputs list 
    # @param optree ML_Operation object added at the end of inputs list
    def push(self, optree):
        """ add a new unary statement at the beginning of the input list """
        self.inputs = (optree,) + self.inputs
        self.arity += 1


    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.arity = self.arity


class FieldExtraction(ArithmeticOperationConstructor("FieldExtraction")):
    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        self.arity = len(args)

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.arity = self.arity

class SO_Specifier_Type(object): 
    """ parent to SpecificOperation Specifiers """
    pass

def SO_Specifier_Builder(name, abstract_type_rule, instantiated_type_rule, arity_func = None):
    field_map = {
        "name": name,
        "abstract_type_rule": Callable(abstract_type_rule),
        "instantiated_type_rule": Callable(instantiated_type_rule),
        "arity_func": Callable(arity_func),
    }
    return type(name, (SO_Specifier_Type,), field_map)




class SpecificOperation(AbstractOperationConstructor("SpecificOperation", inheritance = [SpecifierOperation])):
    # specifier init
    DivisionSeed   = SO_Specifier_Builder("DivisionSeed", lambda optree, *ops: ops[0].get_precision(), lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs))  
    InverseSquareRootSeed   = SO_Specifier_Builder("InverseSquareRootSeed", lambda optree, *ops: ops[0].get_precision(), lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs))  
    Subnormalize      = SO_Specifier_Builder("Subnormalize", lambda optree, *ops: optree.get_precision(), lambda backend, op, dprec: op.get_precision())
    GetRndMode        = SO_Specifier_Builder("GetRndMode", lambda optree, *ops: ML_FPRM_Type, lambda backend, op, dprec: ML_FPRM_Type)
    CopySign          = SO_Specifier_Builder("CopySign", lambda optree, *ops: std_merge_abstract_format(ops[0].get_precision(), ops[1].get_precision()), lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs))
    RoundedSignedOverflow = SO_Specifier_Builder("RoundedSignedOverflow", lambda optree, *ops: ops[0].get_precision(), lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs))


    def __init__(self, *args, **kwords): 
        SpecificOperation.__base__.__init__(self, *args, **kwords)
        self.specifier = attr_init(kwords, "specifier", required = True)
        return_value = attr_init(kwords, "return_value", None)
        self.function_name = attr_init(kwords, "function_name", None)
        arg_value = attr_init(kwords, "arg_value", None)
        self.arity = len(args)
        self.extra_inputs = [op for op in [return_value, arg_value] if op != None]
        self.return_value_index = None if return_value == None else self.extra_inputs.index(return_value)
        self.arg_value_index = None if arg_value == None else self.extra_inputs.index(arg_value)
        self.extra_inputs = [implicit_op(op) for op in self.extra_inputs]

    def get_name(self):
        """ return operation name (class.specifier) """
        return  "SpecificOperation.%s" % self.specifier.__name__

    def set_extra_inputs(self, new_extra_inputs):
        self.extra_inputs = new_extra_inputs

    def get_return_value(self):
        return self.extra_inputs[self.return_value_index]

    def get_arg_value(self):
        return self.extra_inputs[self.arg_value_index]

    def get_specifier(self):
        return self.specifier

    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier


    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.specifier = self.specifier 
        new_copy.function_name = self.function_name
        new_copy.arity = self.arity
        new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]
        new_copy.return_value_index = self.return_value_index
        new_copy.arg_value_index = self.arg_value_index

class ExceptionOperation(SpecificOperation, ML_LeafNode):
    # specifier init
    ClearException = SO_Specifier_Builder("ClearException", lambda *ops: None, lambda backend, op, dprec: None)  
    RaiseException = SO_Specifier_Builder("RaiseException", lambda *ops: None, lambda backend, op, dprec: None)  
    RaiseReturn    = SO_Specifier_Builder("RaiseReturn", lambda *ops: None, lambda backend, op, dprec: dprec)  


class NoResultOperation(SpecificOperation, ML_LeafNode):
    SaveFPContext     = SO_Specifier_Builder("SaveFPContext", lambda optree, *ops: None, lambda backend, op, dprec: None)
    RestoreFPContext  = SO_Specifier_Builder("RestoreFPContext", lambda optree, *ops: None, lambda backend, op, dprec: None)
    SetRndMode        = SO_Specifier_Builder("SetRndMode", lambda optree, *ops: None, lambda backend, op, dprec: None)

def ClearException(*args, **kwords):
    kwords["specifier"] = ExceptionOperation.ClearException
    return ExceptionOperation(*args, **kwords)

def Raise(*args, **kwords):
    kwords["specifier"] = ExceptionOperation.RaiseException
    return ExceptionOperation(*args, **kwords)

def RaiseReturn(*args, **kwords):
    kwords["specifier"] = ExceptionOperation.RaiseReturn
    return ExceptionOperation(*args, **kwords)

def DivisionSeed(*args, **kwords):
    kwords["specifier"] = SpecificOperation.DivisionSeed
    return SpecificOperation(*args, **kwords)

def InverseSquareRootSeed(*args, **kwords):
    kwords["specifier"] = SpecificOperation.InverseSquareRootSeed
    return SpecificOperation(*args, **kwords)

def SaveFPContext(**kwords):
    kwords["specifier"] = NoResultOperation.SaveFPContext
    return NoResultOperation(**kwords)

def RestoreFPContext(**kwords):
    kwords["specifier"] = NoResultOperation.RestoreFPContext
    return NoResultOperation(**kwords)

def SetRndMode(new_rnd_mode, **kwords):
    kwords["specifier"] = NoResultOperation.SetRndMode
    return NoResultOperation(new_rnd_mode, **kwords)

def GetRndMode(**kwords):
    kwords["specifier"] = SpecificOperation.GetRndMode
    return SpecificOperation(**kwords)

def CopySign(*args, **kwords):
    kwords["specifier"] = SpecificOperation.CopySign
    return SpecificOperation(*args, **kwords)

def RoundedSignedOverflow(*args, **kwords):
    kwords["specifier"] = SpecificOperation.RoundedSignedOverflow
    return SpecificOperation(*args, **kwords)


class FunctionObject:
    def __init__(self, name, arg_list_precision, output_precision, generator_object):
        self.name = name
        self.arg_list_precision = arg_list_precision
        self.output_precision = output_precision
        self.arity = len(self.arg_list_precision)
        self.generator_object = generator_object

    def __call__(self, *args, **kwords):
        return FunctionCall(self, *args, **kwords)

    def get_declaration(self):
      return "%s %s(%s);" % (self.output_precision.get_c_name(), self.get_function_name(), ", ".join(arg.get_c_name() for arg in self.arg_list_precision))

    def get_precision(self):
        return self.output_precision

    def get_arity(self):
        return self.arity

    def get_arg_precision(self, arg_index):
        return self.arg_list_precision[arg_index]

    def get_arg_precision_tuple(self):
        return tuple(self.arg_list_precision)

    def get_generator_object(self):
        return self.generator_object

    def get_function_name(self):
        return self.name


class FunctionCall(AbstractOperationConstructor("FunctionCall")):
    def __init__(self, function_object, *args, **kwords):
        FunctionCall.__base__.__init__(self, *args, **kwords)
        self.arity = len(args)
        self.function_object = function_object

    def get_precision(self):
        return self.function_object.get_precision()

    def get_function_object(self):
        return self.function_object

    def get_name(self):
      return "FunctionCall to %s" % self.get_function_object().get_function_name()

    def propagate_format_to_cst(optree):
        """ propagate new_optree_format to Constant operand of <optree> with abstract precision """
        index_list = xrange(len(optree.inputs)) 
        for index in index_list:
            inp = optree.inputs[index]
            new_optree_format = optree.get_function_object().get_arg_precision(index)
            if isinstance(inp, Constant) and isinstance(inp.get_precision(), ML_AbstractFormat):
                inp.set_precision(new_optree_format)

    def copy(self, copy_map = {}):
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # else define a new and free copy
        new_copy = self.__class__(self.function_object, *tuple(op.copy(copy_map) for op in self.inputs), __copy = True)
        new_copy.attributes = self.attributes.get_copy()
        self.finish_copy(new_copy, copy_map)
        copy_map[self] = new_copy
        return new_copy


    # static binding
    propagate_format_to_cst = Callable(propagate_format_to_cst)

class SwitchBlock(AbstractOperationConstructor("Switch", arity = 1)):
    def __init__(self, switch_value, case_map, **kwords):
        SwitchBlock.__base__.__init__(self, switch_value, **kwords)
        self.extra_inputs = [case_map[case] for case in case_map]
        self.parent_list = []
        # statement being executed before the condition or either of the branch is executed 
        self.pre_statement = Statement()
        self.case_map = case_map

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.pre_statement = self.statement.copy(copy_map)
        new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]
        new_copy.parent_list = [op.copy(copy_map) for op in self.parent_list] 

    def get_case_map(self):
        return self.case_map

    def set_extra_inputs(self, new_extra_inputs):
        self.extra_inputs = new_extra_inputs

    def set_parent_list(self, parent_list):
        self.parent_list = parent_list

    def get_parent_list(self):
        return self.parent_list

    def get_pre_statement(self):
        return self.pre_statement

    def add_to_pre_statement(self, optree):
        self.pre_statement.add(optree)

    def push_to_pre_statement(self, optree):
        self.pre_statement.push(optree)

    def get_str(self, depth = 2, display_precision = False, tab_level = 0, memoization_map = {}, display_attribute = False, display_id = False):
        """ string conversion for operation graph 
            depth:                  number of level to be crossed (None: infty)
            display_precision:      enable/display format display
        """
        new_depth = None 
        if depth != None:
            if  depth < 0: 
                return "" 
        new_depth = (depth - 1) if depth != None else None
            
        tab_str = AbstractOperation.str_del * tab_level
        silent_str = "[S]" if self.get_silent() else ""
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        if 1:
            memoization_map[self] = str_tag
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            pre_str = tab_str + "%s%s%s%s%s ------> %s\n%s" % (self.get_name(), precision_str, silent_str, id_str, attribute_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id) for inp in self.inputs))
            for case in self.case_map:
                case_str = ""
                if isinstance(case, tuple):
                  case_str = ", ".join([str(sc) for sc in case])
                else:
                  case_str = "%s" % case
                pre_str += "Case: %s" %  case_str #.get_str(new_depth, display_precision, tab_level = 0, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id)
                pre_str += "%s" %  self.case_map[case].get_str(new_depth, display_precision, tab_level = tab_level + 2, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id)
            return pre_str

class VectorElementSelection(ArithmeticOperationConstructor("VectorElementSelection", arity = 2)):
    implicit_arg_precision = {
        ML_Float2: ML_Binary32,
        ML_Float4: ML_Binary32,
        ML_Float8: ML_Binary32,

        ML_Double2: ML_Binary64,
        ML_Double4: ML_Binary64,
        ML_Double8: ML_Binary64,

        ML_Int2: ML_Int32,
        ML_Int4: ML_Int32,
        ML_Int8: ML_Int32,

        ML_UInt2: ML_UInt32,
        ML_UInt4: ML_UInt32,
        ML_UInt8: ML_UInt32,
    }

    def get_codegen_key(self):
        """ return code generation specific key """
        return None

    def __init__(self, vector, elt_index, **kwords):
        self.__class__.__base__.__init__(self, vector, elt_index, **kwords)
        self.elt_index = elt_index

        # setting implicit precision
        if self.get_precision() == None and  vector.get_precision() != None:
            arg_precision = vector.get_precision()
            if arg_precision in VectorElementSelection.implicit_arg_precision:
                self.set_precision(VectorElementSelection.implicit_arg_precision[arg_precision])

    def get_elt_index(self):
        return self.elt_index


def AdditionN(*args, **kwords):
  """ multiple-operand addition wrapper """
  op_list = [op for op in args]
  while len(op_list) > 1:
    op0 = op_list.pop(0)
    op1 = op_list.pop(0)
    op_list.append(Addition(op0, op1, **kwords))
  return op_list[0]


def Likely(optree, likely_value):
  optree.get_likely = lambda: likely_value
  return optree

# end of doxygen group ml_operations
## @}



if __name__ == "__main__":
  # auto doc
  # TODO: to be fixed
  for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and isinstance(obj, ML_Operation):
      print "operation class: ", obj
