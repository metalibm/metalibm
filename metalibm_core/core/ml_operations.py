# -*- coding: utf-8 -*-

## @package ml_operations
#  Metalibm Description Language basic Operation

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
# created:          Dec 23rd, 2013
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


import operator

from sollya import Interval, SollyaObject, inf, sup
import sollya

S2 = sollya.SollyaObject(2)

from ..utility.log_report import Log
from .attributes import Attributes, attr_init
from .ml_formats import (
    ML_Binary32, ML_SingleSingle, ML_Binary64, ML_DoubleDouble, ML_TripleDouble,
    ML_Int32, ML_UInt32, ML_Void,
    v2float32, v4float32, v8float32,
    v2float64, v4float64, v8float64,
    v2int32, v4int32, v8int32,
    v2uint32, v4uint32, v8uint32,
    ML_FloatingPointException, ML_FPE_Type, FP_SpecialValue,
    ML_AbstractFormat, ML_FloatingPoint_RoundingMode, ML_String,
    ML_Bool,
    ML_FPRM_Type,
    ML_FP_Format,
    ML_Fixed_Format,
    ML_Float,
    ML_Bool_Format
)

from metalibm_core.utility.decorator import safe

## \defgroup ml_operations ml_operations
#  @{


# Documentation:
# range_function: this Operation's method is used to determine a node interval
#                 from its parent
#
#

def empty_range(*args):
    """ empty live-range function: whichever the arguments
        always return None """
    return None

## merge abstract format function
#  @return most generic abstract format to unify args formats
def std_merge_abstract_format(*args):
    has_float = False
    has_integer = False
    has_bool = False
    for arg_type in args:
        if ML_FP_Format.is_fp_format(arg_type): has_float = True
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
    arity = None # None for arbitrary arity
    def arityCheck(self, opsTuple):
        """ Generic check for operation arity """
        return self.arity is None or self.arity == len(opsTuple) 


## implicit operation conversion (from number to Constant when required)
#  @brief This function is called on every operations arguments
#         to legalize them
def implicit_op(op):
    """ implicit constant operand promotion """
    if isinstance(op, ML_Operation):
        return op
    elif isinstance(op, SollyaObject) or isinstance(op, int) or isinstance(op, float):
        return Constant(op)
    elif isinstance(op, ML_FloatingPointException):
        return Constant(op, precision = ML_FPE_Type)
    elif isinstance(op, FP_SpecialValue):
        return Constant(op, precision = op.get_precision())
    elif isinstance(op, ML_FloatingPoint_RoundingMode):
        return Constant(op, precision=ML_FPRM_Type)
    elif isinstance(op , str):
        return Constant(op, precision = ML_String)
    elif op is None:
        return EmptyOperand()
    else:
        print("ERROR: unsupported operand in implicit_op conversion ", op, op.__class__)
        raise Exception()


## Parent for abstract operations
#  @brief parent to Metalibm's abstract operation
class AbstractOperation(ML_Operation):
    name = "AbstractOperation"
    extra_inputs = []
    global_index = 0
    str_del = "  "

    ## init operation handle
    def __init__(self, **init_map):
        self.attributes = Attributes(**init_map)
        self.index = AbstractOperation.global_index; AbstractOperation.global_index += 1
        self.get_handle().set_node(self)

    ## extract the High part of the Node
    @property
    def hi(self):
        base_precision = self.precision.get_base_format()
        comp_precision = None if base_precision is None else base_precision.get_limb_precision(0)
        op_tag = self.get_tag()
        limb_tag = op_tag + "_hi" if not op_tag is None else None
        return ComponentSelection(self, specifier=ComponentSelection.Hi, precision=comp_precision, tag=limb_tag)

    @property
    def me(self):
        base_precision = self.precision.get_base_format()
        comp_precision = None if base_precision is None else base_precision.get_limb_precision(1)
        op_tag = self.get_tag()
        limb_tag = op_tag + "_me" if not op_tag is None else None
        return ComponentSelection(self, specifier=ComponentSelection.Me, precision=comp_precision, tag=limb_tag)

    ## extract the Low part of the Node
    @property
    def lo(self):
        base_precision = self.precision.get_base_format()
        comp_precision = None if base_precision is None else base_precision.get_limb_precision(base_precision.limb_num - 1)
        op_tag = self.get_tag()
        limb_tag = op_tag + "_lo" if not op_tag is None else None
        return ComponentSelection(self, specifier=ComponentSelection.Lo, precision=comp_precision, tag=limb_tag)

    def __getitem__(self, index):
        return VectorElementSelection(self, index)
    def __setitem__(self, index, value):
        return ReferenceAssign(VectorElementSelection(self, index), value, precision = value.get_precision())

    ## apply the @p self. bare_range_function to a tuple
    #  of intervals
    def apply_bare_range_function(self, intervals):
        func = self.bare_range_function
        return func(intervals)


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

    def __lshift__(self, op):
        return BitLogicLeftShift(self, implicit_op(op))

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
    def __truediv__(self, op):
        """ implicit division operation between AbstractOperation """
        return Division(self, implicit_op(op))

    ## 2-Operand implicit commuted division operator
    def __rdiv__(self, op):
        """ implicit reflexive division operation between AbstractOperation """
        return Division(implicit_op(op), self)
    def __rtruediv__(self, op):
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

    @property
    def precision(self):
        return self.attributes.get_precision()
    @precision.setter
    def precision(self, value):
        self.attributes.set_precision(value)

    ## set the node output precision
    def set_precision(self, new_precision):
        self.attributes.set_precision(new_precision)


    ## return the complete list of node's input
    #  @return list of nodes
    def get_inputs(self):
        return self.inputs
    ## return @p self's number of inputs
    def get_input_num(self):
        return len(self.inputs)
    ## return a specific input to @p self
    #  @param self Operation node
    #  @param index integer id of the input to extract
    #  @return the @p index-th input of @p self
    def get_input(self, index):
        try:
            return self.inputs[index]
        except IndexError as e:
            Log.report(Log.Error, "input {} is not available from {}", index, self, error=e)
    ## swap an input in node's input list
    #  @param self Operation node
    #  @param index integer id of the input to swap
    #  @param new_input new value of the input to be set
    def set_input(self, index, new_input):
        # FIXME: discard tuple -> list -> tuple transform
        input_list = list(self.inputs)
        input_list[index] = new_input
        self.inputs = tuple(input_list)

    ##
    #  @return the node evaluated live-range (when available)
    def get_interval(self):
        return self.attributes.get_interval()
    @property
    def interval(self):
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

    @property
    def rel_error(self):
        return self.attributes.rel_error
    @rel_error.setter
    def rel_error(self, rel_error):
        self.attributes.rel_error = rel_error

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

    ## Add an extra (hidden) input to the operand's standard input
    def add_to_extra_inputs(self, extra_input):
        self.extra_inputs.append(extra_input)


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

    def __str__(self):
        """ Default conversion of a ML_Operation object to a string """
        return self.get_str(display_precision=True)


    def get_str_descriptor(self, display_precision=True, display_id=False,
                           display_attribute=False, tab_level=0,
                           display_interval=False):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        silent_str = "[S]" if self.get_silent() else ""
        dbg_str = "[DBG]" if self.get_debug() else ""
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        interval_str = "" if not display_interval else "[I={}]".format(self.get_interval())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_leve=tab_level)
        return precision_str + silent_str + interval_str + dbg_str + id_str + attribute_str

    ## string conversion
    #  @param  depth [integer/None] node depth where the display recursion stops
    #  @param  display_precision [boolean] enable/display node's precision display
    #  @param  tab_level number of tab to be inserted left to node's description
    #  @param  memoization_map [dict] hastable to store previously described
    #          node (already generated node tag will be use rather than copying
    #          the full description)
    #  @param  display_attribute [boolean] enable/disable display of node's attributes
    #  @param  display_id [boolean]  enable/disbale display of unique node identified
    #  @return a string describing the node
    def get_str(
            self, depth = 2, display_precision=False,
            tab_level = 0, memoization_map = None,
            display_attribute = False, display_id = False,
            custom_callback = lambda op: "",
            display_interval=False,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        new_depth = None
        if depth != None:
            if  depth < 0:
                return ""
        new_depth = (depth - 1) if depth != None else None

        tab_str = AbstractOperation.str_del * tab_level + custom_callback(self)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        desc_str = self.get_str_descriptor(display_precision, display_id, display_attribute, tab_level, display_interval=display_interval)
        memoization_map[self] = str_tag

        return tab_str + "{name}{desc} -------> {tag}\n{args}".format(
            name=self.get_name(),
            desc=desc_str,
            tag=str_tag,
            args="".join(
                inp.get_str(
                    new_depth, display_precision,
                    tab_level=tab_level + 1,
                    memoization_map=memoization_map,
                    display_attribute=display_attribute,
                    display_id=display_id,
                    custom_callback=custom_callback,
                    display_interval=display_interval
                ) for inp in self.inputs)
        )


    ## virtual function, called after a node's copy
    #  overleaded by inheriter of AbstractOperation
    def finish_copy(self, new_copy, copy_map = {}):
        pass


    ## pure virtual copy  node function
    #  @param copy_map dictionnary of previously built copy, if a node is found within this table, table's value is returned as copy result
    #  @return node's copy (newly generated or memoized)
    def copy(self, copy_map = {}):
        print("Error: copy not implemented")
        print(self, self.__class__)
        raise NotImplementedError

    ## propagate given precision
    #  @param precision
    #  @return None
    def propagate_precision(self, precision, boundary_list = []):
      self.set_precision(precision)
      if not isinstance(self, ML_LeafNode):
        for op in self.inputs:
          if op.get_precision() is None and not op in boundary_list:
            op.propagate_precision(precision, boundary_list)



## Parent for AbstractOperation with no expected input
class ML_LeafNode(AbstractOperation):
    """ """
    pass

class EmptyOperand(ML_LeafNode):
    name = "EmptyOperand"
    def get_str(
            self, depth = None, display_precision = False,
            tab_level = 0, memoization_map = None,
            display_attribute = False, display_id = False,
            custom_callback = lambda op: "",
            display_interval=False,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        id_str        = ("[id=%x]" % id(self)) if display_id else ""
        return AbstractOperation.str_del * tab_level + custom_callback(self) + "EmptyOperand()%s%s%s\n" % (attribute_str, precision_str, id_str)


def is_interval_compatible_object(value):
    """ predicate testing if value is an object from
        a numerical class compatible with sollya.Interval
        constructor

        Args:
            value (object): object to be tested
        Return
            boolean: True if object is compatible with Interval, False otherwise
    """
    if not(isinstance(value, bool)) and isinstance(value, int):
        return True
    elif isinstance(value, float):
        return True
    elif isinstance(value, SollyaObject):
        # TODO: very permissive
        return True
    else:
        return False

def is_leaf_node(node):
    """ Test if node is a leaf one (with no input) """
    return isinstance(node, ML_LeafNode)

## Constant node class
class Constant(ML_LeafNode):
    name = "Constant"
    ## Initializer
    #  @param value numerical value of the constant
    #  @param init_map dictionnary for attributes initialization
    def __init__(self, value, **init_map):
        # value initialization
        AbstractOperation.__init__(self, **init_map)
        self.value = value
        # attribute fields initialization
        # Gitlab's Issue#16 as bool is a sub-class of int, it must be excluded
        # explicitly
        if is_interval_compatible_object(value):
            self.attributes.set_interval(Interval(value))


    ## accessor to the constat value
    #  @return the numerical constant value
    def get_value(self):
        return self.value
    def set_value(self, new_value):
        self.value = new_value

    def get_str(
            self, depth = None, display_precision = False,
            tab_level = 0, memoization_map = None,
            display_attribute = False, display_id = False,
            custom_callback = lambda op: "",
            display_interval=False,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        id_str        = ("[id=%x]" % id(self)) if display_id else ""
        return AbstractOperation.str_del * tab_level + custom_callback(self) + "Cst(%s)%s%s%s\n" % (self.value, attribute_str, precision_str, id_str)


    def copy(self, copy_map = {}):
        """ return a new, free copy of <self> """
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # else define a new and free copy
        new_copy = Constant(self.value)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        return new_copy

    def get_codegen_key(self):
        return None
    ## return empty input list
    def get_inputs(self):
        return []


## class for Variable node, which contains a temporary state of the operation DAG
#  which may have been defined outside the scope of the implementation (input variable)
#  @param var_type = (Variable.Input | Variable.Local)
class AbstractVariable(ML_LeafNode):
    ## Input type for Variable Node
    #  such node is not defined as an input to the function description
    class Input: pass
    class Local: pass
    class Output: pass

    ## Intermediary type for Variable Node
    #  such node is defined within the function description.
    #  It holds an intermediary state
    class Intermediary: pass

    ## constructor
    #  @param tag string name of the Variable object
    #  @param init_map standard ML_Operation attribute dictionnary initialization
    def __init__(self, tag, **init_map):
        AbstractOperation.__init__(self, **init_map)
        assert not tag is None
        self.attributes.set_tag(tag)
        # used to distinguish between input variables (without self.inputs)
        # and intermediary variables
        self.var_type = attr_init(init_map, "var_type", default_value = Variable.Input)

    ## @return the type (Input or Intermediary0 of the Variable node
    def get_var_type(self):
        return self.var_type

    def __call__(self, *args):
        var_type = self.get_precision()
        if not isinstance(var_type, FunctionType):
            Log.report(Log.report("Variable {} of format {} is not callable", self, var_type))
        return FunctionObject(self.get_tag(), var_type.arg_list_precision, var_type.output_format, None)(*args)

    ## generate string description of the Variable node
    def get_str(
            self, depth = None, display_precision = False, tab_level = 0,
            memoization_map = None, display_attribute = False, display_id = False,
            custom_callback = lambda op: "",
            display_interval=False
        ):
        memoization_map = {} if memoization_map is None else memoization_map

        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        id_str        = ("[id=%x]" % id(self)) if display_id else ""
        interval_str = "" if not display_interval else "[I={}]".format(self.get_interval())
        descriptor_str = precision_str + interval_str + attribute_str + id_str
        return AbstractOperation.str_del * tab_level + custom_callback(self) + "%s(%s)%s\n" % (self.name, self.get_tag(), descriptor_str)


    def copy(self, copy_map=None):
        copy_map = {} if copy_map is None else copy_map
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # by default input variable are not copied
        # this behavior can be bypassed by manually
        # defining a copy into copy_map
        if self.get_var_type() == Variable.Input:
            copy_map[self] = self
            return self
        # else define a new and free copy
        new_copy = self.__class__(tag = self.get_tag(), var_type=self.var_type)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        return new_copy

    def get_codegen_key(self):
        return None

class Variable(AbstractVariable):
    """ Base class for non-abstract variable """
    name = "Variable"


class InvalidInterval(Exception):
  """ Invalid interval exception """
  pass

## Check interval validity, verifying that @p lrange
#  is a valid interval
def interval_check(lrange):
    """ check if the argument <lrange> is a valid interval,
        if it is, returns it, else raises an InvalidInterval
        exception """
    if isinstance(lrange, SollyaObject) and lrange.is_range():
        return lrange
    else:
        raise InvalidInterval()

def default_op_interval_getter(node):
    return node.get_interval()

## Extend interval verification to a list of operands
#  an interval is extract from each operand and checked for validity
#  if all intervals are valid, @p interval_op is applied and a
#  resulting interval is returned
def interval_wrapper(self, interval_op, ops, ops_interval_getter=default_op_interval_getter):
    try:
        return interval_op(self, tuple(interval_check(ops_interval_getter(op)) for op in ops))
    except InvalidInterval:
        return None


## Wraps the operation on intervals @p interval_op
#  to an object method which inputs operations trees
def interval_func(interval_op):
    """ interval function builder for multi-arity interval operation """

    def range_function_None(node, ops, ops_interval_getter=default_op_interval_getter):
        return None

    def range_function_wrapper(node, ops, ops_interval_getter=default_op_interval_getter):
        return interval_wrapper(node, interval_op, ops, ops_interval_getter)

    if interval_op == None:
        return range_function_None
    else:
        return range_function_wrapper


class GeneralOperation(AbstractOperation):
    """ parent class for generic operations """
    arity = 2
    bare_range_function = empty_range
    error_function = None

    def __init__(self, *ops, **init_map):
        AbstractOperation.__init__(self, **init_map)
        # inputs assignation is done before arity checking
        # because some arityCheck method implementation rely on
        # inputs being populated first (e.g. ConditionalBranch)
        self.inputs = tuple(implicit_op(op) for op in ops)
        # checking arity
        if not self.arityCheck(ops):
            #python3
            import inspect
            from inspect import currentframe, getframeinfo

            #frameinfo = inspect.getframeinfo(inspect.currentframe())
            frameinfo = inspect.getouterframes(inspect.currentframe())[1]
            Log.report(Log.Error, "number of operands ({}) mismatch with Operation's arity {}, {}, {}, {}",
                       len(ops), self.arity, self.__class__, frameinfo.filename, frameinfo.lineno)
    def get_codegen_key(self):
        return None
    def copy(self, copy_map=None):
        """ base function to copy an abstract operation object,
            copy_map is a memoization hashtable which can be use to factorize
            copies """
        copy_map = {} if copy_map is None else copy_map
        # test for previous definition in memoization map
        if self in copy_map:
            return copy_map[self]
        # else define a new and free copy
        new_copy = self.__class__(*tuple(copy_map[op] if op in copy_map else op.copy(copy_map) for op in self.inputs), __copy = True)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        self.finish_copy(new_copy, copy_map)
        return new_copy

    def range_function(self, ops, ops_interval_getter=lambda op: op.get_interval()):
        """ Generic wrapper for node range evaluation """
        try:
            return self.bare_range_function(
                tuple(interval_check(ops_interval_getter(op)) for op in ops)
            )
        except InvalidInterval:
            return None


class ControlFlowOperation(GeneralOperation):
    """ Parent for all control-flow operation """

class ML_ArithmeticOperation(GeneralOperation):
    """ base class for all arithmetic operation that may depend
        on floating-point context (rounding mode for example) """
    error_function = None

    def get_codegen_key(self):
        return None

## Bitwise bit AND operation
class BitLogicAnd(ML_ArithmeticOperation):
    name = "BitLogicAnd"
    arity = 2
## Bitwise bit OR operation
class BitLogicOr(ML_ArithmeticOperation):
    name = "BitLogicOr"
    arity = 2
## Bitwise bit exclusive-OR operation
class BitLogicXor(ML_ArithmeticOperation):
    name = "BitLogicXor"
    arity = 2
## Bitwise negate operation
class BitLogicNegate(ML_ArithmeticOperation):
    name = "BitLogicNegate"
    arity = 1
## Bit Logic Right Shift
#   2-operand operation, first argument is the value to be shifted
#   the second is the shift amount
class BitLogicRightShift(ML_ArithmeticOperation):
    name = "BitLogicRightShift"
    arity = 2
## Bit Arithmetic Right Shift
#   2-operand operation, first argument is the value to be shifted
#   the second is the shift amount
class BitArithmeticRightShift(ML_ArithmeticOperation):
    name = "BitArithmeticRightShift"
    arity = 2
## Bit Left Shift
#   2-operand operation, first argument is the value to be shifted
#   the second is the shift amount
class BitLogicLeftShift(ML_ArithmeticOperation):
    name = "BitLogicLeftShift"
    arity = 2


## Absolute value operation
#  Expects a single argument and returns
#  its absolute value
class Abs(ML_ArithmeticOperation):
    """ abstract absolute value operation """
    name = "Abs"
    arity = 1
    def bare_range_function(self, ops):
        return safe(abs)(ops[0])

## Unary negation value operation
#  Expects a single argument and returns
#  its opposite value
class Negation(ML_ArithmeticOperation):
    """ abstract negation """
    name = "Negation"
    arity = 1
    def bare_range_function(self, ops):
        return safe(operator.__neg__)(ops[0])




class SpecifierOperation(object):
    """ this class of operation uses a specific attribute called specifier
        as code generation key """
    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def finish_copy(self, new_copy, copy_map=None):
        new_copy.specifier = self.specifier

class ComponentSelectionSpecifier(object):
    """ Parent class for all component selection specifier """
    pass

class Split(ML_ArithmeticOperation):
    """ Splitting Vector in two halves sub-vectors """
    name = "Split"
    arity = 1

class ComponentSelection(SpecifierOperation, ML_ArithmeticOperation):
    arity = 1
    name = "ComponentSelection"
    class Hi(ComponentSelectionSpecifier): pass
    class Me(ComponentSelectionSpecifier): pass
    class Lo(ComponentSelectionSpecifier): pass
    class Field(ComponentSelectionSpecifier):
        def __init__(self, field_index):
            self.field_index = field_index
    class NamedField(ComponentSelectionSpecifier):
        def __init__(self, field_name):
            self.field_name = field_name

    implicit_arg_precision = {
        ML_SingleSingle: ML_Binary32,
        ML_DoubleDouble: ML_Binary64,
        ML_TripleDouble: ML_Binary64,
    }

    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def finish_copy(self, new_copy, copy_map=None):
        SpecifierOperation.finish_copy(self, new_copy, copy_map)

    def __init__(self, *args, **kwords):
        self.specifier = attr_init(kwords, "specifier", ComponentSelection.Hi)
        ML_ArithmeticOperation.__init__(self, *args, **kwords)

        # setting implicit precision
        if self.get_precision() == None and len(args) > 0 and args[0].get_precision() != None:
            arg_precision = args[0].get_precision()
            if arg_precision in ComponentSelection.implicit_arg_precision:
                self.set_precision(ComponentSelection.implicit_arg_precision[arg_precision])


class BuildFromComponent(ML_ArithmeticOperation):
    name = "BuildFromComponent"
    def __init__(self, *args, **kwords):
        ML_ArithmeticOperation.__init__(self, *args, **kwords)
        self.arity = len(args)


class FMASpecifier(object):
    """ Common parent to all Test specifiers """
    pass


## Fused Multiply-Add (FMA) operation
#
# specifier by one of the following  specifiers:
# - FusedMultiplyAdd.Standard FMA         op0 * op1 + op2
# - FusedMultiplyAdd.Subtract FMA         op0 * op1 - op2
# - FusedMultiplyAdd.Negate FMA         - op0 * op1 - op2
# - FusedMultiplyAdd.SubtractNegate FMA - op0 * op1 + op2
# - FusedMultiplyAdd.SubtractNegate FMA - op0 * op1 + op2
# - FusedMultiplyAdd.DotProduct         op0 * op1 + op2 * op3
# - FusedMultiplyAdd.DotProductNegate   op0 * op1 + op2 * op3
class FusedMultiplyAdd(SpecifierOperation, ML_ArithmeticOperation):
    """ abstract fused multiply and add operation op0 * op1 + op2 """
    name = "FusedMultiplyAdd"
    arity = 3
    ## standard FMA op0 * op1 + op2
    class Standard(FMASpecifier):
        """ op0 * op1 + op2 """
        name = "Standard"
        arity = 3
        @staticmethod
        def range_function(optree, ops):
            return ops[0] * ops[1] + ops[2]
    ## Subtract FMA op0 * op1 - op2
    class Subtract(FMASpecifier):
        """ op0 * op1 - op2 """
        name = "Subtract"
        arity = 3
        @staticmethod
        def range_function(optree, ops):
            return ops[0] * ops[1] - ops[2]
    ## Negate FMA - op0 * op1 - op2
    class Negate(FMASpecifier):
        """ -op0 * op1 - op2 """
        name = "Negate"
        arity = 3
        @staticmethod
        def range_function(optree, ops):
            return - ops[0] * ops[1] - ops[2]
    ## Subtract Negate FMA - op0 * op1 + op2
    class SubtractNegate(FMASpecifier):
        """ -op0 * op1 + op2 """
        name = "SubtractNegate"
        arity = 3
        @staticmethod
        def range_function(optree, ops):
            return - ops[0] * ops[1] + ops[2]
    ## Dot Product op0 * op1 + op2 * op3
    class DotProduct(FMASpecifier):
        """ op0 * op1 + op2 * op3 """
        name = "DotProduct"
        arity = 4
        @staticmethod
        def range_function(optree , ops):
            return ops[0] * ops[1] + ops[2] * ops[3]
    ## Dot Product Negate op0 * op1 - op2 * op3
    class DotProductNegate(FMASpecifier):
        """ op0 * op1 - op2 * op3 """
        name = "DotProductNegate"
        arity = 4
        @staticmethod
        def range_function(optree, opts):
            return ops[0] * ops[1] - ops[2] * ops[3]

    def bare_range_function(self, ops):
        """ FMA's range_function is determined by specifiers """
        return self.specifier.range_function(self, ops)

    def __init__(self, *args, **kwords):
        self.specifier = attr_init(kwords, "specifier", FusedMultiplyAdd.Standard)
        # indicates whether a base operation commutation has been processed
        # used in proof generation to make sure generated code
        # is closest as possible to evaluation scheme
        self.commutated = attr_init(kwords, "commutated", False)
        ML_ArithmeticOperation.__init__(self, *args, **kwords)
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
    """ op0 * op1 + op2 """
    kwords["specifier"] = FusedMultiplyAdd.Standard
    return FusedMultiplyAdd(op0, op1, op2, **kwords)

def FMS(op0, op1, op2, **kwords):
    """ op0 * op1 - op2 """
    kwords["specifier"] = FusedMultiplyAdd.Subtract
    return FusedMultiplyAdd(op0, op1, op2, **kwords)

def FMSN(op0, op1, op2, **kwords):
    """ - op0 * op1 + op2 """
    kwords["specifier"] = FusedMultiplyAdd.SubtractNegate
    return FusedMultiplyAdd(op0, op1, op2, **kwords)

class Addition(ML_ArithmeticOperation):
    """ 2-operand Addition node class """
    name = "Addition"
    arity = 2
    def bare_range_function(self, ops):
        return safe(operator.__add__)(ops[0], ops[1])

class Subtraction(ML_ArithmeticOperation):
    """ Subtraction operation class node """
    name = "Subtraction"
    arity = 2
    def bare_range_function(self, ops):
        return safe(operator.__sub__)(ops[0], ops[1])


class Multiplication(ML_ArithmeticOperation):
    """  Multiplication operation node class"""
    name = "Multiplication"
    arity = 2
    def bare_range_function(self, ops):
        return safe(operator.__mul__)(ops[0], ops[1])


class Division(ML_ArithmeticOperation):
    """ abstract addition """
    name = "Division"
    arity = 2
    def bare_range_function(self, ops):
        return safe(operator.__truediv__)(ops[0], ops[1])

class Extract(ML_ArithmeticOperation):
    """ (UNUSED) abstract word or vector extract-from-vector operation """
    name = "Extract"

class Splat(ML_ArithmeticOperation):
    """ Fill a vector by broadcasting a scalar input to all elements """
    name = "Splat"
    arity = 1

def numerical_modulo(lhs, rhs):
    """ compute lhs % rhs """
    if (isinstance(rhs, SollyaObject) and rhs.is_range()) or isinstance(rhs, (MetaInterval, MetaIntervalList)):
        # TODO: manage properly range for euclidian modulo operation
        return rhs
    else:
        raise NotImplementedError("support for {} object not implemented in numerical_modulo".format(rhs))


class Modulo(ML_ArithmeticOperation):
    """ modulo operation node class """
    name = "Modulo"
    arity = 2
    def bare_range_function(self, ops):
        return safe(numerical_modulo)(ops[0], ops[1])


class NearestInteger(ML_ArithmeticOperation):
    """ nearest integer operation node class"""
    name = "NearestInteger"
    arity = 1
    def bare_range_function(self, ops):
        return safe(sollya.nearestint)(ops[0])

class Permute(ML_ArithmeticOperation):
    """ (UNSUED) abstract word-permutations inside a vector operation """
    name = "Permute"

## Round to an integer value, rounding towards zero
#  returns the minimal integer greater than or equal to the node's input
class Ceil(ML_ArithmeticOperation):
  """ Round to integer upward """
  arity = 1
  name = "Ceil"
  range_function = interval_func(lambda self, ops: safe(sollya.ceil)(ops[0]))
  bare_range_function = lambda self, ops: safe(sollya.ceil)(ops[0])

## Round to an integer value, rounding towards zero
class Trunc(ML_ArithmeticOperation):
  """ Round to integer towards zero """
  arity = 1
  name = "Trunc"
  range_function = interval_func(lambda self, ops: None)#safe(sollya.trunc)(ops[0]))
  bare_range_function = lambda self, ops: None

## Round to an integer value, rounding towards zero
#  returns the maximal integer less than or equal to the node's input
class Floor(ML_ArithmeticOperation):
  """ Round to integer downward """
  arity = 1
  name = "Floor"
  range_function = interval_func(lambda self, ops: safe(sollya.floor)(ops[0]))
  bare_range_function = lambda self, ops: safe(sollya.floor)(ops[0])


class PowerOf2(ML_ArithmeticOperation):
    """ abstract power of 2 operation node class, computes 2^inputs[0] """
    name = "PowerOf2"
    arity = 1
    def bare_range_function(self, ops):
        return safe(operator.__pow__)(S2, ops[0])


## Side effect operator which stops the function and
#  returns a result value
class Return(GeneralOperation):
    """ abstract return value operation """
    name = "Return"
    arity = 1
    def bare_range_function(self, ops):
        return ops[0]

    def arityCheck(self, opsTuple):
        # 0 for void Return
        # 1 for Return with value
        return len(opsTuple) in [0, 1]


## Memory Load from a Multi-Dimensional
#  The first argument is the table, following arguments
#  are the table index in each dimension (from 1 to ...)
class TableLoad(ML_ArithmeticOperation):
    """ abstract load from a table operation """
    name = "TableLoad"
    arity = None
    def bare_range_function(self, ops):
        # TODO/FIXME: coarse-grained: could be refined by using index
        # interval to refine sub-table interval
        return ops[0]

    def arityCheck(self, opsTuple):
        # 2 (1D index), and 3 (2D index) are valid arity for TableLoad
        return len(opsTuple) in [2, 3]

## Memory Store to a Multi-Dimensional
#  The first argument is the table to store to,
#  the second argument is the value to be stored
#  the following arguments are the table index
#  in each dimension (from 1 to ...)
#   By default the precision of this operation is ML_Void
class TableStore(ML_ArithmeticOperation):
    """ abstract store to a table operation """
    name = "TableStore"
    arity = 3

class VectorUnpack(SpecifierOperation, ML_ArithmeticOperation):
    """ abstract vector unpacking operation """
    name = "VectorUnpack"
    # High and Low specifiers for Unpack operation
    class Hi(object): name = 'Hi'
    class Lo(object): name = 'Lo'

    def __init__(self, *args, **kwords):
        self.specifier = attr_init(kwords, "specifier", VectorUnpack.Lo)
        ML_ArithmeticOperation.__init__(*args, **kwords)

    def get_name(self):
        return  "VectorUnpack.{}".format(self.specifier.name)

    def get_codegen_key(self):
        return self.specifier


## Compute the union of two intervals
def interval_union(int0, int1):
    return Interval(min(inf(int0), inf(int1)), max(sup(int0), sup(int1)))

## Ternary selection operator: the first operand is a condition
#  when True the node returns the 2nd operand else its returns the
#  3rd operand
class Select(ML_ArithmeticOperation):
    """ Ternary operator """
    name = "Select"
    arity = 3
    def range_function(self, ops, ops_interval_getter=default_op_interval_getter):
        # we can not rely on bare_range_function as range evaluation for ops[0]
        # may throw an exception
        return safe(interval_union)(ops_interval_getter(ops[1]), ops_interval_getter(ops[2]))


## Computes the maximum of its 2 operands
#def Max(op0, op1, **kwords):
#    return Select(Comparison(op0, op1, specifier = Comparison.Greater), op0, op1, **kwords)

def min_interval(a, b):
    return Interval(min(inf(a), inf(b)), min(sup(a), sup(b)))
def max_interval(a, b):
    return Interval(max(inf(a), inf(b)), max(sup(a), sup(b)))

class Min(ML_ArithmeticOperation):
    """ Minimum of 2 inputs """
    name = "Min"
    arity = 2
    def bare_range_function(self, ops):
        return min_interval(ops[0], ops[1])
class Max(ML_ArithmeticOperation):
    """ Maximum of 2 inputs """
    name = "Max"
    arity = 2
    def bare_range_function(self, ops):
        return max_interval(ops[0], ops[1])

## Computes the minimum of its 2 operands
# def Min(op0, op1, **kwords):
#    return Select(Comparison(op0, op1, specifier = Comparison.Less), op0, op1, **kwords)

## Control-flow loop construction
#  1st operand is a loop initialization block
#  2nd operand is a loop exit c ondition block
#  3rd operand is a loop body block
class Loop(ControlFlowOperation):
    """ abstract loop constructor
        loop (init_statement, exit_condition, loop_body)
    """
    name = "Loop"
    # Some operations inheriting from Loop (e.g. RangeLoop)
    # have less than 3 in arity (e.g. 2 for RangeLoop)
    arity = 3

    def arityCheck(self, opsTuple):
        # 3 for standard Loop
        # 2 for RangeLoop
        # are valid arities for ConditionBlock
        return len(opsTuple) in [2, 3]

class WhileLoop(ControlFlowOperation):
    """ abstract while loop constructor
        WhileLoop (exit_condition, loop_body)
    """
    name = "WhileLoop"
    arity = 2

## Control-flow if-then-else construction
#  1st operand is a condition expression
#  2nd operand is the true-condition branch
#  3rd operand (optinal) is the false-condition branch
class ConditionBlock(ControlFlowOperation):
    """ abstract if/then(/else) block """
    name = "ConditionBlock"
    arity = None

    def __init__(self, *args, **kwords):
        """ condition block initialization """
        super().__init__(*args, **kwords)
        self.parent_list = []
        # statement being executed before the condition or either of the branch is executed
        self.pre_statement = Statement()
        self.extra_inputs = [self.pre_statement]

    def arityCheck(self, opsTuple):
        # 2 (if (cond) then st0 ) and
        # 3 (if (cond) then st0 else st1)
        # are valid arities for ConditionBlock
        return len(opsTuple) in [2, 3]

    def set_extra_inputs(self, new_extra_inputs):
        self.extra_inputs = new_extra_inputs

    def set_parent_list(self, parent_list):
        self.parent_list = parent_list

    def get_pre_statement(self):
        return self.pre_statement

    def add_to_pre_statement(self, optree):
        raise NotImplementedError
        self.pre_statement.add(optree)

    def push_to_pre_statement(self, optree):
        raise NotImplementedError
        self.pre_statement.push(optree)

    def finish_copy(self, new_copy, copy_map = None):
        copy_map = {} if copy_map is None else copy_map
        new_copy.pre_statement = self.pre_statement.copy(copy_map)
        new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]
        new_copy.parent_list  = [op.copy(copy_map) for op in self.parent_list]


class Conversion(ML_ArithmeticOperation):
    """ abstract conversion operation """
    name = "Conversion"
    arity = 1

def is_conversion(optree):
    """ Predicate to test Conversion class operations """
    return isinstance(optree, Conversion)

class TypeCast(ML_ArithmeticOperation):
    """ abstract conversion operation """
    name = "TypeCast"
    arity = 1
    def bare_range_function(self, ops):
        return None

def is_typecast(optree):
    return isinstance(optree, TypeCast)

class Dereference(ML_ArithmeticOperation):
    """ abstract pointer derefence operation """
    name = "Dereference"
    arity = 1

class ReferenceAssign(GeneralOperation):
    """ abstract assignation to reference operation """
    name = "ReferenceAssign"
    arity = 2

class ExponentInsertion(SpecifierOperation, ML_ArithmeticOperation):
    """ insertion of a number in the exponent field of a floating-point value """
    name = "ExponentInsertion"
    arity = 1
    class Default: pass
    class NoOffset: pass

    def __init__(self, expValue, **kwords):
        ML_ArithmeticOperation.__init__(self, expValue, **kwords)
        self.specifier = attr_init(kwords, "specifier", default_value = ExponentInsertion.Default)

    def get_codegen_key(self):
        """ return code generation specific key """
        return self.specifier

    def bare_range_function(self, op_interval):
        if op_interval is None:
            return None
        else:
            # TODO/FIXME: manage cases when inf/nan are part of op_interval
            lo_bound = S2**inf(op_interval[0])
            hi_bound = S2**sup(op_interval[0])
            return Interval(lo_bound, hi_bound)

class MantissaExtraction(ML_ArithmeticOperation):
    """ return the input's significand/mantissa as a floating-point value, whose absolute
        value lies between 1 (included) and 2 (excluded), input sign is kept
        unmodified  """
    name = "MantissaExtraction"
    arity = 1

    def bare_range_function(self, op_interval):
        # TODO/FIXME upper bound 2 could be refined to 2 - 2**-mantissa_size
        # if mantissa size if known
        return Interval(-2, 2)

class ExponentExtraction(ML_ArithmeticOperation):
    """ extraction of the exponent field of a floating-point value
        the result is the unbiased (real) exponent, that is an
        integer e such that the input node can be written s.m.2^e
        with m in [1, 2), s in {-1, 1} """
    name = "ExponentExtraction"
    arity = 1

    def bare_range_function(self, input_intervals):
        op_interval = input_intervals[0]
        if op_interval is None:
            return None
        else:
            # TODO/FIXME: manage cases when inf/nan are part of op_interval
            lo_bound = sollya.floor(sollya.log2(inf(abs(op_interval))))
            hi_bound = sollya.floor(sollya.log2(sup(abs(op_interval))))
            return Interval(lo_bound, hi_bound)

class RawExponentExtraction(ML_ArithmeticOperation):
    """ raw extraction of the exponent field of a floating-point value
        the result is the biased exponent, that is an
        integer e such that the input node can be written s.m.2^e+b
        with m in [1, 2), s in {-1, 1} and b a bias a format-specific
        paramater (e.g -127 for binary32) """
    name = "RawExponentExtraction"
    arity = 1

    def bare_range_function(self, input_intervals):
        op_interval = input_intervals[0]
        if op_interval is None:
            return None
        else:
            # TODO/FIXME: manage cases when inf/nan are part of op_interval
            lo_bound = sollya.floor(sollya.log2(inf(abs(op_interval))))
            hi_bound = sollya.floor(sollya.log2(sup(abs(op_interval))))
            return Interval(lo_bound, hi_bound)

## Unary operation, count the number of leading zeros in the operand
#  If the operand equals 0, then the result is the bit size of the
#  operand
class CountLeadingZeros(ML_ArithmeticOperation):
    name = "CountLeadingZeros"
    arity = 1

    def range_function(self, ops, ops_interval_getter=None):
        op = ops[0]
        if op.get_precision() is None:
            return None
        else:
            # TODO/FIXME: could be refined by looking at ops range
            # alongside its format bit_size
            return Interval(0, op.get_precision().get_bit_size())

class TestSpecifier(object):
    """ Common parent to all Test specifiers """
    pass

class LikelyPossible(object):
    """ likely true or false """
    pass

class BooleanOperation(object):
    """ Boolean operation parent """
    def __init__(self, likely):
        """ # likely indicate if the boolean operation is likely
            #  -> to be True (likely = True)
            #  -> to be False (likely = False)
            #  -> to be either True or False (likely = LikelyPossible)
            #  -> undetermined (likely = None)"""
        self.likely = likely
        # default precision is ML_Bool
        self.precision = ML_Bool if self.precision is None else self.precision

    def get_likely(self):
        """ return likely value """
        return self.likely

    def set_likely(self, likely):
        """ set likely value """
        self.likely = likely

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.likely = self.likely

## Extract likely properties if @p optree is a BooleanOperation
#  else returns None
def get_arg_likely(optree):
  return optree.get_likely() if isinstance(optree, BooleanOperation) else None

def logic_operation_init(self, *args, **kwords):
    self.__class__.__base__.__init__(self, *args, **kwords)
    BooleanOperation.__init__(self, attr_init(kwords, "likely"))
    # if likely has not been overloaded
    # trying to determine it with respect to input likeliness
    if self.likely == None:
        self.likely = self.likely_function(*tuple(get_arg_likely(arg) for arg in args))


class LogicOperation(BooleanOperation, ML_ArithmeticOperation):
    """ base class for Logical Operation """
    def __init__(self, *args, **kw):
        ML_ArithmeticOperation.__init__(self, *args, **kw)
        BooleanOperation.__init__(self, attr_init(kw, "likely"))
        # if likely has not been overloaded
        # trying to determine it with respect to input likeliness
        if self.likely == None:
            self.likely = self.likely_function(*tuple(get_arg_likely(arg) for arg in args))

class LogicalAnd(LogicOperation):
    name = "LogicalAnd"
    arity = 2

    def likely_function(self, op0, op1):
        return op0 and op1

class LogicalOr(LogicOperation):
    name = "LogicalOr"
    arity = 2

    def likely_function(self, op0, op1):
        return op0 or op1

class LogicalNot(LogicOperation):
    name = "LogicalNot"
    arity = 1

    def likely_function(self, op):
        return not op


class Test(SpecifierOperation, BooleanOperation, ML_ArithmeticOperation):
    """ Abstract Test operation class """
    name = "Test"
    arity = None
    class IsNaN(TestSpecifier):
        name = "IsNaN"
        arity = 1
    class IsQuietNaN(TestSpecifier):
        name = "IsQuietNaN"
        arity = 1
    class IsSignalingNaN(TestSpecifier):
        name = "IsSignalingNaN"
        arity = 1
    class IsInfty(TestSpecifier):
        name = "IsInfty"
        arity = 1
    class IsPositiveInfty(TestSpecifier):
        name = "IsPositiveInfty"
        arity = 1
    class IsNegativeInfty(TestSpecifier):
        name= "IsNegativeInfty"
        arity=1
    class IsIEEENormalPositive(TestSpecifier):
        name= "IsIEEENormalPositive"
        arity=1
    class IsInfOrNaN(TestSpecifier):
        name= "IsInfOrNaN"
        arity=1
    class IsZero(TestSpecifier):
        name= "IsZero"
        arity=1
    class IsPositiveZero(TestSpecifier):
        name= "IsPositiveZero"
        arity=1
    class IsNegativeZero(TestSpecifier):
        name= "IsNegativeZero"
        arity=1
    class IsSubnormal(TestSpecifier):
        name= "IsSubnormal"
        arity=1
    class CompSign(TestSpecifier):
        name= "CompSign"
        arity=1
    class SpecialCases(TestSpecifier):
        name= "SpecialCases"
        arity=1
    class IsInvalidInput(TestSpecifier):
        name= "IsInvalidInput"
        arity=1
    class IsMaskAllZero(TestSpecifier):
        """ predicate checking if all lanes of a vector mask
            are equal to zero """
        name= "IsMaskAllZero"
        arity=1
    class IsMaskNotAllZero(TestSpecifier):
        """ predicate checking if at least one lane of a vector mask
            is not equal to zero """
        name= "IsMaskNotAllZero"
        arity=1
    class IsMaskAnyZero(TestSpecifier):
        """ predicate checking if at least one lane of a vector mask is equal
            to zero """
        name= "IsMaskAnyZero"
        arity=1
    class IsMaskNotAnyZero(TestSpecifier):
        """ predicate checking if no lane of a vector mask is equal to zero """
        name= "IsMaskNotAnyZero"
        arity=1

    def __init__(self, *args, **kwords):
        SpecifierOperation.__init__(self)
        ML_ArithmeticOperation.__init__(self, *args, **kwords)
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

class ComparisonSpecifier(object):
    opcode = "UNDEF ComparisonSpecifier opcode"
    symbol = "UNDEF ComparisonSpecifier symbol"


## Comparison operator
class Comparison(BooleanOperation, SpecifierOperation, ML_ArithmeticOperation):
    """ Abstract Comparison operation """
    name = "Comparison"
    arity = 2

    class Equal:
        name, opcode, symbol = "Equal", "eq", "=="
    class NotEqual:
        name, opcode, symbol = "NotEqual", "ne", "!="
    class Less:
        name, opcode, symbol = "Less",  "lt", "<"
    class LessOrEqual:
        name, opcode, symbol = "LessOrEqual", "le", "<="
    class Greater:
        name, opcode, symbol = "Greater", "gt", ">"
    class GreaterOrEqual:
        name, opcode, symbol = "GreaterOrEqual", "ge", ">="
    class LessSigned:
        name, opcode, symbol =  "LessSigned",  "lt", "<"
    class LessOrEqualSigned:
        name, opcode, symbol = "LessOrEqualSigned", "le", "<="
    class GreaterSigned:
        name, opcode, symbol = "GreaterSigned", "gt", ">"
    class GreaterOrEqualSigned:
        name, opcode, symbol = "GreaterOrEqualSigned", "ge", ">="


    def __init__(self, *args, **kwords):
        ML_ArithmeticOperation.__init__(self, *args, **kwords)
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


def Equal(op0, op1, precision=ML_Bool, **kwords):
    """ syntactic sugar for equality comparison """
    return Comparison(op0, op1, specifier=Comparison.Equal, precision= precision, **kwords)


def NotEqual(op0, op1, precision=ML_Bool, **kwords):
    """ syntactic sugar for non-equality comparison """
    return Comparison(op0, op1, specifier=Comparison.NotEqual, precision=precision, **kwords)


## Sequential statement block, can have an arbitrary number of
#  sub-statement operands. Each of those is executed sequentially in
#  operands order
# Basic imperative-style Statement (list of separate operations, returning
#  void)
class Statement(ControlFlowOperation):
    name = "Statement"
    arity = None # multi-ary
    def __init__(self, *args, **kwords):
        ControlFlowOperation.__init__(self, *args, **kwords)
        self.arity = len(args)

    # add a new statement at the end of the inputs list
    # @param optree ML_Operation object added at the end of inputs list
    def add(self, optree):
        self.inputs = self.inputs + (optree,)
        self.arity += 1

    # push a new statement at the beginning of the inputs list
    # @param optree ML_Operation object added at the end of inputs list
    def push(self, optree):
        """ add a new unary statement at the beginning of the input list """
        self.inputs = (optree,) + self.inputs
        self.arity += 1

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.arity = self.arity


class FieldExtraction(ML_ArithmeticOperation):
    name = "FieldExtraction"
    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        self.arity = len(args)

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.arity = self.arity


class DivisionSeed(ML_ArithmeticOperation):
    """ Seed for the division operation """
    arity = 2
    name = "DivisionSeed"
class ReciprocalSeed(ML_ArithmeticOperation):
    """ Seed for the reciprocal operation """
    arity = 1
    name = "ReciprocalSeed"

    def bare_range_function(self, ops):
        op_range = ops[0]
        # TODO/FIXME: not taking into account op accuracy
        return  1 / op_range

class ReciprocalSquareRootSeed(ML_ArithmeticOperation):
    arity = 1
    name = "ReciprocalSquareRootSeed"

    def bare_range_function(self, ops):
        op_range = ops[0]
        # TODO/FIXME: not taking into account op accuracy
        return  1 / sollya.sqrt(op_range)

class Subnormalize(ML_ArithmeticOperation):
    """ Subnormalize(op, shift) round op as if its exponent was (emin - shift)
        where emin is the minimal normal exponent for the current format  """
    name = "Subnormalize"
    arity = 2
    
    @staticmethod
    def abstract_type_rule(optree, *ops):
        return optree.get_precision()
    @staticmethod
    def instantiated_type_rule(backend, op, dprec):
        return op.get_precision()


class CopySign(ML_ArithmeticOperation):
    """ CopySign(x, y) = abs(x) . sign(y) """
    name = "CopySign"
    arity = 2

    @staticmethod
    def abstract_type_rule(optree, *ops):
        return std_merge_abstract_format(ops[0].get_precision(), ops[1].get_precision())
    @staticmethod
    def instantiated_type_rule(backend, op, dprec):
        return backend.merge_ops_abstract_format(op, op.inputs)


class ReadTimeStamp(GeneralOperation):
    name = "ReadTimeStamp"
    arity = 0

    @staticmethod
    def abstract_type_rule(optree, *ops):
        return optree.get_precision()
    @staticmethod
    def instantiated_type_rule(backend, op, dprec):
        return backend.merge_abstract_format(op, op.inputs)


class ExceptionOperation(GeneralOperation):
    """ Operation manipulating exceptions """
    pass


class ClearException(ExceptionOperation):
    """ Clear any previous exception / flags from the default
        environment """
    name = "ClearException"
    arity = 0


class RaiseException(ExceptionOperation):
    """ raise a given exception / flag in the default environment"""
    name = "RaiseException"
    arity = 1


class FunctionType(object):
    """ Function prototype object, should be close to ML_Format """
    def __init__(self, name, arg_list_precision, output_format, attributes=None):
        self.name = name
        self.arg_list_precision = arg_list_precision
        self.output_format = output_format
        self.attributes = [] if attributes is None else attributes

    def get_name(self, language=None):
        return self.name
    @property
    def arity(self):
        # as self.arg_list_precision is volatile and can be updated
        # silently, arity is updated dynamically upon read
        return len(self.arg_list_precision)


class FunctionObject(object):
    """ Object to wrap a function """
    def __init__(self, name, arg_list_precision, output_format, generator_object, attributes=None, range_function=None):
        self.function_type = FunctionType(name, arg_list_precision, output_format, attributes)
        self.generator_object = generator_object
        self.range_function = range_function

    @property
    def name(self):
        return self.function_type.name
    @property
    def output_format(self):
        return self.function_type.output_format
    @property
    def arity(self):
        return self.function_type.arity

    def __call__(self, *args, **kwords):
        call_kwords = {"precision": self.output_format}
        call_kwords.update(kwords)
        return FunctionCall(self, *args, **call_kwords)

    def get_declaration(self, code_generator, language=None):
        """ Generate code declaration for FunctionObject """
        return code_generator.get_function_declaration(self.function_type, language=language)

    def get_precision(self):
        return self.output_format

    def get_arity(self):
        return self.arity

    def get_arg_precision(self, arg_index):
        if arg_index >= len(self.function_type.arg_list_precision):
            Log.report(
                Log.Error,
                "arg indexed {} does not exist in function {} expecting {}",
                arg_index, self.name, self.function_type.arg_list_precision
            )
        return self.function_type.arg_list_precision[arg_index]

    def get_arg_precision_tuple(self):
        return tuple(self.function_type.arg_list_precision)

    def get_generator_object(self):
        return self.generator_object

    def get_function_name(self):
        return self.name

    def evaluate_range(self, ops):
        return self.range_function(*ops)


class FunctionCall(GeneralOperation):
    name = "FunctionCall"
    def __init__(self, function_object, *args, **kwords):
        self.function_object = function_object
        self.arity = len(args)
        GeneralOperation.__init__(self, *args, **kwords)

    def get_precision(self):
        return self.function_object.get_precision()

    def get_function_object(self):
        return self.function_object

    def get_name(self):
      return "FunctionCall to %s" % self.get_function_object().get_function_name()

    def bare_range_function(self, ops):
        result =  self.get_function_object().evaluate_range(ops)
        print("{}({}) = {} ".format(self.function_object.name, ops, result))
        return result

    @staticmethod
    def propagate_format_to_cst(optree):
        """ propagate new_optree_format to Constant operand of <optree> with abstract precision """
        index_list = range(len(optree.inputs))
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
        copy_map[self] = new_copy
        self.finish_copy(new_copy, copy_map)
        return new_copy


class SwitchBlock(ControlFlowOperation):
    """ switch block (multiple selection) construct """
    name = "Switch"
    def __init__(self, switch_value, case_map, **kwords):
        SwitchBlock.__base__.__init__(self, switch_value, **kwords)
        self.extra_inputs = [case_map[case] for case in case_map]
        self.parent_list = []
        # statement being executed before the condition or either of
        # the branch is executed
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

    def get_pre_statement(self):
        raise NotImplementedError("pre_statement is no longer supported")
        return self.pre_statement

    def add_to_pre_statement(self, optree):
        raise NotImplementedError("pre_statement is no longer supported")
        #self.pre_statement.add(optree)

    def push_to_pre_statement(self, optree):
        raise NotImplementedError("pre_statement is no longer supported")
        #self.pre_statement.push(optree)

    def get_str(
            self, depth = 2, display_precision = False,
            tab_level = 0, memoization_map = None,
            display_attribute = False, display_id = False,
            custom_callback = lambda optree: "",
            display_interval=False
        ):
        """ string conversion for operation graph
            depth:                  number of level to be crossed (None: infty)
            display_precision:      enable/display format display
        """
        memoization_map = {} if memoization_map is None else memoization_map
        new_depth = None
        if depth != None:
            if  depth < 0:
                return ""
        new_depth = (depth - 1) if depth != None else None

        tab_str = AbstractOperation.str_del * tab_level + custom_callback(self)
        silent_str = "[S]" if self.get_silent() else ""
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        if 1:
            memoization_map[self] = str_tag
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            pre_str = tab_str + "%s%s%s%s%s ------> %s\n%s" % (self.get_name(), precision_str, silent_str, id_str, attribute_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id, custom_callback = custom_callback) for inp in self.inputs))
            for case in self.case_map:
                case_str = ""
                if isinstance(case, tuple):
                  case_str = ", ".join([str(sc) for sc in case])
                else:
                  case_str = "%s" % case
                pre_str += "Case: %s" %  case_str #.get_str(new_depth, display_precision, tab_level = 0, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id)
                pre_str += "%s" %  self.case_map[case].get_str(new_depth, display_precision, tab_level = tab_level + 2, memoization_map = memoization_map, display_attribute = display_attribute, display_id = display_id, custom_callback = custom_callback)
            return pre_str

class VectorAssembling(ML_ArithmeticOperation):
    name = "VectorAssembling"
    arity = None

class SubVectorExtract(ML_ArithmeticOperation):
    """ extraction of a sub-vector from a larger vector """
    name = "SubVectorExtract"
    arity = None

    def __init__(self, vector, *elt_indexes, **kwords):
        self.__class__.__base__.__init__(self, vector, *elt_indexes, **kwords)
        self.elt_index_list = list(elt_indexes)
        self.arity = len(elt_indexes) + 1

    def get_codegen_key(self):
        """ return code generation specific key """
        return None


    def finish_copy(self, new_copy, copy_map=None):
        # TODO/FIXME: extend to manage non-static elt_index_list
        # For example if list contains copiable nodes
        new_copy.elt_index_list = self.elt_index_list


class VectorElementSelection(ML_ArithmeticOperation):
    name = "VectorElementSelection"
    arity = 2
    implicit_arg_precision = {
        v2float32: ML_Binary32,
        v4float32: ML_Binary32,
        v8float32: ML_Binary32,

        v2float64: ML_Binary64,
        v4float64: ML_Binary64,
        v8float64: ML_Binary64,

        v2int32: ML_Int32,
        v4int32: ML_Int32,
        v8int32: ML_Int32,

        v2uint32: ML_UInt32,
        v4uint32: ML_UInt32,
        v8uint32: ML_UInt32,
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

    def finish_copy(self, new_copy, copy_map=None):
        # TODO/FIXME: extend to manage non-static elt_index
        # For example if elt_index is a copiable node
        new_copy.elt_index = self.elt_index


class VectorBroadcast(VectorAssembling):
    """ Specialization of VectorAssembling operator which broadcast a single
        element to all the vector lanes """
    arity = 1
    name = "VectorBroadcast"


## N-operand addition
def AdditionN(*args, **kwords):
  """ multiple-operand addition wrapper """
  op_list = [op for op in args]
  while len(op_list) > 1:
    op0 = op_list.pop(0)
    op1 = op_list.pop(0)
    op_list.append(Addition(op0, op1, **kwords))
  return op_list[0]


## Overloads @p optree get_likely
#  with custom defined value @p likely_value
def Likely(optree, likely_value):
  optree.get_likely = lambda: likely_value
  return optree

# end of doxygen group ml_operations
## @}



