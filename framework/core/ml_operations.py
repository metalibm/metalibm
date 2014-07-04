# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 23rd, 2013
# last-modified:    Mar 20th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import Interval, SollyaObject, PSI_is_range, nearestint

from utility.log_report import Log
from utility.common import Callable
from core.attributes import Attributes, attr_init
from core.ml_formats import * # FP_SpecialValue, ML_FloatingPointException, ML_FloatingPoint_RoundingMode, ML_FPRM_Type, ML_FPE_Type


def std_merge_abstract_format(*args):
    """ return the most generic abstract format
        to unify args formats """
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


class ML_Operation(object):
    """ parent to Metalibm's operation """
    pass


def implicit_op(op):
    """ implicit operation conversion (from number to Constant when required) """
    if isinstance(op, ML_Operation):
        return op
    elif isinstance(op, SollyaObject) or isinstance(op, int) or isinstance(op, float) or isinstance(op, FP_SpecialValue):
        return Constant(op)
    elif isinstance(op, ML_FloatingPointException):
        return Constant(op, precision = ML_FPE_Type)
    elif isinstance(op, ML_FloatingPoint_RoundingMode):
        return Constant(op, precision = ML_FPRM_Type)
    else:
        print "ERROR: unsupport operand in implicit_op conversion ", op
        raise Exception()


class AbstractOperation(ML_Operation):
    """ parent to Metalibm's abstrat operation """
    name = "AbstractOperation"
    extra_inputs = []
    global_index = 0
    str_del = "| "

    def __init__(self, **init_map):
        # init operation handle
        self.attributes = Attributes(**init_map)
        self.index = AbstractOperation.global_index; AbstractOperation.global_index += 1
        self.get_handle().set_node(self)

    @property
    def hi(self):
        return ComponentSelection(self, specifier = ComponentSelection.Hi)

    @property
    def lo(self):
        return ComponentSelection(self, specifier = ComponentSelection.Lo)


    def __not__(self):
        return LogicalNot(self)

    def __and__(self, op):
        return LogicalAnd(self, op)

    def __or__(self, op):
        return LogicalOr(self, op)

    def __neg__(self):
        return Negation(self)

    def __add__(self, op):
        """ implicit add operation between AbstractOperation """
        return Addition(self, implicit_op(op))

    def __sub__(self, op):
        """ implicit add operation between AbstractOperation """
        return Subtraction(self, implicit_op(op))

    def __radd__(self, op):
        """ implicit reflexive add operation between AbstractOperation """
        return Addition(implicit_op(op), self)

    def __mul__(self, op):
        """ implicit multiply operation between AbstractOperation """
        return Multiplication(self, implicit_op(op))

    def __rsub__(self, op):
        return Subtraction(implicit_op(op), self)

    def __rmul__(self, op):
        """ implicit reflexive multiply operation between AbstractOperation """
        return Multiplication(implicit_op(op), self)

    def __div__(self, op):
        """ implicit division operation between AbstractOperation """
        return Division(self, implicit_op(op))
        
    def __rdiv__(self, op):
        """ implicit reflexive division operation between AbstractOperation """
        return Division(implicit_op(op), self)
        
    def __mod__(self, op):
        """ implicit modulo operation between AbstractOperation """
        return Modulo(self, implicit_op(op))
        
    def __rmod__(self, op):
        """ implicit reflexive modulo operation between AbstractOperation """
        return Modulo(self, implicit_op(op))

    def __lt__(self, op):
        """ implicit less than operation """
        return Comparison(self, implicit_op(op), specifier = Comparison.Less)

    def __le__(self, op):
        """ implicit less or egual operation """
        return Comparison(self, implicit_op(op), specifier = Comparison.LessOrEqual)

    def __ge__(self, op):
        """ implicit greater or equal operation """
        return Comparison(self, implicit_op(op), specifier = Comparison.GreaterOrEqual)

    def __gt__(self, op):
        """ implciit greater than operation """
        return Comparison(self, implicit_op(op), specifier = Comparison.Greater)

    def get_precision(self):
        """ precision getter (transmit to self.attributes field) """
        return self.attributes.get_precision()
    def set_precision(self, new_precision):
        """ precision setter (transmit to self.attributes field) """
        self.attributes.set_precision(new_precision)

    def get_interval(self):
        """ interval getter (transmit to self.attributes field) """
        return self.attributes.get_interval()
    def set_interval(self, new_interval):
        """ interval setter (transmit to self.attributes field) """
        return self.attributes.set_interval(new_interval)

    def get_tag(self, default = None):
        """ tag getter (transmit to self.attributes field) """
        op_tag = self.attributes.get_tag()
        return default if op_tag == None else op_tag

    def set_tag(self, new_tag):
        """ tag setter (transmit to self.attributes field) """
        return self.attributes.set_tag(new_tag)

    def get_debug(self):
        """ debug getter (transmit to self.attributes field) """
        return self.attributes.get_debug()
    def set_debug(self, new_debug):
        """ debug setter (transmit to self.attributes field) """
        return self.attributes.set_debug(new_debug)

    def get_silent(self):
        return self.attributes.get_silent()
    def set_silent(self, silent_value):
        return self.attributes.set_silent(silent_value)

    def get_handle(self):
        return self.attributes.get_handle()
    def set_handle(self, new_handle):
        self.attributes.set_handle(new_handle)

    def get_clearprevious(self):
        return self.attributes.get_clearprevious()
    def set_clearprevious(self, new_clearprevious):
        return self.attributes.set_clearprevious(new_clearprevious)

    def set_attributes(self, **kwords):
        self.attributes.set_attr(**kwords)

    def get_index(self):
        """ index getter function """
        return self.index
    def set_index(self, new_index):
        """ index setter function """
        self.index = new_index 

    def get_rounding_mode(self):
        """ rounding mode getter function (attributes)"""
        return self.attributes.get_rounding_mode()
    def set_rounding_mode(self, new_rounding_mode):
        """ rounding mode setter function (attributes) """
        self.attributes.set_rounding_mode(new_rounding_mode)


    def get_max_abs_error(self):
        return self.attributes.get_max_abs_error()
    def set_max_abs_error(self, new_max_abs_error):
        self.attributes.set_max_abs_error(new_max_abs_error)


    def get_name(self):
        """ return operation name (by default class name) """
        return self.name

    def get_extra_inputs(self):
        """ return list of non-standard inputs """
        return self.extra_inputs

    def change_to(self, optree):
        """ change <self> operation to match optree """
        self.__class__ = optree.__class__
        self.inputs = optree.inputs
        self.arity = optree.arity
        self.attributes = optree.attributes
        if isinstance(optree, SpecifierOperation):
            self.specifier = optree.specifier


    def get_str(self, depth = 2, display_precision = False, tab_level = 0, memoization_map = {}):
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
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        if self.arity == 1:
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            memoization_map[self] = str_tag
            return tab_str + "%s%s%s -------> %s\n%s" % (self.get_name(), precision_str, silent_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map) for inp in self.inputs))
        else:
            memoization_map[self] = str_tag
            precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
            return tab_str + "%s%s%s ------> %s\n%s" % (self.get_name(), precision_str, silent_str, str_tag, "".join(inp.get_str(new_depth, display_precision, tab_level = tab_level + 1, memoization_map = memoization_map) for inp in self.inputs))


    def finish_copy(self, new_copy, copy_map = {}):
        pass


class ML_ArithmeticOperation(AbstractOperation):
    """ base class for all arithmetic operation that may depend
        on floating-point context (rounding mode for example) """
    pass

class ML_LeafNode(AbstractOperation): 
    """ AbstractOperation with no expected input """
    pass

class Constant(ML_LeafNode):
    """ Constant operation class """
    def __init__(self, value, **init_map):
        # value initialization
        AbstractOperation.__init__(self, **init_map)
        self.value = value
        # attribute fields initialization
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, SollyaObject):
            self.attributes.set_interval(Interval(value))


    def get_value(self):
        return self.value

    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        return AbstractOperation.str_del * tab_level + "Cst(%s)%s\n" % (self.value, precision_str)


    def copy(self, copy_map = {}):
        """ return a new, free copy of <self> """
        # test for previous definition in memoization map
        if self in copy_map: return copy_map[self]
        # else define a new and free copy
        new_copy = Constant(self.value)
        new_copy.attributes = self.attributes.get_copy()
        copy_map[self] = new_copy
        return new_copy


class Variable(ML_LeafNode):
    class Input: pass
    class Intermediay: pass
    """ Variable operator class """
    def __init__(self, tag, **init_map):
        AbstractOperation.__init__(self, **init_map)
        self.attributes.set_tag(tag)
        # used to distinguish between input variables (without self.inputs) 
        # and intermediary variables 
        self.var_type = attr_init(init_map, "var_type", default_value = Variable.Input)  

    def get_var_type(self):
        return self.var_type

    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_precision())
        return AbstractOperation.str_del * tab_level + "Var(%s)%s\n" % (self.get_tag(), precision_str)


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
    # test for previous definition in memoization map
    if self in copy_map: return copy_map[self]
    # else define a new and free copy
    new_copy = self.__class__(*tuple(op.copy(copy_map) for op in self.inputs), __copy = True)
    new_copy.attributes = self.attributes.get_copy()
    self.finish_copy(new_copy, copy_map)
    copy_map[self] = new_copy
    return new_copy


class InvalidInterval(Exception): pass

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


class BitLogicAnd(AbstractOperationConstructor("BitLogicAnd")):
    pass
class BitLogicOr(AbstractOperationConstructor("BitLogicOr")):
    pass
class BitLogicXor(AbstractOperationConstructor("BitLogicXor")):
    pass
class BitLogicNegate(AbstractOperationConstructor("BitLogicNegate", arity = 1)):
    pass
class BitLogicRightShift(AbstractOperationConstructor("BitLogicRightShift", arity = 2)):
    pass
class BitLogicLeftShift(AbstractOperationConstructor("BitLogicLeftShift", arity = 2)):
    pass


class Abs(AbstractOperationConstructor("Abs", range_function = lambda self, ops: abs(ops[0]))):
    """ abstract absolute value operation """
    pass

class Negation(AbstractOperationConstructor("Negation", range_function = lambda self, ops: - ops[0])): 
    """ abstract negation """
    pass

class Addition(ArithmeticOperationConstructor("Addition", range_function = lambda self, ops: ops[0] + ops[1])): 
    """ abstract addition """
    pass

class Negate(AbstractOperationConstructor("Negate", range_function = lambda self, ops: -ops[0])): 
    """ abstract negation """
    pass


class SpecifierOperation: pass

class ComponentSelectionSpecifier: pass

class Split(AbstractOperationConstructor("Split", arity = 1)):
    pass
class ComponentSelection(AbstractOperationConstructor("ComponentSelection", inheritance = [SpecifierOperation], arity = 1)):
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


class Modulo(AbstractOperationConstructor("Modulo", range_function = lambda self, ops: ops[0] % ops[1])):
    """ abstract modulo operation """
    pass


class NearestInteger(ArithmeticOperationConstructor("NearestInteger", arity = 1, range_function = lambda self, ops: nearestint(ops[0]))): 
    """ abstract addition """
    pass


class PowerOf2(AbstractOperationConstructor("PowerOf2", arity = 1, range_function = lambda self, ops: S2**ops[0])):
    """ abstract power of 2 operation """
    pass

class Return(AbstractOperationConstructor("Return", arity = 1, range_function = lambda self, ops: ops[0])):
    """ abstract return value operation """
    pass

class TableLoad(AbstractOperationConstructor("TableLoad", arity = 2, range_function = lambda self, ops: None)):
    """ abstract load from a table operation """
    pass


def interval_union(int0, int1):
    return Interval(min(inf(int0), inf(int1)), max(sup(int0), sup(int1)))

class Select(AbstractOperationConstructor("Select", arity = 3, range_function = lambda self, ops: interval_union(ops[1], ops[2]))):
    pass


def Max(op0, op1, **kwords):
    return Select(Comparison(op0, op1, specifier = Comparison.Greater), op0, op1, **kwords)

def Min(op0, op1, **kwords):
    return Select(Comparison(op0, op1, specifier = Comparison.Less), op0, op1, **kwords)


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
        new_copy.pre_statement = self.statement.copy(copy_map)
        new_copy.extra_inputs = [op.copy(copy_map) for op in self.extra_inputs]
        new_copy.parent_list = [op.copy(copy_map) for op in self.parent_list] 

class Conversion(AbstractOperationConstructor("Conversion", arity = 1)):
    """ abstract conversion operation """
    pass

class TypeCast(AbstractOperationConstructor("TypeCast", arity = 1)):
    """ abstract conversion operation """
    pass

class ExponentInsertion(AbstractOperationConstructor("ExponentInsertion", arity = 1)):
    pass

class MantissaExtraction(AbstractOperationConstructor("MantissaExtraction", arity = 1)):
    pass

class ExponentExtraction(AbstractOperationConstructor("ExponentExtraction", arity = 1)):
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
    return type(op_name, (AbstractOperationConstructor(op_name, inheritance = [BooleanOperation]),), field_map)



LogicalAnd = LogicOperationBuilder("LogicalAnd", arity = 2, likely_function = lambda self, *ops: ops[0] and ops[1])
LogicalOr  = LogicOperationBuilder("LogicalOr",  arity = 2, likely_function = lambda self, *ops: ops[0] or ops[1])
LogicalNot = LogicOperationBuilder("LogicalNot", arity = 1, likely_function = lambda self, *ops: not ops[0] )

        

class Test(AbstractOperationConstructor("Test", inheritance = [BooleanOperation, SpecifierOperation])):
    """ Abstract Test operation class """
    class IsNaN(TestSpecifier_Builder("IsNaN", 1)): pass
    class IsQuietNaN(TestSpecifier_Builder("IsQuietNaN", 1)): pass
    class IsSignalingNaN(TestSpecifier_Builder("IsSignalingNaN", 1)): pass
    class IsInfty(TestSpecifier_Builder("IsInfty", 1)): pass
    class IsPositiveInfty(TestSpecifier_Builder("IsPositiveInfty", 1)): pass
    class IsNegativeInfty(TestSpecifier_Builder("IsNegativeInfty", 1)): pass
    class IsInfOrNaN(TestSpecifier_Builder("IsInfOrNaN", 1)): pass
    class IsZero(TestSpecifier_Builder("IsZero", 1)): pass
    class IsPositiveZero(TestSpecifier_Builder("IsPositiveZero", 1)): pass
    class IsNegativeZero(TestSpecifier_Builder("IsNegativeZero", 1)): pass
    class IsSubnormal(TestSpecifier_Builder("IsSubnormal", 1)): pass
    class CompSign(TestSpecifier_Builder("CompSign", 2)): pass
    class SpecialCases(TestSpecifier_Builder("SpecialCases", 1)): pass
    class IsInvalidInput(TestSpecifier_Builder("IsInvalidInput", 1)): pass

    def __init__(self, *args, **kwords):
        self.__class__.__base__.__init__(self, *args, **kwords)
        BooleanOperation.__init__(self, attr_init(kwords, "likely"))
        self.specifier = attr_init(kwords, "specifier", required = True)
        self.arity = self.specifier.arity


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


class Comparison(AbstractOperationConstructor("Comparison", arity = 2, inheritance = [BooleanOperation, SpecifierOperation])):
    """ Abstract Comparison operation """
    class Equal: pass
    class Less: pass
    class LessOrEqual: pass
    class Greater: pass
    class GreaterOrEqual: pass
    class NotEqual: pass


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


    def add(self, optree):
        """ add a new unary statement at the end of the input list """
        self.inputs = self.inputs + (optree,)
        self.arity += 1


    def push(self, optree):
        """ add a new unary statement at the beginning of the input list """
        self.inputs = (optree,) + self.inputs
        self.arity += 1


    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.arity = self.arity


class FieldExtraction(AbstractOperationConstructor("FieldExtraction")):
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


class FunctionCall(AbstractOperationConstructor("FunctionCall")):
    def __init__(self, function_object, *args, **kwords):
        FunctionCall.__base__.__init__(self, *args, **kwords)
        self.arity = len(args)
        self.function_object = function_object

    def get_precision(self):
        return self.function_object.get_precision()

    def get_function_object(self):
        return self.function_object

    def propagate_format_to_cst(optree):
        """ propagate new_optree_format to Constant operand of <optree> with abstract precision """
        index_list = xrange(len(optree.inputs)) 
        for index in index_list:
            inp = optree.inputs[index]
            new_optree_format = optree.get_function_object().get_arg_precision(index)
            if isinstance(inp, Constant) and isinstance(inp.get_precision(), ML_AbstractFormat):
                inp.set_precision(new_optree_format)



    # static binding
    propagate_format_to_cst = Callable(propagate_format_to_cst)
