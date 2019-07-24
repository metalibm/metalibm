# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

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

from sollya import inf, sup

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import (
    ML_LeafNode, Constant, Variable,
    Statement, ConditionBlock, ReferenceAssign,
    ConditionBlock, SwitchBlock, Abs, Select, Negation,
    Addition, Max, Min, FusedMultiplyAdd, Subtraction,
    Multiplication, Division, Modulo,
    NearestInteger, ExponentInsertion, ExponentExtraction,
    MantissaExtraction, RawSignExpExtraction, CountLeadingZeros,
    Comparison, Test, LogicalAnd, LogicalOr, LogicalNot,
    BitLogicAnd, BitLogicOr, BitLogicXor, BitLogicNegate,
    BitLogicLeftShift, BitLogicRightShift, BitArithmeticRightShift,
    Return, TableLoad, SpecificOperation, ExceptionOperation,
    NoResultOperation, Split, ComponentSelection, FunctionCall,
    Conversion, DivisionSeed, ReciprocalSquareRootSeed, ReciprocalSeed,
    VectorElementSelection,
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Loop, ComponentInstance, Assert, Wait, PlaceHolder
)
from metalibm_core.core.passes import (
    Pass, LOG_PASS_INFO, FunctionPass
)

abstract_typing_rule = {
    ConditionBlock:
        lambda optree, *ops: None,
    SwitchBlock:
        lambda optree, *ops: None,
    Abs:
        lambda optree, op0: merge_abstract_format(op0.get_precision()),
    Select:
        lambda optree, cond, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()),
    Negation:
        lambda optree, op0: merge_abstract_format(op0.get_precision()),
    Addition: 
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    Max:
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    Min:
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    FusedMultiplyAdd: 
        lambda optree, *ops: merge_abstract_format(*tuple(op.get_precision() for op in ops)),
    Subtraction: 
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    Multiplication: 
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    Division: 
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    Modulo: 
        lambda optree, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()), 
    NearestInteger:
        lambda optree, op0: ML_Integer,
    ExponentInsertion:
        lambda *ops: ML_Float,
    ExponentExtraction: 
        lambda *ops: ML_Integer,
    MantissaExtraction: 
        lambda *ops: ML_Float,
    RawSignExpExtraction: 
        lambda *ops: ML_Integer,
    CountLeadingZeros: 
        lambda *ops: ML_Integer,
    Comparison: 
        lambda *ops: ML_AbstractBool,
    Test: 
        lambda *ops: ML_AbstractBool,
    LogicalAnd: 
        lambda *ops: ML_AbstractBool,
    LogicalOr: 
        lambda *ops: ML_AbstractBool,
    LogicalNot: 
        lambda *ops: ML_AbstractBool,
    BitLogicAnd:
        lambda *ops: ML_Integer,
    BitLogicOr:
        lambda *ops: ML_Integer,
    BitLogicXor:
        lambda *ops: ML_Integer,
    BitLogicNegate:
        lambda *ops: ML_Integer,
    BitLogicLeftShift:
        lambda *ops: ML_Integer,
    BitLogicRightShift:
        lambda *ops: ML_Integer,
    BitArithmeticRightShift:
        lambda *ops: ML_Integer,
    Return:
        lambda *ops: ops[0].get_precision(),
    TableLoad: 
        lambda optree, *ops: ops[0].get_storage_precision(),
    SpecificOperation:
        lambda optree, *ops: optree.specifier.abstract_type_rule(optree, *ops),
    ExceptionOperation:
        lambda optree, *ops: optree.specifier.abstract_type_rule(optree, *ops),
    NoResultOperation:
        lambda optree, *ops: optree.specifier.abstract_type_rule(optree, *ops),
    Split:
        lambda optree, *ops: ML_Float,
    ComponentSelection:
        lambda optree, *ops: ML_Float,
    FunctionCall:
        lambda optree, *ops: optree.get_function_object().get_precision(),
    VectorElementSelection:
        lambda optree, *ops: ops[0].get_precision().get_scalar_format(),
}

practical_typing_rule = {
    Select: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs[1:]),
    Abs:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Negation:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Min:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Max:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Addition: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Subtraction: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Multiplication: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    FusedMultiplyAdd: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Division: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    DivisionSeed:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    ReciprocalSeed:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    ReciprocalSquareRootSeed:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    Modulo: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    ExponentInsertion:
        lambda backend, op, dprec: dprec,  
    ExponentExtraction:
        lambda backend, op, dprec: get_integer_format(backend, op),  
    MantissaExtraction: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    RawSignExpExtraction:
        lambda backend, op, dprec: get_integer_format(backend, op),
    CountLeadingZeros:
        lambda backend, op, dprec: get_integer_format(backend, op),
    Return:
        lambda backend, op, dprec: dprec,  
    NearestInteger: 
        lambda backend, op, dprec: get_integer_format(backend, op),
    Comparison: 
        lambda backend, op, dprec: get_boolean_format(backend, op),
    Test: 
        lambda backend, op, dprec: get_boolean_format(backend, op),
    LogicalAnd: 
        lambda backend, op, dprec: get_boolean_format(backend, op),
    LogicalNot: 
        lambda backend, op, dprec: get_boolean_format(backend, op),
    LogicalOr: 
        lambda backend, op, dprec: get_boolean_format(backend, op),
    BitLogicRightShift: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitArithmeticRightShift:
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitLogicLeftShift: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitLogicAnd: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitLogicOr: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitLogicXor: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    BitLogicNegate: 
        lambda backend, op, dprec: merge_ops_abstract_format(op, op.inputs),
    TableLoad:
        lambda backend, op, dprec: op.inputs[0].get_storage_precision(),
    Conversion: 
        lambda backend, op, dprec: op.get_precision(),
    ExceptionOperation:
        lambda backend, op, dprec: dprec if op.get_specifier() == ExceptionOperation.RaiseReturn else None,
    SpecificOperation:
        lambda backend, op, dprec: op.specifier.instantiated_type_rule(backend, op, dprec),
    NoResultOperation:
        lambda backend, op, dprec: None,
    Split: 
        lambda backend, op, dprec: op.get_precision(),
    ComponentSelection: 
        lambda backend, op, dprec: op.get_precision(),
    FunctionCall:
        lambda backend, op, dprec: op.get_precision(),
}

post_typing_process_rules = {
    Min:
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Max:
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Addition: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Subtraction: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Multiplication: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    FusedMultiplyAdd: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Division: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Modulo: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    Comparison: 
        lambda backend, op: propagate_format_to_cst(op, merge_format(op, [inp.get_precision() for inp in op.inputs])), 
    TableLoad: 
        lambda backend, op: propagate_format_to_cst(op, merge_format(op, [get_integer_format(backend, index_op) for index_op in op.inputs[1:]])),
    ExponentInsertion: 
        lambda backend, op: propagate_format_to_cst(op, backend.default_integer_format),
    Select:
        lambda backend, op: propagate_format_to_cst(op, op.get_precision(), index_list = [1, 2]),
    BitLogicAnd: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    BitLogicOr: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    BitLogicXor: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    BitLogicNegate: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    BitLogicRightShift: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    BitArithmeticRightShift:
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()),
    BitLogicLeftShift: 
        lambda backend, op: propagate_format_to_cst(op, op.get_precision()), 
    FunctionCall:
        lambda backend, op: FunctionCall.propagate_format_to_cst(op),
}


def instantiate_abstract_precision(optree, default_precision=None,
                                   memoization_map=None):
    """ recursively determine an abstract precision for each node """
    memoization_map = memoization_map or {}
    if optree in memoization_map:
        return memoization_map[optree]
    elif optree.get_precision() != None: 
        memoization_map[optree] = optree.get_precision()
        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            for inp in optree.get_extra_inputs():
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
        return optree.get_precision()
    else:
        if isinstance(optree, Constant):
            if isinstance(optree.get_value(), FP_SpecialValue):
                optree.set_precision(optree.get_value().get_precision())
                memoization_map[optree] = optree.get_precision()
                return optree.get_precision()
            else:
                if default_precision:
                  new_precision = default_precision
                else:
                  new_precision = ML_Integer if isinstance(optree.get_value(), int) else ML_Float
                optree.set_precision(new_precision)
                memoization_map[optree] = new_precision
                return new_precision

        elif isinstance(optree, Variable):
            if optree.get_var_type() in [Variable.Input, Variable.Local]:
                Log.report(Log.Error, "%s Variable %s has no defined precision" % (optree.get_var_type(), optree.get_tag()))
            else:
                Log.report(Log.Error, "Variable %s error: only Input Variables are supported in instantiate_abstract_precision" % optree.get_tag())

        elif isinstance(optree, TableLoad):
            # TableLoad operations
            for inp in optree.inputs[1:]:
                instantiate_abstract_precision(inp, ML_Integer, memoization_map = memoization_map)
            format_rule = abstract_typing_rule[optree.__class__]
            abstract_format = format_rule(optree, *optree.inputs)
            optree.set_precision(abstract_format)
            memoization_map[optree] = abstract_format
            return abstract_format

        elif isinstance(optree, ConditionBlock):
            # pre statement
            instantiate_abstract_precision(optree.get_pre_statement(), default_precision, memoization_map = memoization_map)
            # condition and branches
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            for inp in optree.get_extra_inputs():
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            memoization_map[optree] = None
            return None

        elif isinstance(optree, SwitchBlock):
            # pre statement
            instantiate_abstract_precision(optree.get_pre_statement(), default_precision, memoization_map = memoization_map)

            case_map = optree.get_case_map()
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            for inp in optree.get_extra_inputs():
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            memoization_map[optree] = None
            return None

        elif isinstance(optree, Statement):
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)

            memoization_map[optree] = None
            return None

        elif isinstance(optree, Loop):
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)

            memoization_map[optree] = None
            return None

        elif isinstance(optree, ReferenceAssign):
            var = optree.inputs[0]
            value = optree.inputs[1]
            var_type = instantiate_abstract_precision(var, default_precision, memoization_map = memoization_map)
            value_type = instantiate_abstract_precision(value, var_type, memoization_map = memoization_map)
            return None
                    
        else:
            # all other operations
            for inp in optree.inputs:
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            for inp in optree.get_extra_inputs():
                instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)

            try:
                format_rule = abstract_typing_rule[optree.__class__]
            except KeyError as e:
                Log.report(
                    Log.Error,
                    "not able to found {} in abstract_typing_rule",
                    optree.__class__, error=e)
            abstract_format = format_rule(optree, *optree.inputs)
            optree.set_precision(abstract_format)

            memoization_map[optree] = abstract_format
            return abstract_format

def merge_ops_abstract_integer_format(arg_format_list, default_precision=None):
    is_signed = any(op_format.get_signed() for op_format in arg_format_list if not is_abstract_format(op_format))
    # + [0] is required for python2 compatibility (python2 does not support
    # default keyword argument in max)
    arg_bit_size = max([arg_format.get_bit_size() for arg_format in arg_format_list if not is_abstract_format(arg_format)] + [0])

    INTEGER_MERGE_FORMAT = {
        True: {# signed
            32: ML_Int32,
            64: ML_Int64
        },
        False: {#unsigned
            32: ML_UInt32,
            64: ML_UInt64

        },
    }

    return INTEGER_MERGE_FORMAT[is_signed][arg_bit_size]


def merge_ops_abstract_float_format(arg_format_list, default_precision=None):
    # + [0] is required for python2 compatibility (python2 does not support
    # default keyword argument in max)
    arg_bit_size = max(arg_format.get_bit_size() for arg_format in arg_format_list if not is_abstract_format(arg_format))

    FORMAT_MERGE_TABLE = {
        32: ML_Binary32,
        64: ML_Binary64
    }

    return FORMAT_MERGE_TABLE[arg_bit_size]


def merge_ops_abstract_format(optree, args, default_precision = None):
    """ merging input format in multi-ary operation to determined result format.
        This function assumes that the result should be of the larger format
        which appears in the operand list. """
    try:
        if optree.get_precision() is ML_Float:
            return merge_ops_abstract_float_format(list(arg.get_precision() for arg in args))
        elif optree.get_precision() is ML_Integer:
            return merge_ops_abstract_integer_format(list(arg.get_precision() for arg in args))
        else:
            raise NotImplementedError
    except KeyError as e:
        Log.report(Log.Error, "unable to find record in merge_table for {}".format(
            optree.get_precision(), error=e
        ))


def get_integer_format(backend, optree):
    """ return integer format to use for optree """
    int_range = optree.get_interval()
    if int_range == None:
        return backend.default_integer_format
    elif inf(int_range) < 0:
        # signed
        if sup(int_range) > 2**31-1 or inf(int_range) < -2**31:
            return ML_Int64
        else:
            return ML_Int32
    else:
        # unsigned
        if sup(int_range) >= 2**32-1:
            return ML_UInt64
        else:
            return ML_UInt32

def get_boolean_format(backend, optree):
    """ return boolean format to use for optree """
    return backend.default_boolean_precision


def propagate_format_to_cst(optree, new_optree_format, index_list = []):
    """ propagate new_optree_format to Constant operand of <optree> with abstract precision """
    index_list = range(len(optree.inputs)) if index_list == [] else index_list
    for index in index_list:
        inp = optree.inputs[index]
        if isinstance(inp, Constant) and isinstance(inp.get_precision(), ML_AbstractFormat):
            inp.set_precision(new_optree_format)

def merge_format(optree, args, default_precision=None):
    """ merging input format in multi-ary operation to determined result format """
    max_binary_size = 0
    for arg in args:
        if isinstance(arg, ML_AbstractFormat): continue
        arg_bit_size = arg.get_bit_size()
        if arg_bit_size > max_binary_size:
            max_binary_size = arg_bit_size
    merge_table = {
        ML_Float: {
            32: ML_Binary32,
            64: ML_Binary64,
        },
        ML_Integer: {
            32: ML_Int32,
            64: ML_Int64,
        },
        ML_AbstractBool: {
          32: ML_Bool,

        }
    }
    
    try:
      merged_abstract_format = merge_abstract_format(*args)
      # if we have only unsigned integer, we merge to an unsigned integer also
      is_signed = merged_abstract_format is ML_Integer and not any(arg.get_signed() for arg in args if not arg.is_vector_format() and not is_abstract_format(arg))
      if is_signed:
        result_format = {
            32: ML_UInt32,
            64: ML_UInt64
        }[max_binary_size]
      else:
        result_format = merge_table[merged_abstract_format][max_binary_size]
    except KeyError:
      Log.report(Log.Info, "KeyError in merge_format")
      return None
    return result_format

def instantiate_precision(optree, default_precision=None, memoization_map=None, backend=None):
    """ instantiate final precisions and insert required conversions
        if the operation is not supported """
    memoization_map = memoization_map if not memoization_map is None else {}
    result_precision = optree.get_precision()

    if optree in memoization_map:
        return memoization_map[optree]

    if not isinstance(optree, ML_LeafNode):
        for inp in optree.inputs:
            instantiate_precision(
                inp, default_precision,
                memoization_map=memoization_map,
                backend=backend
            )

        # instanciating if abstract precision
        if isinstance(result_precision, ML_AbstractFormat):
            format_rule = practical_typing_rule[optree.__class__]
            result_precision = format_rule(backend, optree, default_precision)
            optree.set_precision(result_precision)

    for inp in optree.get_extra_inputs():
        instantiate_precision(inp, default_precision, memoization_map=memoization_map, backend=backend)

    memoization_map[optree] = optree.get_precision()

    if optree.__class__ in post_typing_process_rules:
        post_rule = post_typing_process_rules[optree.__class__]
        post_rule(backend, optree)

    return optree.get_precision()

class PassInstantiateAbstractPrecision(FunctionPass):
    """ Instantiate node formats: determining an abstract precision for each
        node whose format is undefined """
    pass_tag = "instantiate_abstract_prec"
    def __init__(self, target, default_precision=None):
        FunctionPass.__init__(self, "instantiate_abstract_prec")
        self.default_precision = default_precision

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        memoization_map = memoization_map or {}
        instantiate_abstract_precision(
            optree, self.default_precision, memoization_map
        )
        return optree

class PassInstantiatePrecision(FunctionPass):
    """ Instantiate node formats: determining a physical format for each node  """
    pass_tag = "instantiate_prec"
    def __init__(self, target, default_precision=None, default_integer_format=ML_Int32, default_boolean_precision=ML_Int32):
        FunctionPass.__init__(self, "instantiate_prec")
        self.default_integer_format = default_integer_format
        self.default_boolean_precision = default_boolean_precision
        self.default_precision = default_precision

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        memoization_map = memoization_map if not memoization_map is None else {}
        optree_precision = instantiate_precision(
            optree, self.default_precision, memoization_map=memoization_map, backend=self)
        return optree

# register pass
Log.report(LOG_PASS_INFO, "Registering instantiate_abstract_prec pass")
Pass.register(PassInstantiateAbstractPrecision)

Log.report(LOG_PASS_INFO, "Registering instantiate_prec pass")
Pass.register(PassInstantiatePrecision)

