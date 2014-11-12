# -*- coding: utf-8 -*-
###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2014)
# All rights reserved
# created:          Mar 20th, 2014
# last-modified:    Apr 14th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from .ml_operations import *
from .ml_formats import *


def merge_abstract_format(*args):
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


abstract_typing_rule = {
    ConditionBlock: 
        lambda optree, *ops: None,
    Abs:
        lambda optree, op0: merge_abstract_format(op0.get_precision()),
    Select:
        lambda optree, cond, op0, op1: merge_abstract_format(op0.get_precision(), op1.get_precision()),
    Negation:
        lambda optree, op0: merge_abstract_format(op0.get_precision()),
    Addition: 
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
    Comparison: 
        lambda *ops: ML_Bool,
    Test: 
        lambda *ops: ML_Bool,
    LogicalAnd: 
        lambda *ops: ML_Bool,
    LogicalOr: 
        lambda *ops: ML_Bool,
    LogicalNot: 
        lambda *ops: ML_Bool,
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
        lambda optree, *ops: None,
}

practical_typing_rule = {
    Select: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs[1:]),
    Abs:
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Negation:
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Addition: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Subtraction: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Multiplication: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    FusedMultiplyAdd: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Division: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Modulo: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    ExponentInsertion:
        lambda backend, op, dprec: dprec,  
    ExponentExtraction:
        lambda backend, op, dprec: backend.get_integer_format(op),  
    MantissaExtraction: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    Return:
        lambda backend, op, dprec: dprec,  
    NearestInteger: 
        lambda backend, op, dprec: backend.get_integer_format(op),
    Comparison: 
        lambda backend, op, dprec: backend.get_boolean_format(op),
    Test: 
        lambda backend, op, dprec: backend.get_boolean_format(op),
    LogicalAnd: 
        lambda backend, op, dprec: backend.get_boolean_format(op),
    LogicalNot: 
        lambda backend, op, dprec: backend.get_boolean_format(op),
    LogicalOr: 
        lambda backend, op, dprec: backend.get_boolean_format(op),
    BitLogicRightShift: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    BitLogicLeftShift: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    BitLogicAnd: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    BitLogicOr: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    BitLogicXor: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
    BitLogicNegate: 
        lambda backend, op, dprec: backend.merge_abstract_format(op, op.inputs),
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
    Addition: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    Subtraction: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    Multiplication: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    FusedMultiplyAdd: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    Division: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    Modulo: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    Comparison: 
        lambda backend, op: backend.propagate_format_to_cst(op, backend.merge_format(op, [inp.get_precision() for inp in op.inputs])), 
    TableLoad: 
        lambda backend, op: backend.propagate_format_to_cst(op, backend.merge_format(op, [backend.get_integer_format(index_op) for index_op in op.inputs[1:]])),
    ExponentInsertion: 
        lambda backend, op: backend.propagate_format_to_cst(op, backend.default_integer_format),
    Select:
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision(), index_list = [1, 2]),
    BitLogicAnd: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    BitLogicOr: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    BitLogicXor: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    BitLogicNegate: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    BitLogicRightShift: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    BitLogicLeftShift: 
        lambda backend, op: backend.propagate_format_to_cst(op, op.get_precision()), 
    FunctionCall:
        lambda backend, op: FunctionCall.propagate_format_to_cst(op),
}


type_escalation = {
    Addition: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
        },
    },
    Multiplication: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
        },
    },
    FusedMultiplyAdd: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
            lambda op_type: isinstance(op_type, ML_FP_Format):
                lambda op: op.get_precision(),
        },
    },
    ExponentInsertion: {
        lambda result_type: True: {
            lambda op_type: isinstance(op_type, ML_FP_Format): 
                lambda op: {32: ML_Int32, 64: ML_Int64}[op.get_precision().get_bit_size()],
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: ML_Int32,
        },
    },
}


# Table of transformation rule to translate an operation into its exact (no rounding error) counterpart
exactify_rule = {
    Addition: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    Multiplication: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    Subtraction: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    FusedMultiplyAdd: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
}

support_simplification = {
    FusedMultiplyAdd: {
        FusedMultiplyAdd.Standard: {
            lambda optree: True: 
                lambda optree: Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()),
        },
        FusedMultiplyAdd.Subtract: {
            lambda optree: True: 
                lambda optree: Subtraction(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()),
        },
        FusedMultiplyAdd.Negate: {
            lambda optree: True: 
                lambda optree: Negate(Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()), precision = optree.get_precision()),
        },
        FusedMultiplyAdd.SubtractNegate: {
            lambda optree: True: 
                lambda optree: Subtraction(optree.inputs[2], Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), precision = optree.get_precision()),
        },
        FusedMultiplyAdd.DotProduct: {
            lambda optree: True:
                lambda optree: Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), Multiplication(optree.inputs[2], optree.inputs[3], precision = optree.get_precision()), precision = optree.get_precision()), 
        },
        FusedMultiplyAdd.DotProductNegate: {
            lambda optree: True: 
                lambda optree: Subtraction(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), Multiplication(optree.inputs[2], optree.inputs[3], precision = optree.get_precision()), precision = optree.get_precision()), 
            
        },
    },
}





class OptimizationEngine:
    """ backend (precision instanciation and optimization passes) class """
    def __init__(self, processor, default_integer_format = ML_Int32, default_fp_precision = ML_Binary32, change_handle = True, dot_product_enabled = True):
        self.processor = processor
        self.default_integer_format = default_integer_format
        self.default_fp_precision = default_fp_precision
        self.change_handle = change_handle
        self.dot_product_enabled = dot_product_enabled

    def set_dot_product_enabled(self, dot_product_enabled):
        self.dot_product_enabled = dot_product_enabled

    def get_dot_product_enabled(self):
        return self.dot_product_enabled

    def copy_optree(self, optree, copy_map = {}):
        return optree.copy(copy_map)


    def get_default_fp_precision(self, optree):
        return self.default_fp_precision


    def get_integer_format(self, optree):
        """ return integer format to use for optree """
        int_range = optree.get_interval()
        if int_range == None:
            return self.default_integer_format
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


    def get_boolean_format(self, optree):
        """ return boolean format to use for optree """
        return ML_Int32


    def propagate_format_to_cst(self, optree, new_optree_format, index_list = []):
        """ propagate new_optree_format to Constant operand of <optree> with abstract precision """
        index_list = xrange(len(optree.inputs)) if index_list == [] else index_list
        for index in index_list:
            inp = optree.inputs[index]
            if isinstance(inp, Constant) and isinstance(inp.get_precision(), ML_AbstractFormat):
                inp.set_precision(new_optree_format)


    def merge_abstract_format(self, optree, args, default_precision = None):
        """ merging input format in multi-ary operation to determined result format """
        max_binary_size = 0
        for arg in args:
            if isinstance(arg.get_precision(), ML_AbstractFormat): continue
            try:
                arg_bit_size = arg.get_precision().get_bit_size()
            except:
                print optree.get_precision(), optree.get_str()
                print arg.get_precision(), arg.get_str()

                raise Exception()
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
        }
        
        result_format = merge_table[optree.get_precision()][max_binary_size]
        return result_format


    def merge_format(self, optree, args, default_precision = None):
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
        }
        
        result_format = merge_table[merge_abstract_format(*args)][max_binary_size]
        return result_format


    def instantiate_abstract_precision(self, optree, default_precision = None, memoization_map = {}):    
        """ recursively determine an abstract precision for each node """
        if optree in memoization_map:
            return memoization_map[optree]
        elif optree.get_precision() != None: 
            if not isinstance(optree, ML_LeafNode):
                for inp in optree.inputs:
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
                for inp in optree.get_extra_inputs():
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
            memoization_map[optree] = optree.get_precision()
            return optree.get_precision()
        else:
            if isinstance(optree, Constant):
                if isinstance(optree.get_value(), FP_SpecialValue):
                    optree.set_precision(optree.get_value().get_precision())
                    memoization_map[optree] = optree.get_precision()
                    return optree.get_precision()
                else:
                    new_precision = ML_Integer if isinstance(optree.get_value(), int) else ML_Float
                    optree.set_precision(new_precision)
                    memoization_map[optree] = new_precision
                    return new_precision

            elif isinstance(optree, Variable):
                if optree.get_var_type() is Variable.Input:
                    Log.report(Log.Error, "Input Variable %s has no defined precision" % optree.get_tag())
                else:
                    Log.report(Log.Error, "Variable %s error: only Input Variables are supported in instantiate_abstract_precision" % optree.get_tag())

            elif isinstance(optree, TableLoad):
                # TableLoad operations
                for inp in optree.inputs[1:]:
                    self.instantiate_abstract_precision(inp, ML_Integer, memoization_map = memoization_map)
                format_rule = abstract_typing_rule[optree.__class__]
                abstract_format = format_rule(optree, *optree.inputs)
                optree.set_precision(abstract_format)
                memoization_map[optree] = abstract_format
                return abstract_format

            elif isinstance(optree, ConditionBlock):
                # pre statement
                self.instantiate_abstract_precision(optree.get_pre_statement(), default_precision, memoization_map = memoization_map)
                # condition and branches
                for inp in optree.inputs:
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
                for inp in optree.get_extra_inputs():
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
                memoization_map[optree] = None
                return None

            elif isinstance(optree, Statement):
                for inp in optree.inputs:
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)

                memoization_map[optree] = None
                return None
                        
            else:
                # all other operations
                for inp in optree.inputs:
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)
                for inp in optree.get_extra_inputs():
                    self.instantiate_abstract_precision(inp, default_precision, memoization_map = memoization_map)

                format_rule = abstract_typing_rule[optree.__class__]
                abstract_format = format_rule(optree, *optree.inputs)
                optree.set_precision(abstract_format)

                memoization_map[optree] = abstract_format
                return abstract_format


    def simplify_fp_context(optree):
        """ factorize exception clearing and rounding mode changes accross
            connected DAG of floating-point operations """

        def is_fp_op(_optree):
            return isinstance(_optree.get_precision(), ML_FP_Format)

        if is_fp_op(optree):
            pass
        


    def instantiate_precision(self, optree, default_precision = None, memoization_map = {}):
        """ instantiate final precisions and insert required conversions
            if the operation is not supported """
        result_precision = optree.get_precision()

        if optree in memoization_map:
            return memoization_map[optree]

        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs: 
                self.instantiate_precision(inp, default_precision, memoization_map = memoization_map)

            # instanciating if abstract precision
            if isinstance(result_precision, ML_AbstractFormat): 
                format_rule = practical_typing_rule[optree.__class__]
                result_precision = format_rule(self, optree, default_precision)

                optree.set_precision(result_precision)

        for inp in optree.get_extra_inputs(): 
            self.instantiate_precision(inp, default_precision, memoization_map = memoization_map)

        if optree.__class__ in post_typing_process_rules:
            post_rule = post_typing_process_rules[optree.__class__]
            post_rule(self, optree)
        
        memoization_map[optree] = optree.get_precision()
        return optree.get_precision()


    def cb_parent_tagging(self, optree, parent_block = None):
        """ tries to factorize subexpression sharing between branch of ConditionBlock """
        if isinstance(optree, ConditionBlock):
            optree.parent = parent_block
            for op in optree.inputs: 
                self.cb_parent_tagging(op, parent_block = optree)
        elif not isinstance(optree, ML_LeafNode):
            for op in optree.inputs:
                self.cb_parent_tagging(op, parent_block = parent_block)


    def subexpression_sharing(self, optree, sharing_map = {}, level_sharing_map = [{}], current_parent_list = []):
        def search_level_map(optree):
            """ search if optree has been defined among the active node """
            for level in level_sharing_map:
                if optree in level: return True
            return False

        def common_ancestor(parent_list_0, parent_list_1):
            for b in parent_list_0[::-1]:
                if b in parent_list_1:
                    return b
            return None

        if isinstance(optree, ConditionBlock):
            optree.set_parent_list(current_parent_list)
            # condition
            self.subexpression_sharing(optree.inputs[0], sharing_map, level_sharing_map, current_parent_list + [optree])
            # branches
            for op in optree.inputs[1:]:
                self.subexpression_sharing(op, sharing_map, [{}] + level_sharing_map, current_parent_list + [optree])

        elif isinstance(optree, Statement):
            for op in optree.inputs:
                self.subexpression_sharing(op, sharing_map, [{}] + level_sharing_map, current_parent_list)

        elif isinstance(optree, ML_LeafNode):
            pass
        else:
            if optree in sharing_map:
                if not search_level_map(optree): 
                    # parallel branch sharing possibility
                    ancestor = common_ancestor(sharing_map[optree], current_parent_list)            
                    if ancestor != None:
                        ancestor.add_to_pre_statement(optree)
            else:
                sharing_map[optree] = current_parent_list
                level_sharing_map[0][optree] = current_parent_list
                for op in optree.inputs:
                    self.subexpression_sharing(op, sharing_map, level_sharing_map, current_parent_list)


    def extract_fast_path(self, optree):
        """ extracting fast path (most likely execution) from <optree> """
        if isinstance(optree, ConditionBlock):
            cond = optree.inputs[0]
            likely = cond.get_likely()
            pre_statement_fast_path = self.extract_fast_path(optree.get_pre_statement())
            if pre_statement_fast_path != None:
                return pre_statement_fast_path
            else:
                if likely:
                    return self.extract_fast_path(optree.inputs[1])
                elif likely == False and len(optree.inputs) >= 3:
                    return self.extract_fast_path(optree.inputs[2])
                else:
                    return None
        elif isinstance(optree, Statement):
            for sub_stat in optree.inputs:
                ss_fast_path = self.extract_fast_path(sub_stat)
                if ss_fast_path != None: return ss_fast_path
            return None
        elif isinstance(optree, Return):
            return optree.inputs[0]
        else:
            return None

    def factorize_fast_path(self, optree):
        """ extract <optree>'s fast path and add it to be pre-computed at 
            the start of <optree> computation """
        fast_path = self.extract_fast_path(optree)
        if fast_path == None: 
            return
        elif isinstance(optree, ConditionBlock):
            optree.push_to_pre_statement(fast_path)
        elif isinstance(optree, Statement):
            optree.push(fast_path)
        else:
            Log.report(Log.Error, "unsupported root for fast path factorization")


    def fuse_multiply_add(self, optree, silence = False, memoization = {}):
        """ whenever possible fuse a multiply and add/sub into a FMA/FMS """
        if (isinstance(optree, Addition) or isinstance(optree, Subtraction)) and not optree.get_unbreakable():
            if len(optree.inputs) != 2:
                # more than 2-operand addition are not supported yet
                optree.inputs = tuple(self.fuse_multiply_add(op, silence = silence, memoization = memoization) for op in optree.inputs)
                return optree

            else:
                if optree in memoization: 
                    return memoization[optree]

                elif optree.get_unbreakable():
                    optree.inputs = tuple(self.fuse_multiply_add(op, silence = silence, memoization = memoization) for op in optree.inputs)
                    memoization[optree] = optree
                    return optree

                elif self.get_dot_product_enabled() and isinstance(optree.inputs[0], Multiplication) and isinstance(optree.inputs[1], Multiplication) and not optree.inputs[0].get_prevent_optimization() and not optree.inputs[1].get_prevent_optimization():
                    specifier = FusedMultiplyAdd.DotProductNegate if isinstance(optree, Subtraction) else FusedMultiplyAdd.DotProduct 
                    mult0 = self.fuse_multiply_add(optree.inputs[0].inputs[0], silence = silence, memoization = memoization)
                    mult1 = self.fuse_multiply_add(optree.inputs[0].inputs[1], silence = silence, memoization = memoization)
                    mult2 = self.fuse_multiply_add(optree.inputs[1].inputs[0], silence = silence, memoization = memoization)
                    mult3 = self.fuse_multiply_add(optree.inputs[1].inputs[1], silence = silence, memoization = memoization)
                    new_op = FusedMultiplyAdd(mult0, mult1, mult2, mult3, specifier = specifier)
                    new_op.attributes = optree.attributes.get_light_copy()
                    new_op.set_silent(silence)
                    new_op.set_index(optree.get_index())
                    # propagating exact attribute
                    if optree.inputs[0].get_exact() and optree.inputs[1].get_exact() and optree.get_exact():
                        new_op.set_exact(True)
                    # modifying handle
                    if self.change_handle: optree.get_handle().set_node(new_op)
                    memoization[optree] = new_op
                    return new_op

                elif isinstance(optree.inputs[0], Multiplication) and not optree.inputs[0].get_prevent_optimization():
                    specifier = FusedMultiplyAdd.Subtract if isinstance(optree, Subtraction) else FusedMultiplyAdd.Standard 
                    mult0 = self.fuse_multiply_add(optree.inputs[0].inputs[0], silence = silence, memoization = memoization)
                    mult1 = self.fuse_multiply_add(optree.inputs[0].inputs[1], silence = silence, memoization = memoization)
                    addend = self.fuse_multiply_add(optree.inputs[1], silence = silence, memoization = memoization)

                    #print "fusing fma label", optree.get_interval()
                    new_op = FusedMultiplyAdd(mult0, mult1, addend, specifier = specifier)
                    new_op.attributes = optree.attributes.get_light_copy()
                    new_op.set_silent(silence)
                    new_op.set_index(optree.get_index())
                    #print "fusing fma label", mult0.get_interval(), mult1.get_interval(), addend.get_interval()

                    # propagating exact attribute
                    if optree.inputs[0].get_exact() and optree.get_exact():
                        new_op.set_exact(True)

                    # modifying handle
                    if self.change_handle: optree.get_handle().set_node(new_op)

                    memoization[optree] = new_op
                    return new_op

                elif isinstance(optree.inputs[1], Multiplication) and not optree.inputs[1].get_prevent_optimization():
                    specifier = FusedMultiplyAdd.SubtractNegate if isinstance(optree, Subtraction) else FusedMultiplyAdd.Standard 
                    mult0 = self.fuse_multiply_add(optree.inputs[1].inputs[0], silence = silence, memoization = memoization)
                    mult1 = self.fuse_multiply_add(optree.inputs[1].inputs[1], silence = silence, memoization = memoization)
                    addend = self.fuse_multiply_add(optree.inputs[0], silence = silence)
                    new_op = FusedMultiplyAdd(mult0, mult1, addend, specifier = specifier)
                    new_op.attributes = optree.attributes.get_light_copy()
                    new_op.set_silent(silence)
                    new_op.set_commutated(True)
                    memoization[optree] = new_op

                    new_op.set_index(optree.get_index())
                    # propagating exact attribute
                    if optree.inputs[1].get_exact() and optree.get_exact():
                        new_op.set_exact(True)

                    # modifying handle
                    if self.change_handle: optree.get_handle().set_node(new_op)

                    return new_op
                else:
                    optree.inputs = tuple(self.fuse_multiply_add(op, silence = silence, memoization = memoization) for op in optree.inputs)
                    memoization[optree] = optree
                    return optree
        else:
            if optree.get_extra_inputs() != []: 
                optree.set_extra_inputs([self.fuse_multiply_add(op, silence = silence, memoization = memoization) for op in optree.get_extra_inputs()])

            if isinstance(optree, ML_LeafNode):
                memoization[optree] = optree
                return optree
            else:
                optree.inputs = tuple(self.fuse_multiply_add(op, silence = silence, memoization = memoization) for op in optree.inputs)
                memoization[optree] = optree
                return optree

    def silence_fp_operations(self, optree, force = False):
        if isinstance(optree, ML_LeafNode):
            pass
        else:
            for op in optree.inputs:
                self.silence_fp_operations(op, force = force)
            for op in optree.get_extra_inputs():
                self.silence_fp_operations(op, force = force)
            if isinstance(optree, Multiplication) or isinstance(optree, Addition) or isinstance(optree, FusedMultiplyAdd) or isinstance(optree, Subtraction):
                if optree.get_silent() == None: optree.set_silent(True)


    def register_nodes_by_tag(self, optree, node_map = {}):
        """ build a map tag->optree """
        # registering node if tag is defined
        if optree.get_tag() != None:
            node_map[optree.get_tag()] = optree

        # processing extra_inputs list
        for op in optree.get_extra_inputs():
            self.register_nodes_by_tag(op, node_map)

        # processing inputs list for non ML_LeafNode optree
        if not isinstance(optree, ML_LeafNode):
            for op in optree.inputs:
                self.register_nodes_by_tag(op, node_map)

    def has_support_simplification(self, optree):
        if optree.__class__ in support_simplification:
            code_gen_key = optree.get_codegen_key()
            if code_gen_key in support_simplification[optree.__class__]:
                for cond in support_simplification[optree.__class__][code_gen_key]:
                    if cond(optree): return True
        return False

    def get_support_simplification(self, optree):
        code_gen_key = optree.get_codegen_key()
        for cond in support_simplification[optree.__class__][code_gen_key]:
            if cond(optree):
                return support_simplification[optree.__class__][code_gen_key][cond](optree)
        Log.report(Log.Error, "support simplification mapping not found")


    def check_processor_support(self, optree, memoization_map = {}):
        """ check if all precision-instantiated operation are supported by the processor """
        if optree in memoization_map:
            return True
        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs:
                self.check_processor_support(inp, memoization_map)

            if isinstance(optree, ConditionBlock):
                self.check_processor_support(optree.get_pre_statement(), memoization_map)
                pass
            elif isinstance(optree, Statement):
                pass
            elif isinstance(optree, Return):
                pass
            elif not self.processor.is_supported_operation(optree):
                # trying operand format escalation
                while optree.__class__ in type_escalation:
                    match_found = False
                    for result_type_cond in type_escalation[optree.__class__]:
                        if result_type_cond(optree.get_precision()): 
                            for op_index in xrange(len(optree.inputs)):
                                op = optree.inputs[op_index]
                                for op_type_cond in type_escalation[optree.__class__][result_type_cond]:
                                    if op_type_cond(op.get_precision()): 
                                        #Log.report(Log.Info, "type escalation for %s %s" % (op.get_str(depth = 1, display_precision = True), optree.get_str(depth = 1, display_precision = True)))
                                        new_type = type_escalation[optree.__class__][result_type_cond][op_type_cond](optree) 
                                        if op.get_precision() != new_type:
                                            # conversion insertion
                                            input_list = list(optree.inputs)
                                            input_list[op_index] = Conversion(op, precision = new_type)
                                            optree.inputs = tuple(input_list)
                                            match_found = True
                                            break
                            break
                    if not match_found:
                        break
                # checking final processor support
                if not self.processor.is_supported_operation(optree):
                    # look for possible simplification
                    if self.has_support_simplification(optree):
                        simplified_tree = self.get_support_simplification(optree)
                        #Log.report(Log.Info, "simplifying %s" % optree.get_str(depth = 2, display_precision = True))
                        #Log.report(Log.Info, "into %s" % simplified_tree.get_str(depth = 2, display_precision = True))
                        optree.change_to(simplified_tree)
                        if self.processor.is_supported_operation(optree):
                            memoization_map[optree] = True
                            return True
                        
                    print optree
                    print self.processor.get_operation_keys(optree)
                    print optree.get_str(display_precision = True, memoization_map = {})
                    Log.report(Log.Error, "unsupported operation\n")
        # memoization
        memoization_map[optree] = True
        return True

    def swap_format(self, optree, new_format):
        optree.set_precision(new_format)
        return optree


    def exactify(self, optree, exact_format = ML_Exact, memoization_map = {}):
        """ recursively process <optree> according to table exactify_rule 
            to translete each node into is exact counterpart (no rounding error)
            , generally by setting its precision to <exact_format> """
        if optree in memoization_map:
            return memoization_map[optree]
        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs:
                self.exactify(inp, exact_format, memoization_map)
            for inp in optree.get_extra_inputs():
                self.exactify(inp, exact_format, memoization_map)

        if optree.__class__ in exactify_rule:
            for cond in exactify_rule[optree.__class__][None]:
                if cond(optree, exact_format):
                    new_optree = exactify_rule[optree.__class__][None][cond](self, optree, exact_format)
                    memoization_map[optree] = new_optree
                    return new_optree   

        memoization_map[optree] = optree
        return optree
        


    def optimization_process(self, pre_scheme, default_precision, copy = False, fuse_fma = True, subexpression_sharing = True, silence_fp_operations = True, factorize_fast_path = True):
        # copying when required
        scheme = pre_scheme if not copy else pre_scheme.copy({})

        # fusing FMA
        if fuse_fma:
            Log.report(Log.Info, "fusing FMA")
        scheme_post_fma = scheme if not fuse_fma else self.fuse_multiply_add(scheme, silence = silence_fp_operations)
        
        # instantiating abstract precisions
        Log.report(Log.Info, "instantiating abstract precisions")
        self.instantiate_abstract_precision(scheme_post_fma, None)
        Log.report(Log.Info, "instantiating precisions")
        self.instantiate_precision(scheme_post_fma, default_precision)

        if subexpression_sharing:
            Log.report(Log.Info, "sharing sub-expressions")
            self.subexpression_sharing(scheme_post_fma)
        if silence_fp_operations:
            Log.report(Log.Info, "silencing fp operations")
            self.silence_fp_operations(scheme_post_fma)

        # check processor support
        Log.report(Log.Info, "checking processor support")
        self.check_processor_support(scheme_post_fma)

        if factorize_fast_path:
            Log.report(Log.Info, "factorizing fast path")
            self.factorize_fast_path(scheme_post_fma)

        return scheme_post_fma
        


            


        
