# -*- coding: utf-8 -*-

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
# This file is part of Metalibm tool
# created:          Aug 8th, 2017
# last-modified:    Mar 8th, 2018
#
# author(s):    Nicolas Brunie (nbrunie@kalray.eu)
# description:
###############################################################################

import operator

import sollya

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_operations import (
    Comparison, Select, Constant, TypeCast, Multiplication, Addition,
    Subtraction, Negation, Test,
    Max, Min,
    Conversion,
    BitLogicRightShift, BitLogicLeftShift, BitLogicAnd,
    LogicalAnd, LogicalOr,
    ReciprocalSeed, CopySign,
    ReciprocalSquareRootSeed,
    TableLoad, Division, Modulo, Equal,
    ExponentExtraction, ExponentInsertion,
    FunctionCall,
)
from metalibm_core.core.ml_hdl_operations import (
    SubSignalSelection
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Int64,
    ML_Bool, ML_Integer, v2bool, v3bool, v4bool, v8bool,
    v2int32, v4int32, v8int32,
    v2int64, v4int64, v8int64,
    v2float32, v4float32, v8float32,
    v2float64, v4float64, v8float64,
    ML_Binary32,
    is_std_integer_format, ML_FP_Format,
    VECTOR_TYPE_MAP
)
from metalibm_core.core.ml_hdl_format import (
    is_fixed_point, ML_StdLogicVectorFormat
)

from metalibm_core.core.generic_approximation import invsqrt_approx_table, generic_inv_approx_table
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_core.opt.opt_utils import (
    forward_attributes, forward_stage_attributes
)

S2 = sollya.SollyaObject(2)

def get_compatible_bool_format(optree):
    """ Return a boolean format whose vector-size is compatible
        with optree format """
    if optree.get_precision().is_vector_format():
        scalar_format = optree.get_precision().get_scalar_format()
        scalar_size = scalar_format.get_bit_size()
        vector_size = optree.get_precision().get_vector_size()
        return VECTOR_TYPE_MAP[ML_Bool][scalar_size][vector_size]
    else:
        return ML_Bool


def minmax_legalizer_wrapper(predicate, bool_prec=None):
    """ Legalize a min/max node by converting it to a Select operation
        with the predicate given as argument """
    def minmax_legalizer(optree):
        op0 = optree.get_input(0)
        op1 = optree.get_input(1)
        local_bool_prec = get_compatible_bool_format(optree) if bool_prec is None else bool_prec
        comp = Comparison(
            op0, op1, specifier=predicate,
            precision=local_bool_prec,
            tag="minmax_pred"
        )
        # forward_stage_attributes(optree, comp)
        result = Select(
            comp,
            op0, op1,
            precision=optree.get_precision()
        )
        forward_attributes(optree, result)
        return result
    return minmax_legalizer

## Min node legalizer
min_legalizer = minmax_legalizer_wrapper(Comparison.Less)
## Max node legalizer
max_legalizer = minmax_legalizer_wrapper(Comparison.Greater)


def safe(operation):
    """ function decorator to forward None value """
    return lambda *args: None if None in args else operation(*args)

# Metalibm Description Language operation predicate
def is_addition(optree):
    return isinstance(optree, Addition)
def is_subtraction(optree):
    return isinstance(optree, Subtraction)
def is_multiplication(optree):
    return isinstance(optree, Multiplication)
def is_division(optree):
    return isinstance(optree, Division)
def is_constant(optree):
    return isinstance(optree, Constant)

def is_max(optree):
    """ Max operation predicate """
    return isinstance(optree, Max)
def is_min(optree):
    """ Min operation predicate """
    return isinstance(optree, Min)
def is_function_call(node):
    """ determine if node is a FunctionCall node """
    return isinstance(node, FunctionCall)

def evaluate_bin_op(operation, recursive_eval):
    """ functor to create a 2-operand formal evaluation
        using operator @p operation """
    def eval_routine(optree):
        lhs = optree.get_input(0)
        rhs = optree.get_input(1)
        return safe(operation)(
            recursive_eval(lhs),
            recursive_eval(rhs)
        )
    return eval_routine

def evaluate_un_op(operation, recursive_eval):
    """ functor to create a 1-operand formal evaluation
        using operator @p operation """
    def eval_routine(optree):
        op = optree.get_input(0)
        return safe(operation)(
            recursive_eval(op),
        )
    return eval_routine

def is_fixed_point_position(optree):
    return isinstance(optree, FixedPointPosition)
def is_negation(optree):
    return isinstance(optree, Negation)

def default_prec_solver(optree):
    return optree.get_precision()

def fixed_point_position_legalizer(optree, input_prec_solver=default_prec_solver):
    """ Legalize a FixedPointPosition node to a constant """
    assert isinstance(optree, FixedPointPosition)
    fixed_input = optree.get_input(0)
    fixed_precision = input_prec_solver(fixed_input)
    Log.report(Log.Verbose, "fixed_precision of {} is {}".format(fixed_input, fixed_precision))
    if not is_fixed_point(fixed_precision):
        Log.report(
            Log.Error,
            "in fixed_point_position_legalizer: precision of {} should be fixed-point but is {}".format(
                fixed_input,
                fixed_precision
            )
        )

    position = optree.get_input(1).get_value()

    align = optree.get_align()

    value_computation_map = {
        FixedPointPosition.FromLSBToLSB: position,
        FixedPointPosition.FromMSBToLSB: fixed_precision.get_bit_size() - 1 - position,
        FixedPointPosition.FromPointToLSB: fixed_precision.get_frac_size() + position,
        FixedPointPosition.FromPointToMSB: fixed_precision.get_integer_size() - position
    }
    cst_value = value_computation_map[align]
    # display value
    Log.report(Log.LogLevel("FixedPoint"), "fixed-point position {tag} has been resolved to {value}".format(
        tag=optree.get_tag(),
        value=cst_value
        )
    )
    result = Constant(
        cst_value,
        precision=ML_Integer
    )
    forward_attributes(optree, result)
    return result

def evaluate_graph(optree, value_mapping, fct_mapping):
    """ evaluate value for graph optree assuming
        the nodes in value_mapping have the associated value
        this function does not support FixedPointPosition """
    def recursive_eval(node):
        if node in value_mapping:
            return value_mapping[node]
        for predicate in evaluation_map:
            if predicate(node):
                return evaluation_map[predicate](node)
        return None

    def eval_func_call(node):
        # FIXME: need cleaner way of linking numeric_emulate
        func_tag = node.get_function_object().name
        op = node.get_input(0)
        return safe(fct_mapping[func_tag])(recursive_eval(op))

    evaluation_map = {
        is_max:
            lambda optree: evaluate_bin_op(max, recursive_eval)(optree),
        is_min:
            lambda optree: evaluate_bin_op(min, recursive_eval)(optree),
        is_addition:
            lambda optree: evaluate_bin_op(operator.__add__, recursive_eval)(optree),
        is_subtraction:
            lambda optree: evaluate_bin_op(operator.__sub__, recursive_eval)(optree),
        is_multiplication:
            lambda optree: evaluate_bin_op(operator.__mul__, recursive_eval)(optree),
        is_division:
            lambda optree: evaluate_bin_op(operator.__truediv__, recursive_eval)(optree),
        is_constant:
            lambda optree: optree.get_value(),
        is_negation:
            lambda optree: evaluate_un_op(operator.__neg__, recursive_eval)(optree),
        is_function_call:
            lambda optree: eval_func_call(optree),
    }
    return recursive_eval(optree)

def evaluate_cst_graph(optree, input_prec_solver=default_prec_solver):
    """ evaluate a Operation Graph if its leaves are Constant """
    def recursive_eval(node):
        for predicate in evaluation_map:
            if predicate(node):
                result = evaluation_map[predicate](node)
                return result
        return None
    evaluation_map = {
        is_max:
            lambda optree: evaluate_bin_op(max, recursive_eval)(optree),
        is_min:
            lambda optree: evaluate_bin_op(min, recursive_eval)(optree),
        is_addition:
            lambda optree: evaluate_bin_op(operator.__add__, recursive_eval)(optree),
        is_subtraction:
            lambda optree: evaluate_bin_op(operator.__sub__, recursive_eval)(optree),
        is_multiplication:
            lambda optree: evaluate_bin_op(operator.__mul__, recursive_eval)(optree),
        is_constant:
            lambda optree: optree.get_value(),
        is_negation:
            lambda optree: evaluate_un_op(operator.__neg__, recursive_eval)(optree),
        is_fixed_point_position:
            lambda optree: recursive_eval(
                fixed_point_position_legalizer(optree, input_prec_solver=input_prec_solver)
            ),
    }
    return recursive_eval(optree)

def legalize_fixed_point_subselection(optree, input_prec_solver = default_prec_solver):
    """ Legalize a SubSignalSelection on a fixed-point node """
    sub_input = optree.get_input(0)
    inf_index = evaluate_cst_graph(optree.get_inf_index(), input_prec_solver = input_prec_solver)
    sup_index = evaluate_cst_graph(optree.get_sup_index(), input_prec_solver = input_prec_solver)
    assert not inf_index is None and not sup_index is None
    input_format = ML_StdLogicVectorFormat(sub_input.get_precision().get_bit_size())
    output_format = ML_StdLogicVectorFormat(sup_index - inf_index + 1)
    subselect = SubSignalSelection(
        TypeCast(
            sub_input,
            precision = input_format
        ),
        inf_index,
        sup_index,
        precision=output_format
    )
    result = TypeCast(
        subselect,
        precision = optree.get_precision(),
    )
    # TypeCast may be simplified during code generation so attributes
    # may also be forwarded to previous node
    forward_attributes(optree, subselect)
    forward_attributes(optree, result)
    return result

def subsignalsection_legalizer(optree, input_prec_solver = default_prec_solver):
    if is_fixed_point(optree.get_precision()) and is_fixed_point(optree.get_input(0).get_precision()):
        return legalize_fixed_point_subselection(optree, input_prec_solver = input_prec_solver)
    else:
        return optree

def vectorize_cst(value, cst_precision):
    """ if cst_precision is a vector format return a list uniformly initialized
        with value whose size macthes the vector size of cst_precision, else
        return the scalar value """
    if cst_precision.is_vector_format():
        return [value] * cst_precision.get_vector_size()
    else:
        return value

def generate_field_extraction(optree, precision, lo_index, hi_index):
    """ extract bit-field optree[lo_index:hi_index-1] and cast to precision """
    if optree.precision != precision:
        optree = TypeCast(optree, precision=precision)
    result = optree
    if lo_index != 0:
        result = BitLogicRightShift(
            optree,
            Constant(vectorize_cst(lo_index, precision), precision=precision),
            precision=precision
        )
    if (hi_index - lo_index + 1) != precision.get_bit_size():
        mask = Constant(
            vectorize_cst(2**(hi_index-lo_index+1) - 1, precision),
            precision=precision)
        result = BitLogicAnd(
            result,
            mask,
            precision=precision
        )
    return result

def generate_raw_exp_extraction(optree):
    """ Generate an expanded implementation of ExponentExtraction node
        with @p optree as input """
    if optree.precision.is_vector_format():
        base_precision = optree.precision.get_scalar_format()
        int_precision = {
            v2float32: v2int32,
            v2float64: v2int64,

            v4float32: v4int32,
            v4float64: v4int64,

            v8float32: v8int32,
            v8float64: v8int64,
        }[optree.precision]
        #base_precision.get_integer_format()
    else:
        base_precision = optree.precision
        int_precision = base_precision.get_integer_format()
    return generate_field_extraction(
        optree,
        int_precision,
        base_precision.get_field_size(),
        base_precision.get_field_size() + base_precision.get_exponent_size() - 1
    )


def generate_exp_extraction(optree):
    if optree.precision.is_vector_format():
        base_precision = optree.precision.get_scalar_format()
        vector_size = optree.precision.get_vector_size()
        int_precision = {
            v2float32: v2int32,
            v2float64: v2int64,

            v4float32: v4int32,
            v4float64: v4int64,

            v8float32: v8int32,
            v8float64: v8int64,
        }[optree.precision]
        #base_precision.get_integer_format()
        bias_cst = [base_precision.get_bias()] * vector_size
    else:
        base_precision = optree.precision
        int_precision = base_precision.get_integer_format()
        bias_cst = base_precision.get_bias()
    return Addition(
        generate_raw_exp_extraction(optree),
        Constant(bias_cst, precision=int_precision),
        precision=int_precision
    )


def generate_raw_mantissa_extraction(optree):
    int_precision = optree.precision.get_integer_format()
    return generate_field_extraction(
        optree,
        int_precision,
        0,
        optree.precision.get_field_size() - 1,
    )

def generate_exp_insertion(optree, result_precision):
    """ generate the expanded version of ExponentInsertion
        with @p optree as input and assuming @p result_precision
        as output precision """
    if result_precision.is_vector_format():
        scalar_format = optree.precision.get_scalar_format()
        vector_size = optree.precision.get_vector_size()
        # determine the working format (for expression)
        work_format = VECTOR_TYPE_MAP[result_precision.get_scalar_format().get_integer_format()][vector_size] 
        bias_cst = [-result_precision.get_scalar_format().get_bias()] * vector_size
        shift_cst = [result_precision.get_scalar_format().get_field_size()] * vector_size
    else:
        scalar_format = optree.precision
        work_format = result_precision.get_integer_format()
        bias_cst = -result_precision.get_bias() 
        shift_cst = result_precision.get_field_size()
    if not is_std_integer_format(scalar_format):
        Log.report(
            Log.Error,
            "{} should be a std integer format in generate_exp_insertion {} with precision {}",
            scalar_format, optree, result_precision
        )
    assert is_std_integer_format(scalar_format)
    biased_exponent = Addition(
        Conversion(optree, precision=work_format) if not optree.precision is work_format else optree,
        Constant(
            bias_cst,
            precision=work_format),
        precision=work_format
    )
    result = BitLogicLeftShift(
        biased_exponent,
        Constant(
            shift_cst,
            precision=work_format,
        ),
        precision=work_format
    )
    return TypeCast(
        result,
        precision=result_precision
    )

def legalize_exp_insertion(result_precision):
    """ Legalize an ExponentInsertion node to a sequence of basic
        operations """
    def legalizer(exp_insertion_node):
        optree = exp_insertion_node.get_input(0)
        return generate_exp_insertion(optree, result_precision)
    return legalizer

def generate_test_expansion(predicate, test_input):
    """ transform a Test optree into a sequence of basic
        node """
    test_bool_format = get_compatible_bool_format(test_input)
    if test_input.precision.is_vector_format():
        input_scalar_precision = test_input.precision.get_scalar_format()
        vector_size = test_input.precision.get_vector_size()
        int_precision = {
            ML_Int32: {
                2: v2int32,
                4: v4int32,
                8: v8int32,
            },
            ML_Int64: {
                2: v2int64,
                4: v4int64,
                8: v8int64,
            }
        }[input_scalar_precision.get_integer_format()][vector_size]
        nanorinf_cst =  [input_scalar_precision.get_nanorinf_exp_field()] * vector_size
        zero_cst = [0] * vector_size
        one_cst = [1] * vector_size
    else:
        input_scalar_precision = test_input.precision
        int_precision = input_scalar_precision.get_integer_format()
        nanorinf_cst =  input_scalar_precision.get_nanorinf_exp_field()
        zero_cst = 0
        one_cst = 1
    if predicate is Test.IsInfOrNaN:
        return Comparison(
            generate_raw_exp_extraction(test_input),
            Constant(nanorinf_cst, precision=int_precision),
            specifier=Comparison.Equal,
            precision=test_bool_format
        )
    elif predicate is Test.IsNaN:
        return LogicalAnd(
            Comparison(
                generate_raw_exp_extraction(test_input),
                Constant(nanorinf_cst, precision=int_precision),
                specifier=Comparison.Equal,
                precision=test_bool_format
            ),
            Comparison(
                generate_raw_mantissa_extraction(test_input),
                Constant(zero_cst, precision=int_precision),
                specifier=Comparison.NotEqual,
                precision=test_bool_format
            ),
            precision=test_bool_format
        )
    elif predicate is Test.IsSubnormal:
        return Comparison(
            generate_raw_exp_extraction(test_input),
            Constant(zero_cst, precision=int_precision),
            specifier=Comparison.Equal,
            precision=test_bool_format
        )
    elif predicate is Test.IsSignalingNaN:
        quiet_bit_index = input_scalar_precision.get_field_size() - 1
        return LogicalAnd(
            Comparison(
                generate_raw_exp_extraction(test_input),
                Constant(nanorinf_cst, precision=int_precision),
                specifier=Comparison.Equal,
                precision=test_bool_format
            ),
            LogicalAnd(
                Comparison(
                    generate_raw_mantissa_extraction(test_input),
                    Constant(zero_cst, precision=int_precision),
                    specifier=Comparison.NotEqual,
                    precision=test_bool_format
                ),
                Comparison(
                    generate_field_extraction(test_input, int_precision, quiet_bit_index, quiet_bit_index),
                    Constant(zero_cst, precision=int_precision),
                    specifier=Comparison.Equal,
                    precision=test_bool_format
                ),
                precision=test_bool_format
            ),
            precision=test_bool_format
        )
    elif predicate is Test.IsQuietNaN:
        quiet_bit_index = input_scalar_precision.get_field_size() - 1
        return LogicalAnd(
            Comparison(
                generate_raw_exp_extraction(test_input),
                Constant(nanorinf_cst, precision=int_precision),
                specifier=Comparison.Equal,
                precision=test_bool_format
            ),
            LogicalAnd(
                Comparison(
                    generate_raw_mantissa_extraction(test_input),
                    Constant(zero_cst, precision=int_precision),
                    specifier=Comparison.NotEqual,
                    precision=test_bool_format
                ),
                Comparison(
                    generate_field_extraction(test_input, int_precision, quiet_bit_index, quiet_bit_index),
                    Constant(one_cst, precision=int_precision),
                    specifier=Comparison.Equal,
                    precision=test_bool_format
                ),
                precision=test_bool_format
            ),
            precision=test_bool_format
        )
    elif predicate is Test.IsInfty:
        return LogicalAnd(
            Comparison(
                generate_raw_exp_extraction(test_input),
                Constant(nanorinf_cst, precision=int_precision),
                specifier=Comparison.Equal,
                precision=test_bool_format
            ),
            Comparison(
                generate_raw_mantissa_extraction(test_input),
                Constant(zero_cst, precision=int_precision),
                specifier=Comparison.Equal,
                precision=test_bool_format
            ),
            precision=test_bool_format
        )
    elif predicate is Test.IsZero:
        # extract all bits except the sign and compare with zero
        hi_index = input_scalar_precision.get_bit_size() - 2
        return Comparison(
            generate_field_extraction(test_input, int_precision, 0, hi_index),
            Constant(zero_cst, precision=int_precision),
            specifier=Comparison.Equal,
            precision=test_bool_format
        )
    else:
        Log.report(Log.Error, "unsupported predicate in generate_test_expansion: {}".format(predicate),
                   error=NotImplementedError)

def legalize_test(optree):
    return generate_test_expansion(optree.specifier, optree.get_input(0))

def legalize_comp_sign(node):
    """ legalize a Test.CompSign node to a series of
        comparison with 0 and logical operation """
    # TODO/IDEA: could also be implemented by two 2 copy sign with 1.0 and valuda
    # comparison
    lhs = node.get_input(0)
    lhs_zero = Constant(0, precision=lhs.get_precision())
    rhs = node.get_input(1)
    rhs_zero = Constant(0, precision=rhs.get_precision())
    return LogicalOr(
        LogicalAnd(lhs >= lhs_zero, rhs >= rhs_zero),
        LogicalAnd(lhs <= lhs_zero, rhs <= rhs_zero),
    )

def legalize_fma_to_std(node):
    """ legalize any FusedMultiplyAdd node to a
        FusedMultiplyAdd.Standard node """
    # map of specifier -> input transformer, output transformer
    SPECIFIER_TRANSFORMER = {
        FusedMultiplyAdd.Standard:
            ((lambda op: op.inputs),
            (lambda op: op)),
        FusedMultiplyAdd.Negate:
            ((lambda op: op.inputs),
            (lambda op: Negation(op, precision=op.get_precision()))),
        FusedMultiplyAdd.Subtract:
            ((lambda op: (op.get_input(0), op.get_input(1), Negation(op.get_input(2), precision=op.get_input(2).get_precision()))),
            (lambda op: op)),
        FusedMultiplyAdd.SubtractNegate:
            ((lambda op: (
                Negation(op.get_input(0), precision=op.get_input(2).get_precision()),
                op.get_input(1), op.get_input(2))),
            (lambda op: op)),
    }
    transformation = SPECIFIER_TRANSFORMER[node.specifier]
    legalized_inputs = transformation[0](node)
    pre_node = FusedMultiplyAdd(
        legalized_inputs[0], legalized_inputs[1], legalized_inputs[2],
        specifier=FusedMultiplyAdd.Standard,
        precision=node.get_precision()
    )
    return transformation[1](pre_node)


def legalize_reciprocal_seed(optree):
    """ Legalize an ReciprocalSeed optree """
    assert isinstance(optree, ReciprocalSeed)
    op_prec = optree.get_precision()
    initial_prec = op_prec
    back_convert = False
    op_input = optree.get_input(0)

    INV_APPROX_TABLE_FORMAT = generic_inv_approx_table.get_storage_precision()

    if op_prec != INV_APPROX_TABLE_FORMAT:
        op_input = Conversion(op_input, precision=INV_APPROX_TABLE_FORMAT)
        op_prec = INV_APPROX_TABLE_FORMAT
        back_convert = True
    # input = 1.m_hi-m_lo * 2^e
    # approx = 2^(-int(e/2)) * approx_insqrt(1.m_hi) * (e % 2 ? 1.0 : ~2**-0.5)

    # TODO: fix integer precision selection
    #       as we are in a late code generation stage, every node's precision
    #       must be set
    int_prec = op_prec.get_integer_format()
    op_sign = CopySign(op_input, Constant(1.0, precision=op_prec), precision=op_prec)
    op_exp = ExponentExtraction(op_input, tag="op_exp", debug=debug_multi, precision=int_prec)
    neg_exp = Negation(op_exp, precision=int_prec)
    approx_exp = ExponentInsertion(neg_exp, tag="approx_exp", debug=debug_multi, precision=op_prec)
    table_index = generic_inv_approx_table.get_index_function()(op_input)
    table_index.set_attributes(tag="inv_index", debug=debug_multi)
    approx = Multiplication(
        TableLoad(
            generic_inv_approx_table,
            table_index,
            precision=op_prec
        ),
        Multiplication(
            approx_exp,
            op_sign,
            precision=op_prec
        ),
        tag="inv_approx",
        debug=debug_multi,
        precision=op_prec
    )
    if back_convert:
        return Conversion(approx, precision=initial_prec)
    else:
        return approx

def legalize_invsqrt_seed(optree):
    """ Legalize an InverseSquareRootSeed optree """
    assert isinstance(optree, ReciprocalSquareRootSeed) 
    op_prec = optree.get_precision()
    # input = 1.m_hi-m_lo * 2^e
    # approx = 2^(-int(e/2)) * approx_insqrt(1.m_hi) * (e % 2 ? 1.0 : ~2**-0.5)
    op_input = optree.get_input(0)
    convert_back = False
    approx_prec = ML_Binary32

    if op_prec != approx_prec:
        op_input = Conversion(op_input, precision=ML_Binary32)
        convert_back = True


    # TODO: fix integer precision selection
    #       as we are in a late code generation stage, every node's precision
    #       must be set
    op_exp = ExponentExtraction(op_input, tag="op_exp", debug=debug_multi, precision=ML_Int32)
    neg_half_exp = Division(
        Negation(op_exp, precision=ML_Int32),
        Constant(2, precision=ML_Int32),
        precision=ML_Int32
    )
    approx_exp = ExponentInsertion(neg_half_exp, tag="approx_exp", debug=debug_multi, precision=approx_prec)
    op_exp_parity = Modulo(
        op_exp, Constant(2, precision=ML_Int32), precision=ML_Int32)
    approx_exp_correction = Select(
        Equal(op_exp_parity, Constant(0, precision=ML_Int32)),
        Constant(1.0, precision=approx_prec),
        Select(
            Equal(op_exp_parity, Constant(-1, precision=ML_Int32)),
            Constant(S2**0.5, precision=approx_prec),
            Constant(S2**-0.5, precision=approx_prec),
            precision=approx_prec
        ),
        precision=approx_prec,
        tag="approx_exp_correction",
        debug=debug_multi
    )
    table_index = invsqrt_approx_table.get_index_function()(op_input)
    table_index.set_attributes(tag="invsqrt_index", debug=debug_multi)
    approx = Multiplication(
        TableLoad(
            invsqrt_approx_table,
            table_index,
            precision=approx_prec
        ),
        Multiplication(
            approx_exp_correction,
            approx_exp,
            precision=approx_prec
        ),
        tag="invsqrt_approx",
        debug=debug_multi,
        precision=approx_prec
    )
    if approx_prec != op_prec:
        return Conversion(approx, precision=op_prec)
    else:
        return approx



class Legalizer:
    """ """
    def __init__(self, input_prec_solver = lambda optree: optree.get_precision()):
        ## in case a node has no define precision @p input_prec_solver
        #  is used to determine one
        self.input_prec_solver = input_prec_solver

    def determine_precision(self, optree):
        return self.input_prec_solver(optree)

    def legalize_fixed_point_position(self, optree):
        return fixed_point_position_legalizer(optree, self.determine_precision)

    def legalize_subsignalsection(self, optree):
        return subsignalsection_legalizer(optree, self.determine_precision)
