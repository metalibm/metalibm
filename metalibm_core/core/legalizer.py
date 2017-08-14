# -*- coding: utf-8 -*-

###############################################################################
# This file is part of New Metalibm tool
# Copyrights  Nicolas Brunie (2017)
# All rights reserved
# created:          Aug 8th, 2017
# last-modified:    Aug 8th, 2017
#
# author(s):    Nicolas Brunie (nibrunie@gmail.com)
# description:
###############################################################################

from metalibm_core.core.ml_operations import (
    Comparison, Select, Constant, TypeCast, Multiplication, Addition,
    Subtraction, Negation
)
from metalibm_core.core.ml_hdl_operations import (
    SubSignalSelection
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_formats import ML_Bool, ML_Integer
from metalibm_core.core.ml_hdl_format import (
    is_fixed_point, ML_StdLogicVectorFormat
)

from metalibm_core.opt.opt_utils import forward_attributes


def minmax_legalizer_wrapper(predicate):
    """ Legalize a min/max node by converting it to a Select operation
        with the predicate given as argument """
    def minmax_legalizer(optree):
        op0 = optree.get_input(0)
        op1 = optree.get_input(1)
        result = Select(
            Comparison(
                op0, op1, specifier = predicate,
                precision = ML_Bool,
                tag = "minmax_pred"
            ), op0, op1,
            precision = optree.get_precision()
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

def is_addition(optree):
    return isinstance(optree, Addition)
def is_subtraction(optree):
    return isinstance(optree, Subtraction)
def is_multiplication(optree):
    return isinstance(optree, Multiplication)
def is_constant(optree):
    return isinstance(optree, Constant)


def evalute_bin_op(operation):
    def eval_routine(optree):
        lhs = optree.get_input(0)
        rhs = optree.get_input(1)
        return safe(operation)(
            evaluate_cst_graph(lhs),
            evaluate_cst_graph(rhs)
        )
    return eval_routine

def evalute_un_op(operation):
    def eval_routine(optree):
        op = optree.get_input(0)
        return safe(operation)(
            evaluate_cst_graph(op),
        )
    return eval_routine

def is_fixed_point_position(optree):
    return isinstance(optree, FixedPointPosition)
def is_negation(optree):
    return isinstance(optree, Negation)

def default_prec_solver(optree):
    return optree.get_precision()

def fixed_point_position_legalizer(optree, input_prec_solver = default_prec_solver):
    """ Legalize a FixedPointPosition node to a constant """
    assert isinstance(optree, FixedPointPosition)
    fixed_input = optree.get_input(0)
    fixed_precision = input_prec_solver(fixed_input)
    assert is_fixed_point(fixed_precision)

    position = optree.get_input(1).get_value()

    align = optree.get_align()

    value_computation_map = {
        FixedPointPosition.FromLSBToLSB: position,
        FixedPointPosition.FromMSBToLSB: fixed_precision.get_bit_size() - 1 - position,
        FixedPointPosition.FromPointToLSB: fixed_precision.get_frac_size() + position
    }
    cst_value = value_computation_map[align]
    result = Constant(
        cst_value,
        precision = ML_Integer
    )
    forward_attributes(optree, result)
    return result

def evaluate_cst_graph(optree, input_prec_solver = default_prec_solver):
    """ evaluate a Operation Graph if its leaves are Constant """
    evaluation_map = {
        is_addition:
            lambda optree: evaluate_bin_op(operator.__add__),
        is_subtraction:
            lambda optree: evaluate_bin_op(operator.__sub__),
        is_multiplication:
            lambda optree: evaluate_bin_op(operator.__mul__),
        is_constant:
            lambda optree: optree.get_value(),
        is_negation:
            lambda optree: evalute_un_op(operator.__neg__),
        is_fixed_point_position:
            lambda optree: evaluate_cst_graph(
                fixed_point_position_legalizer(optree, input_prec_solver = input_prec_solver)
            ),
    }
    for predicate in evaluation_map:
        if predicate(optree):
            return evaluation_map[predicate](optree)
    return None

def legalize_fixed_point_subselection(optree, input_prec_solver = default_prec_solver):
    """ Legalize a SubSignalSelection on a fixed-point node """
    sub_input = optree.get_input(0)
    inf_index = evaluate_cst_graph(optree.get_inf_index(), input_prec_solver = input_prec_solver)
    sup_index = evaluate_cst_graph(optree.get_sup_index(), input_prec_solver = input_prec_solver)
    assert not inf_index is None and not sup_index is None
    input_format = ML_StdLogicVectorFormat(sub_input.get_precision().get_bit_size())
    output_format = ML_StdLogicVectorFormat(sup_index - inf_index + 1)
    result = TypeCast(
        SubSignalSelection(
            TypeCast(
                sub_input,
                precision = input_format
            ),
            inf_index,
            sup_index,
            precision = output_format
        ),
        precision = optree.get_precision(),
    )
    forward_attributes(optree, result)
    return result

def subsignalsection_legalizer(optree, input_prec_solver = default_prec_solver):
    if is_fixed_point(optree.get_precision()) and is_fixed_point(optree.get_input(0).get_precision()):
        return legalize_fixed_point_subselection(optree, input_prec_solver = input_prec_solver)
    else:
        return optree

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






