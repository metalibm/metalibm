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
""" Optimization pass which finely tune datapath widths in
    RTL entities """

import sollya
import operator

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, ML_LeafNode, Conversion,
    Statement, ReferenceAssign, BitLogicNegate, Subtraction,
    SpecificOperation, Negation, BitLogicRightShift, BitLogicLeftShift,
    BitArithmeticRightShift,
    TypeCast,
    Min, Max, CountLeadingZeros, Multiplication,
    LogicalOr, LogicalAnd, LogicalNot
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_operations import (
    Process, ComponentInstance, Concatenation, SubSignalSelection,
    SignCast,
)
from metalibm_core.opt.rtl_fixed_point_utils import (
    test_format_equality,
    solve_equal_formats
)
from metalibm_core.core.ml_formats import (
    ML_Bool, ML_Integer
)
from metalibm_core.core.ml_hdl_format import (
    is_fixed_point, fixed_point, ML_StdLogic, ML_StdLogicVectorFormat,
    is_unevaluated_format
)
from metalibm_core.core.legalizer import (
    legalize_fixed_point_subselection, fixed_point_position_legalizer,
    evaluate_cst_graph
)
from metalibm_core.core.special_values import FP_SpecialValue

from .opt_utils import evaluate_range

from metalibm_core.utility.decorator import safe

from functools import reduce

# The pass implemented in this file processes an optree and replaces
#  each None precision by a std_logic_vector's like precision whose
#  size has been adapted to avoid overflow
# By default the pass assumes that operation are made on unsigned
# integers. If a node has an attached range attributes, then this
# attributes will be used to determine output live range and tune
# the output format



## determine generic operation precision
def solve_format_BooleanOp(optree):
    """ legalize BooleanOperation precision

        Args:
            optree (BooleanOperation): input boolean node

        Returns:
            (ML_Format): legalized optree format
    """
    if optree.get_precision() is None:
        optree.set_precision(ML_Bool)
    return optree.get_precision()

## Determine comparison node precision
def solve_format_Comparison(optree):
    """ Legalize Comparison precision

        Args:
            optree (Comparison): input node

        Returns:
            (ML_Format) legalized node format
    """
    assert isinstance(optree, Comparison)
    if optree.get_precision() is None:
        optree.set_precision(ML_Bool)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    merge_format = solve_equal_formats([lhs, rhs])
    propagate_format_to_input(merge_format, optree, [0, 1])
    return solve_format_BooleanOp(optree)

def solve_format_CLZ(optree):
    """ Legalize CountLeadingZeros precision
    
        Args:
            optree (CountLeadingZeros): input node
            
        Returns:
            ML_Format: legal format for CLZ
    """
    assert isinstance(optree, CountLeadingZeros)
    op_input = optree.get_input(0)
    input_precision = op_input.get_precision()

    if is_fixed_point(input_precision):
        if input_precision.get_signed():
            Log.report(Log.Warning , "signed format in solve_format_CLZ")
        # +1 for carry overflow
        int_size = int(sollya.floor(sollya.log2(input_precision.get_bit_size()))) + 1 
        frac_size = 0
        return fixed_point(
            int_size,
            frac_size,
            signed=False
        )
    else:
        Log.report(Log.Warning , "unsupported format in solve_format_CLZ")
        return optree.get_precision()


def solve_unevaluated_format(precision, format_solver):
    """ resolve unevaluated @p precision using format solver @p format_solver """
    assert is_unevaluated_format(precision)

    def node_evaluator(node):
        return evaluate_cst_graph(node, input_prec_solver=format_solver)

    evaluated_format = precision.evaluate(node_evaluator)
    Log.report(Log.Info , "Solving unevaluated format {} to {}", str(precision), str(evaluated_format))
    return evaluated_format

def solve_format_ArithOperation(optree,
    integer_size_func = lambda lhs_prec, rhs_prec: None,
    frac_size_func = lambda lhs_prec, rhs_prec: None,
    signed_func = lambda lhs, lhs_prec, rhs, rhs_prec: False,
    format_solver=None
    ):
    """ determining fixed-point format for a generic 2-op arithmetic
        operation (e.g. Multiplication, Addition, Subtraction)
    """
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()

    abstract_operation = (lhs_precision is ML_Integer) and (rhs_precision is ML_Integer)
    if abstract_operation:
        return ML_Integer

    if lhs_precision is ML_Integer:
        cst_eval = evaluate_cst_graph(lhs, input_prec_solver=format_solver)
        lhs_precision = solve_format_Constant(Constant(cst_eval))

    if rhs_precision is ML_Integer:
        cst_eval = evaluate_cst_graph(rhs, input_prec_solver=format_solver)
        rhs_precision = solve_format_Constant(Constant(cst_eval))

    if is_unevaluated_format(lhs_precision):
        lhs_precision = solve_unevaluated_format(lhs_precision, format_solver)

    if is_unevaluated_format(lhs_precision):
        rhs_precision = solve_unevaluated_format(rhs_precision, format_solver)

    if is_fixed_point(lhs_precision) and is_fixed_point(rhs_precision):
        # +1 for carry overflow
        int_size = integer_size_func(lhs_precision, rhs_precision)
        frac_size = frac_size_func(lhs_precision, rhs_precision)
        is_signed = signed_func(lhs, lhs_precision, rhs, rhs_precision)
        return fixed_point(
            int_size,
            frac_size,
            signed=is_signed
        )
    else:
        return optree.get_precision()

def addsub_signed_predicate(lhs, lhs_prec, rhs, rhs_prec, op=operator.__sub__, default=True):
    """ determine whether subtraction output on a signed or
        unsigned format """
    left_range = evaluate_range(lhs)
    right_range = evaluate_range(rhs)
    result_range = safe(op)(left_range, right_range)
    if result_range is None:
        return default
    elif sollya.inf(result_range) < 0:
        return True
    else:
        return False
def sub_signed_predicate(lhs, lhs_prec, rhs, rhs_prec):
    """ determine whether subtraction output on a signed or
        unsigned format """
    return addsub_signed_predicate(lhs, lhs_prec, rhs, rhs_prec, op=operator.__sub__)

def add_signed_predicate(lhs, lhs_prec, rhs, rhs_prec):
    """ determine whether subtraction output on a signed or
        unsigned format """
    if is_fixed_point(lhs_prec) and is_fixed_point(rhs_prec):
        default = lhs_prec.get_signed() or rhs_prec.get_signed()
    else:
        default = True
    return addsub_signed_predicate(lhs, lhs_prec, rhs, rhs_prec, op=operator.__add__, default=default)


## determine Addition node precision
def solve_format_Addition(optree, format_solver=None):
    """ Legalize Addition precision """
    assert isinstance(optree, Addition)

    return solve_format_ArithOperation(
        optree,
        lambda l, r: max(l.get_integer_size(), r.get_integer_size()) + 1,
        lambda l, r: max(l.get_frac_size(), r.get_frac_size()),
        add_signed_predicate,
        format_solver=format_solver
    )


## determine Multiplication node precision
def solve_format_Multiplication(optree, format_solver=None):
    """ Legalize Multiplication node """
    assert isinstance(optree, Multiplication)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()

    extra_sign_digit = 1 if rhs_precision.get_signed() ^ lhs_precision.get_signed() else 0

    return solve_format_ArithOperation(
        optree,
        lambda l, r: l.get_integer_size() + r.get_integer_size() + extra_sign_digit,
        lambda l, r: l.get_frac_size() + r.get_frac_size(),
        lambda l, lp, r, rp: lp.get_signed() or rp.get_signed(),
        format_solver=format_solver
    )


def solve_format_Subtraction(optree, format_solver=None):
    """ Legalize Subtraction node

        Args:
            optree (Subtraction): input subtraction node

        Returns:
            Subtraction: legalize subtraction node
    """
    assert isinstance(optree, Subtraction)

    def sub_integer_size(lhs, rhs):
        int_inc = 1 if not lhs.get_signed() else 0
        int_size = max(
            lhs.get_integer_size(),
            rhs.get_integer_size()
        ) + int_inc
        return int_size


    return solve_format_ArithOperation(
        optree,
        sub_integer_size,
        lambda lp, rp: max(lp.get_frac_size(), rp.get_frac_size()),
        sub_signed_predicate,
        format_solver=format_solver
    )


def solve_format_SpecificOperation(optree):
    assert isinstance(optree, SpecificOperation)
    specifier = optree.get_specifier()
    if specifier == SpecificOperation.CopySign:
        return ML_StdLogic
    else: 
        raise NotImplementedError

def solve_format_bitwise_op(optree):
    """ legalize format of bitwise logical operation

        Args:
            optree (ML_Operation): bitwise logical input node

        Returns:
            (ML_Format): legal format for input node
    """
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()

    if is_fixed_point(lhs_precision) and is_fixed_point(rhs_precision):
        assert(lhs_precision.get_integer_size() == rhs_precision.get_integer_size())
        assert(lhs_precision.get_frac_size() == rhs_precision.get_frac_size())
        return lhs_precision
    else:
        return optree.get_precision()

def solve_format_BitLogicNegate(optree):
    """ legalize format of bitwise logical operation

        Args:
            optree (ML_Operation): bitwise logical input node

        Returns:
            (ML_Format): legal format for input node
    """
    op_input = optree.get_input(0)
    input_precision = op_input.get_precision()

    return input_precision

def solve_format_Negation(optree):
    """ legalize format of bitwise logical operation

        Args:
            optree (ML_Operation): bitwise logical input node

        Returns:
            (ML_Format): legal format for input node
    """
    op_input = optree.get_input(0)
    input_precision = op_input.get_precision()
    # TODO: OPTIMIZATION, increment according to input range
    #       rather than according to input bit-size
    int_inc = 1 if not input_precision.get_signed() else 0

    int_size = input_precision.get_integer_size() + int_inc
    frac_size = input_precision.get_frac_size()

    return fixed_point(int_size, frac_size, signed = True)

def solve_format_SignCast(optree):
    """ Resolve the format for a SignCast node """
    assert isinstance(optree, SignCast)
    precision = optree.get_input(0).get_precision()
    int_size = precision.get_integer_size()
    frac_size = precision.get_frac_size()

    if optree.specifier is SignCast.Signed:
        signed_precision = fixed_point(int_size, frac_size, signed=True)
        return signed_precision
    elif optree.specifier is SignCast.Unsigned:
        unsigned_precision = fixed_point(int_size, frac_size, signed=False)
        return unsigned_precision
    else:
        Log.report(Log.Error, "unknown specifier {} in solve_format_SignCast".format(optree.specifier))
        
def solve_format_TypeCast(optree, format_solver):
    """ Resolve the format for a TypeCast node """
    assert isinstance(optree, TypeCast)
    precision = optree.get_precision()

    if is_unevaluated_format(precision):
        def node_evaluator(node):
            return evaluate_cst_graph(node, input_prec_solver=format_solver)
        evaluated_format = precision.evaluate(node_evaluator)
        Log.report(Log.Info , "Solving unevaluated format {} to {}", str(precision), str(evaluated_format))
        precision = evaluated_format

    return precision

def solve_format_shift(optree):
    """ Legalize shift node """
    assert isinstance(optree, BitLogicRightShift) or isinstance(optree, BitLogicLeftShift) or isinstance(optree, BitArithmeticRightShift)
    shift_input = optree.get_input(0)
    shift_input_precision = shift_input.get_precision()
    shift_amount = optree.get_input(1)

    shift_amount_prec = shift_amount.get_precision()
    if is_fixed_point(shift_amount_prec):
        sa_range = evaluate_range(shift_amount)
        if sollya.inf(sa_range) < 0:
            Log.report(Log.Error, "shift amount of {} may be negative {}\n".format(
                optree,
                sa_range
                )
            )
    if is_fixed_point(shift_input_precision):
        return shift_input_precision
    else:
        return optree.get_precision()

## determine Constant node precision
def solve_format_Constant(optree, input_prec_solver=None):
    """ Legalize Constant node """
    assert isinstance(optree, Constant)
    value = optree.get_value()
    if FP_SpecialValue.is_special_value(value):
        return optree.get_precision()
    elif is_unevaluated_format(optree.get_precision()):
        # TODO: dangerous complexity, this is the only elif-branch
        #       which require input_prec_solver != None
        assert not input_prec_solver is None
        # managing unevaluated format
        return solve_unevaluated_format(optree.get_precision(), input_prec_solver)
    elif not optree.get_precision() is None and optree.get_precision() != ML_Integer:
        # if precision is already set (manually forced), returns it
        return optree.get_precision()
    else:
        # fixed-point format solving
        frac_size = -1
        FRAC_THRESHOLD = 100 # maximum number of frac bit to be tested
        # TODO: fix
        for i in range(FRAC_THRESHOLD):
            if int(value*2**i) == value * 2**i:
                frac_size = i
                break
        if frac_size < 0:
            Log.report(Log.Error, "value {} is not an integer, from node:\n{}", value, optree)
        abs_value = abs(value)
        signed = value < 0
        # int_size = max(int(sollya.ceil(sollya.log2(abs_value+2**frac_size))), 0) + (1 if signed else 0)
        int_size = max(int(sollya.ceil(sollya.log2(abs_value + 1))), 0) + (1 if signed else 0)
        if frac_size == 0 and int_size == 0:
            int_size = 1
        return fixed_point(int_size, frac_size, signed=signed)

def solve_format_FixedPointPosition(optree):
    """ resolve the format of a FixedPointPosition Node """
    assert isinstance(optree, FixedPointPosition)
    return ML_Integer

def solve_format_Concatenation(optree):
    """ legalize Concatenation operation node """
    if not optree.get_precision() is None:
        return optree.get_precision()
    else:
        bit_size = reduce(lambda x, y: x + y, [op.get_precision().get_bit_size() for op in optree.get_inputs()], 0)
        return fixed_point(bit_size, 0, signed = False)

def solve_format_SubSignalSelection(optree, format_solver):
    """ Dummy legalization of SubSignalSelection operation node """
    if optree.get_precision() is None:
        select_input = optree.get_input(0)
        input_prec = select_input.get_precision()

        inf_index = evaluate_cst_graph(optree.get_inf_index(), input_prec_solver=format_solver)
        sup_index = evaluate_cst_graph(optree.get_sup_index(), input_prec_solver=format_solver)

        if is_fixed_point(input_prec):
            frac_size = input_prec.get_frac_size() - inf_index
            integer_size = input_prec.get_integer_size() - (input_prec.get_bit_size() - 1 - sup_index)
            if frac_size + integer_size <= 0:
                Log.report(Log.Error, "range determined for SubSignalSelection format [{}:{}] has negative size: {}, optree is {}", integer_size, frac_size, integer_size + frac_size, optree)
            return fixed_point(integer_size, frac_size)
        else:
            return ML_StdLogicVectorFormat(sup_index - inf_index + 1)
    else:
        return optree.get_precision()


def format_set_if_undef(optree, new_format):
    """ Define a new format to @p optree if no format was previously
        set. """
    if optree.get_precision() is None or is_unevaluated_format(optree.get_precision()):
        optree.set_precision(new_format)
    return optree.get_precision()


def solve_format_Select(optree):
    """ Legalize Select node """
    assert isinstance(optree, Select)
    cond = optree.get_input(0)
    solve_format_BooleanOp(cond)
    true_value = optree.get_input(1)
    false_value = optree.get_input(2)
    unified_format = solve_equal_formats([optree, true_value, false_value])
    format_set_if_undef(true_value, unified_format)
    format_set_if_undef(false_value, unified_format)
    return format_set_if_undef(optree, unified_format)

def solve_format_MinMax(optree):
    """ legalize Min or Max node """
    lhs_value = optree.get_input(0)
    rhs_value = optree.get_input(1)
    unified_format = solve_equal_formats([optree, lhs_value, rhs_value])
    format_set_if_undef(lhs_value, unified_format)
    format_set_if_undef(rhs_value, unified_format)
    return format_set_if_undef(optree, unified_format)


## Test if @p optree is a Operation node propagating format
#  if it does return the list of @p optree's input index
#   where a format should be propagated
def does_node_propagate_format(optree):
    """ Test whether @p optree propagate a format definition
        to its operand. """
    if isinstance(optree, Select):
        return [1, 2]
    return []


def is_constant(optree):
    """ Test if optree is a Constant node  """
    return isinstance(optree, Constant)


def format_does_fit(cst_optree, new_format):
    """ Test if @p cst_optree fits into the precision @p new_format """
    assert is_constant(cst_optree)
    assert is_fixed_point(new_format)
    min_format = solve_format_Constant(cst_optree)
    sign_bias = 1 if (new_format.get_signed() and not min_format.get_signed()) \
        else 0
    return (new_format.get_integer_size() - sign_bias) >= \
        min_format.get_integer_size() and \
        new_format.get_frac_size() >= min_format.get_frac_size() and \
           (new_format.get_signed() or not min_format.get_signed())


# propagate the precision @p new_format to every node in
#  @p optree_list with undefined precision or instanciate
#  a Conversion node if precisions differ
def propagate_format_to_input(new_format, optree, input_index_list):
    """ Propgate new_format to @p optree's input whose index is listed in
        @p input_index_list """
    for op_index in input_index_list:
        op_input = optree.get_input(op_index)
        if op_input.get_precision() is None:
            op_input.set_precision(new_format)
            index_list = does_node_propagate_format(op_input)
            propagate_format_to_input(new_format, op_input, index_list)
        elif not test_format_equality(new_format, op_input.get_precision()):
            if is_constant(op_input):
                if not is_fixed_point(new_format):
                    Log.report(
                        Log.Error,
                        "format {} during propagation to input {} of {} is not a fixed-point format",
                        new_format, op_input, optree
                    )
                elif format_does_fit(op_input, new_format):
                    Log.report(
                        Log.Info,
                        "Simplify Constant Conversion {} to larger Constant: {}",
                        op_input.get_str(display_precision=True) if Log.is_level_enabled(Log.Info) else "",
                        str(new_format)
                    )
                    new_input = op_input.copy()
                    new_input.set_precision(new_format)
                    optree.set_input(op_index, new_input)
                else:
                    Log.report(
                        Log.Error,
                        "Constant is about to be reduced to a too constrained format: {}",
                        op_input.get_str(display_precision=True) if Log.is_level_enabled(Log.Error) else ""
                    )
            else:
                new_input = Conversion(
                    op_input,
                    precision=new_format
                )
                optree.set_input(op_index, new_input)

## Test if @p optree can be skipped
#  during format legalization
def solve_skip_test(optree):
    if isinstance(optree, Process):
        return True
    if isinstance(optree, ComponentInstance):
        return True
    return False

class FormatSolver:
    def __init__(self):
        self.memoization_map = {}

    def __call__(self, optree):
        """ legacy API used as input_prec_solver in evaluate_cst_graph """
        return self.solve_format_rec(optree)

    ## Recursively propagate information on operation node
    #  and tries to legalize every unknown formats
    def solve_format_rec(self, optree):
        """ Recursively legalize formats from @p optree, using memoization_map
            to store resolved results """
        if optree in self.memoization_map:
            return self.memoization_map[optree]
        elif isinstance(optree, ML_LeafNode):
            new_format = optree.get_precision()
            if isinstance(optree, Constant):
                new_format = solve_format_Constant(optree, input_prec_solver=self)

            Log.report(Log.Verbose,
                "new format {} determined for Constant {}",
                str(new_format), optree.get_str(display_precision=True) if Log.is_level_enabled(Log.Verbose) else ""
            )

            # updating optree format
            #optree.set_precision(new_format)
            format_set_if_undef(optree, new_format)
            self.memoization_map[optree] = new_format

            return optree.get_precision()

        elif isinstance(optree, Statement):
            for op_input in optree.get_inputs():
                self.solve_format_rec(op_input)
            self.memoization_map[optree] = None
            return None
        elif isinstance(optree, ReferenceAssign):
            dst = optree.get_input(0)
            src = optree.get_input(1)
            src_precision = self.solve_format_rec(src)
            format_set_if_undef(dst, src_precision)
        elif solve_skip_test(optree):
            Log.report(Log.Verbose, "[solve_format_rec] skipping: {}",
                optree.get_str(display_precision=True, depth=2) if Log.is_level_enabled(Log.Verbose) else ""
            )
            self.memoization_map[optree] = None
            return None
        else:
            for op_input in optree.get_inputs():
                self.solve_format_rec(op_input)
            new_format = optree.get_precision()
            if not new_format is None and not is_unevaluated_format(new_format):
                Log.report(
                    Log.Verbose,
                    "format {} has already been determined for {}",
                    str(new_format), optree.get_str(display_precision=True) if Log.is_level_enabled(Log.Verbose) else ""
                )
            elif isinstance(optree, LogicalOr) or isinstance(optree, LogicalAnd) or isinstance(optree, LogicalNot):
                new_format = solve_format_BooleanOp(optree)
            elif isinstance(optree, Comparison):
                new_format = solve_format_Comparison(optree)
            elif isinstance(optree, CountLeadingZeros):
                new_format = solve_format_CLZ(optree)
            elif isinstance(optree, Multiplication):
                new_format = solve_format_Multiplication(optree, format_solver=self)
            elif isinstance(optree, Addition):
                new_format = solve_format_Addition(optree, format_solver=self)
            elif isinstance(optree, Subtraction):
                new_format = solve_format_Subtraction(optree, format_solver=self)
            elif isinstance(optree, SpecificOperation):
                new_format = solve_format_SpecificOperation(optree)
            elif isinstance(optree, Select):
                new_format = solve_format_Select(optree)
            elif isinstance(optree, Min) or isinstance(optree, Max):
                new_format = solve_format_MinMax(optree)
            elif isinstance(optree, BitLogicNegate):
                new_format = solve_format_BitLogicNegate(optree)
            elif isinstance(optree, Negation):
                new_format = solve_format_Negation(optree)
            elif isinstance(optree, BitLogicRightShift) or isinstance(optree, BitLogicLeftShift) or isinstance(optree, BitArithmeticRightShift):
                new_format = solve_format_shift(optree)
            elif isinstance(optree, Concatenation):
                new_format = solve_format_Concatenation(optree)
            elif isinstance(optree, SubSignalSelection):
                new_format = solve_format_SubSignalSelection(optree, self)
            elif isinstance(optree, FixedPointPosition):
                new_format = solve_format_FixedPointPosition(optree)
            elif isinstance(optree, SignCast):
                new_format = solve_format_SignCast(optree)
            elif isinstance(optree, TypeCast):
                new_format = solve_format_TypeCast(optree, format_solver=self)
            elif isinstance(optree, Conversion):
                Log.report(
                    Log.Error,
                    "Conversion {} must have a defined format, has {}".format(
                        optree.get_str(),
                        str(optree.get_precision(display_precision=True))
                    )
                )
            else:
                Log.report(
                    Log.Error,
                    "unsupported operation in solve_format_rec: {}".format(
                        optree.get_str(display_precision=True))
                )

            # updating optree format
            Log.report(Log.Verbose,
                "new format {} determined for {}",
                str(new_format), optree.get_str(display_precision=True) if Log.is_level_enabled(Log.Verbose) else ""
            )
            # optree.set_precision(new_format)
            real_format = format_set_if_undef(optree, new_format)
            self.memoization_map[optree] = real_format

            # format propagation
            prop_index_list = does_node_propagate_format(optree)
            propagate_format_to_input(new_format, optree, prop_index_list)

            return optree.get_precision()

## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
class Pass_SizeDatapath(OptreeOptimization):
    """ implementation of datapath sizing pass """
    pass_tag = "size_datapath"

    def __init__(self, target):
        """ pass initialization """
        OptreeOptimization.__init__(self, "size_datapath", target)
        self.format_solver = FormatSolver()

    def execute(self, optree):
        """ pass execution """
        return self.format_solver.solve_format_rec(optree)

Log.report(LOG_PASS_INFO, "Registering size_datapath pass")
# register pass
Pass.register(Pass_SizeDatapath)
