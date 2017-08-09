# -*- coding: utf-8 -*-
#
""" Optimization pass which finely tune datapath widths in
    RTL entities """

import sollya

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import OptreeOptimization, Pass
from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, ML_LeafNode, Conversion,
    Statement, ReferenceAssign, BitLogicNegate
)
from metalibm_core.core.ml_hdl_operations import (
    Process, ComponentInstance
)
from metalibm_core.opt.rtl_fixed_point_utils import (
    test_format_equality,
    solve_equal_formats
)
from metalibm_core.core.ml_formats import ML_Bool
from metalibm_core.core.ml_hdl_format import (
    is_fixed_point, fixed_point
)

# The pass implemented in this file processes an optree and replaces
#  each None precision by a std_logic_vector's like precision whose
#  size has been adapted to avoid overflow
# By default the pass assumes that operation are made on unsigned
# integers. If a node has an attached range attributes, then this
# attributes will be used to determine output live range and tune
# the output format



## determine generic operation precision
def solve_format_BooleanOp(optree):
    """ legalize BooleanOperation node

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
    """ Legalize Comparison node

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

## determine Addition node precision
def solve_format_Addition(optree):
    """ Legalize Addition node """
    assert isinstance(optree, Addition)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()

    if is_fixed_point(lhs_precision) and is_fixed_point(rhs_precision):
        # +1 for carry overflow
        int_size = max(
            lhs_precision.get_integer_size(),
            rhs_precision.get_integer_size()
        ) + 1
        frac_size = max(
            lhs_precision.get_frac_size(),
            rhs_precision.get_frac_size()
        )
        is_signed = lhs_precision.get_signed() or rhs_precision.get_signed()
        return fixed_point(
            int_size,
            frac_size,
            signed=is_signed
        )
    else:
        return optree.get_precision()

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

## determine Constant node precision
def solve_format_Constant(optree):
    """ Legalize Constant node """
    assert isinstance(optree, Constant)
    value = optree.get_value()
    assert int(value) == value
    abs_value = abs(value)
    signed = value < 0

    int_size = max(int(sollya.ceil(sollya.log2(abs_value))), 0) + (1 if signed else 0)
    frac_size = 0
    if frac_size == 0 and int_size == 0:
        int_size = 1
    return fixed_point(int_size, frac_size, signed=signed)


def format_set_if_undef(optree, new_format):
    """ Define a new format to @p optree if no format was previously
        set. """
    if optree.get_precision() is None:
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
                if format_does_fit(op_input, new_format):
                    Log.report(
                        Log.Info,
                        "Simplify Constant Conversion {} to larger Constant: {}".format(
                            op_input.get_str(display_precision=True),
                            str(new_format)
                        )
                    )
                    format_set_if_undef(op_input, new_format)
                else:
                    Log.report(
                        Log.Error,
                        "Constant is about to be reduced to a too constrained format: {}".format(
                            op_input.get_str(display_precision=True)
                        )
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

## Recursively propagate information on operation node
#  and tries to legalize every unknown formats
def solve_format_rec(optree, memoization_map=None):
    """ Recursively legalize formats from @p optree, using memoization_map
        to store resolved results """
    memoization_map = {} if memoization_map is None else memoization_map
    if optree in memoization_map:
        return memoization_map[optree]
    elif isinstance(optree, ML_LeafNode):
        new_format = optree.get_precision()
        if isinstance(optree, Constant):
            new_format = solve_format_Constant(optree)

        Log.report(Log.Info,
                   "new format {} determined for {}".format(
                       str(new_format), optree.get_str(display_precision=True)
                   )
                   )

        # updating optree format
        #optree.set_precision(new_format)
        format_set_if_undef(optree, new_format)
        memoization_map[optree] = new_format

    elif isinstance(optree, Statement):
        for op_input in optree.get_inputs():
            solve_format_rec(op_input)
        return None
    elif isinstance(optree, ReferenceAssign):
        dst = optree.get_input(0)
        src = optree.get_input(1)
        src_precision = solve_format_rec(src)
        format_set_if_undef(dst, src_precision)
    elif solve_skip_test(optree):
        pass
        return None
    else:
        for op_input in optree.get_inputs():
            solve_format_rec(op_input)
        new_format = optree.get_precision()
        if not new_format is None:
            Log.report(
                Log.Info,
                "format {} has already been determined for {}".format(
                    str(new_format), optree.get_str(display_precision=True)
                )
            )
        elif isinstance(optree, Comparison):
            new_format = solve_format_Comparison(optree)
        elif isinstance(optree, Addition):
            new_format = solve_format_Addition(optree)
        elif isinstance(optree, Select):
            new_format = solve_format_Select(optree)
        elif isinstance(optree, BitLogicNegate):
            new_format = solve_format_BitLogicNegate(optree) 
        elif isinstance(optree, Conversion):
            Log.report(
                Log.Error,
                "Conversion {} must have a defined format".format(
                    optree.get_str()
                )
            )
        else:
            Log.report(
                Log.Error,
                "unsupported operation in solve_format_rec: {}".format(
                    optree.get_str())
            )

        # updating optree format
        Log.report(Log.Info,
                   "new format {} determined for {}".format(
                       str(new_format), optree.get_str(display_precision=True)
                   )
                   )
        # optree.set_precision(new_format)
        real_format = format_set_if_undef(optree, new_format)
        memoization_map[optree] = real_format

        # format propagation
        prop_index_list = does_node_propagate_format(optree)
        propagate_format_to_input(new_format, optree, prop_index_list)

## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
class Pass_SizeDatapath(OptreeOptimization):
    """ implementation of datapath sizing pass """
    pass_tag = "size_datapath"

    def __init__(self, target):
        """ pass initialization """
        OptreeOptimization.__init__(self, "size_datapath", target)

    def execute(self, optree):
        """ pass execution """
        return solve_format_rec(optree, {})

print "Registering size_datapath pass"
# register pass
Pass.register(Pass_SizeDatapath)
