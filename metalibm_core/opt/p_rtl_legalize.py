# -*- coding: utf-8 -*-
#
""" Optimization pass which finely tune datapath widths in
    RTL entities """

import sollya

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import OptreeOptimization, Pass

from metalibm_core.core.ml_operations import (
    ML_LeafNode, Select, Conversion, Comparison
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_operations import SubSignalSelection
from metalibm_core.core.ml_hdl_format import is_fixed_point

from metalibm_core.opt.p_size_datapath import (
    solve_format_Comparison
)

from metalibm_core.core.legalizer import (
    subsignalsection_legalizer, fixed_point_position_legalizer
)

###############################################################################
# PASS DESCRIPTION:
# The pass implemented in this file processes an optree and  legalize every 
# supported node
# the output format
###############################################################################

def legalize_Select(optree):
    """ legalize Select operation node by converting if and else inputs to
        Select output format if the bit sizes do not match """
    cond = optree.get_input(0)
    op0 = optree.get_input(1)
    op1 = optree.get_input(2)
    precision = optree.get_precision()
    if op0.get_precision().get_bit_size() != precision.get_bit_size():
        optree.set_input(
            1,
            Conversion(
                op0,
                precision = precision
            )
        )
    if op1.get_precision().get_bit_size() != precision.get_bit_size():
        optree.set_input(
            2,
            Conversion(
                op1,
                precision = optree.get_precision()
            )
        )
    return optree

def legalize_Comparison(optree):
    assert isinstance(optree, Comparison)
    new_format = solve_format_Comparison(optree)
    return optree


def legalize_single_operation(optree):
    if isinstance(optree, SubSignalSelection):
        new_optree = subsignalsection_legalizer(optree)
        return True, new_optree
    elif isinstance(optree, FixedPointPosition):
        new_optree = fixed_point_position_legalizer(optree)
        return True, new_optree
    elif isinstance(optree, Select):
        new_optree = legalize_Select(optree)
        return True, new_optree
    elif isinstance(optree, Comparison):
        new_optree = legalize_Comparison(optree)
        return True, optree
    return False, optree


def legalize_operation_rec(optree, memoization_map = None):
    """ """
    memoization_map = {} if memoization_map is None else memoization_map

    # looking into memoization map
    if optree in memoization_map:
        return optree

    # has the npde been modified ?
    arg_changed = False

    if isinstance(optree, ML_LeafNode):
        pass
    else:
        for index, op_input in enumerate(optree.get_inputs()):
            is_modified, new_node = legalize_operation_rec(op_input)
            if is_modified:
                optree.set_input(index, new_node)
                arg_changed = True

    local_changed, new_optree = legalize_single_operation(optree)

    memoization_map[optree] = optree 
    return local_changed or arg_changed, new_optree


## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
class Pass_RTLLegalize(OptreeOptimization):
    """ implementation of datapath sizing pass """
    pass_tag = "rtl_legalize"

    def __init__(self, target):
        """ pass initialization """
        OptreeOptimization.__init__(self, "rtl_Legalize", target)

    def execute(self, optree):
        """ pass execution """
        return legalize_operation_rec(optree, {})

print "Registering size_datapath pass"
# register pass
Pass.register(Pass_RTLLegalize)
