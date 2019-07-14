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

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO

from metalibm_core.core.ml_operations import (
    ML_LeafNode, Select, Conversion, Comparison, Min, Max
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_operations import SubSignalSelection
from metalibm_core.core.ml_hdl_format import is_fixed_point

from metalibm_core.opt.p_size_datapath import (
    solve_format_Comparison, FormatSolver
)

from metalibm_core.core.legalizer import (
    subsignalsection_legalizer, fixed_point_position_legalizer,
    minmax_legalizer_wrapper
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
    if precision is None:
        Log.report(Log.Error, "None precision for Select:\n{}", optree)
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


def legalize_single_operation(optree, format_solver=None):
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
        return True, new_optree
    elif isinstance(optree, Min):
        new_optree = minmax_legalizer_wrapper(Comparison.Less)(optree)
        format_solver(new_optree)
        return True, new_optree
    elif isinstance(optree, Max):
        new_optree = minmax_legalizer_wrapper(Comparison.Greater)(optree)
        format_solver(new_optree)
        return True, new_optree
    return False, optree



## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
class Pass_RTLLegalize(OptreeOptimization):
    """ Legalization of RTL operations """
    pass_tag = "rtl_legalize"

    def __init__(self, target):
        """ pass initialization """
        OptreeOptimization.__init__(self, "rtl_Legalize", target)
        self.memoization_map = {}
        self.format_solver = FormatSolver()

    def legalize_operation_rec(self, optree):
        """ """
        # looking into memoization map
        if optree in self.memoization_map:
            return False, self.memoization_map[optree]

        # has the npde been modified ?
        arg_changed = False

        if isinstance(optree, ML_LeafNode):
            pass
        else:
            for index, op_input in enumerate(optree.get_inputs()):
                is_modified, new_node = self.legalize_operation_rec(op_input)
                if is_modified:
                    optree.set_input(index, new_node)
                    arg_changed = True

        local_changed, new_optree = legalize_single_operation(optree, self.format_solver)

        self.memoization_map[optree] = new_optree 
        return (local_changed or arg_changed), new_optree


    def execute(self, optree):
        """ pass execution """
        return self.legalize_operation_rec(optree)

Log.report(LOG_PASS_INFO, "Registering size_datapath pass")
# register pass
Pass.register(Pass_RTLLegalize)
