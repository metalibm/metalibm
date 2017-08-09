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

from metalibm_core.core.ml_operations import Comparison, Select, Constant
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_formats import ML_Bool, ML_Integer

def minmax_legalizer_wrapper(predicate):
    """ Legalize a min/max node by converting it to a Select operation
        with the predicate given as argument """
    def minmax_legalizer(optree):
        op0 = optree.get_input(0)
        op1 = optree.get_input(1)
        result = Select(Comparison(op0, op1, specifier = predicate, precision = ML_Bool), op0, op1)
        result.attributes = optree.attributes.get_copy()
        return result
    return minmax_legalizer

## Min node legalizer
min_legalizer = minmax_legalizer_wrapper(Comparison.Less)
## Max node legalizer
max_legalizer = minmax_legalizer_wrapper(Comparison.Greater)


def fixed_point_position_legalizer(optree):
    """ Legalize a FixedPointPosition node to a constant """
    assert isinstance(optree, FixedPointPosition)
    fixed_input = optree.get_input(0)
    fixed_precision = fixed_input.get_precision()

    position = optree.get_input(1).get_value()

    align = optree.get_align()

    value_computation_map = {
        FixedPointPosition.FromLSBToLSB: position,
        FixedPointPosition.FromMSBToLSB: fixed_precision.get_bit_size() - 1 - position,
        FixedPointPosition.FromPointToLSB: fixed_precision.get_frac_size() + position
    }
    cst_value = value_computation_map[align]
    return Constant(
        cst_value,
        precision = ML_Integer
    )

