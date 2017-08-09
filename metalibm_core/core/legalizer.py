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

from metalibm_core.core.ml_operations import Comparison, Select
from metalibm_core.core.ml_formats import ML_Bool

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

min_legalizer = minmax_legalizer_wrapper(Comparison.Less)
max_legalizer = minmax_legalizer_wrapper(Comparison.Greater)
