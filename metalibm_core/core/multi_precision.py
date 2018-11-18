# -*- coding: utf-8 -*-

## @package multi_precisions
#  Metalibm utility to manipulate multi-precision nodes

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
# created:          Nov 11th, 2018
# last-modified:    Nov 11th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.core.ml_operations import (
    Comparison, LogicalAnd, LogicalOr, BuildFromComponent,
    VectorElementSelection,
)
from metalibm_core.core.ml_formats import ML_Bool

from metalibm_core.opt.opt_utils import forward_attributes

from metalibm_core.utility.log_report import Log


def legalize_mp_2elt_comparison(optree):
    """ Transform comparison on ML_Compound_FP_Format object into
        comparison on sub-fields """
    specifier = optree.specifier
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    # TODO/FIXME: assume than multi-limb operand are normalized
    if specifier == Comparison.Equal:
        return LogicalAnd(
            Comparison(lhs.hi, rhs.hi, specifier=Comparison.Equal, precision=ML_Bool),
            Comparison(lhs.lo, rhs.lo, specifier=Comparison.Equal, precision=ML_Bool),
            precision=ML_Bool
        )
    elif specifier == Comparison.NotEqual:
        return LogicalOr(
            Comparison(lhs.hi, rhs.hi, specifier=Comparison.NotEqual, precision=ML_Bool),
            Comparison(lhs.lo, rhs.lo, specifier=Comparison.NotEqual, precision=ML_Bool),
            precision=ML_Bool
        )
    elif specifier in [Comparison.Less, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.LessOrEqual]:
        strict_specifier = {
            Comparison.Less: Comparison.Less,
            Comparison.Greater: Comparison.Greater,
            Comparison.LessOrEqual: Comparison.Less,
            Comparison.GreaterOrEqual: Comparison.Greater
        }[specifier]
        return LogicalOr(
            Comparison(lhs.hi, rhs.hi, specifier=strict_specifier, precision=ML_Bool),
            LogicalAnd(
                Comparison(lhs.hi, rhs.hi, specifier=Comparison.Equal, precision=ML_Bool),
                Comparison(lhs.lo, rhs.lo, specifier=specifier, precision=ML_Bool),
                precision=ML_Bool
            ),
            precision=ML_Bool
        )
    else:
        Log.report(Log.Error, "unsupported specifier {} in legalize_mp_2elt_comparison", specifier)

def legalize_mp_3elt_comparison(optree):
    """ Transform comparison on ML_Compound_FP_Format object into
        comparison on sub-fields """
    specifier = optree.specifier
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    # TODO/FIXME: assume than multi-limb operand are normalized
    if specifier == Comparison.Equal:
        return LogicalAnd(
            Comparison(lhs.hi, rhs.hi, specifier=Comparison.Equal, precision=ML_Bool),
            LogicalAnd(
                Comparison(lhs.me, rhs.me, specifier=Comparison.Equal, precision=ML_Bool),
                Comparison(lhs.lo, rhs.lo, specifier=Comparison.Equal, precision=ML_Bool),
                precision=ML_Bool
            ),
            precision=ML_Bool
        )
    elif specifier == Comparison.NotEqual:
        return LogicalOr(
            Comparison(lhs.hi, rhs.hi, specifier=Comparison.NotEqual, precision=ML_Bool),
            LogicalOr(
                Comparison(lhs.me, rhs.me, specifier=Comparison.NotEqual, precision=ML_Bool),
                Comparison(lhs.lo, rhs.lo, specifier=Comparison.NotEqual, precision=ML_Bool),
                precision=ML_Bool
            ),
            precision=ML_Bool
        )
    elif specifier in [Comparison.LessOrEqual, Comparison.GreaterOrEqual, Comparison.Greater, Comparison.Less]:
        strict_specifier = {
            Comparison.Less: Comparison.Less,
            Comparison.Greater: Comparison.Greater,
            Comparison.LessOrEqual: Comparison.Less,
            Comparison.GreaterOrEqual: Comparison.Greater
        }[specifier]
        return LogicalOr(
            Comparison(lhs.hi, rhs.hi, specifier=strict_specifier, precision=ML_Bool),
            LogicalAnd(
                Comparison(lhs.hi, rhs.hi, specifier=Comparison.Equal, precision=ML_Bool),
                LogicalOr(
                    Comparison(lhs.me, rhs.me, specifier=strict_specifier, precision=ML_Bool),
                    LogicalAnd(
                        Comparison(lhs.me, rhs.me, specifier=Comparison.Equal, precision=ML_Bool),
                        Comparison(lhs.lo, rhs.lo, specifier=specifier, precision=ML_Bool),
                        precision=ML_Bool
                    ),
                    precision=ML_Bool
                ),
                precision=ML_Bool
            ),
            precision=ML_Bool
        )
    else:
        Log.report(Log.Error, "unsupported specifier {} in legalize_mp_2elt_comparison", specifier)


def legalize_multi_precision_vector_element_selection(optree):
    """ legalize a VectorElementSelection @p optree on a vector of
        multi-precision elements to a single element """
    assert isinstance(optree, VectorElementSelection)
    multi_precision_vector = optree.get_input(0)
    elt_index = optree.get_input(1)
    hi_vector = multi_precision_vector.hi
    lo_vector = multi_precision_vector.lo
    limb_num = multi_precision_vector.get_precision().get_scalar_format().limb_num 
    if limb_num == 2:
        component_vectors = [hi_vector, lo_vector]
    elif limb_num == 3:
        me_vector = multi_precision_vector.me
        component_vectors = [hi_vector, me_vector, lo_vector]
    else:
        Log.report(Log.Error, "unsupported number of limbs in legalize_multi_precision_vector_element_selection for {}", optree)
    result = BuildFromComponent(
        *tuple(VectorElementSelection(vector, elt_index) for vector in component_vectors)
    )
    forward_attributes(optree, result)
    result.set_precision(optree.precision)
    return result



        
