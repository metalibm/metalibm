# -*- coding: utf-8 -*-
""" Utilities to manipulate, adapt and check fixed-point precisions """

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
# Created:          Jul  8th, 2017
# Last-modified:    Mar  7th, 2018
#
# Author(s): Nicolas Brunie (nbrunie@kalray.eu)
# Description: Utilities for fixed-point node manipulation
###############################################################################

# python3 legacy compatibility
from functools import reduce

from metalibm_core.utility.log_report import Log
from metalibm_core.core.ml_hdl_format import is_fixed_point, fixed_point
from metalibm_core.core.ml_formats import ML_Integer

## test equality between @p unified_format and all
#  the format in @p format_list
def test_format_equality_list(unified_format, format_list):
    for precision in format_list:
        if not test_format_equality(unified_format, precision):
            return False
    return True

## Test equality between two formats/precisions
def test_format_equality(format0, format1):
    if format0 == format1:
        return True
    elif is_fixed_point(format0) and is_fixed_point(format1):
        return \
            (format0.get_integer_size() == format1.get_integer_size()) and \
            (format0.get_frac_size() == format1.get_frac_size()) and \
            (format0.get_signed() == format1.get_signed())
    return False

## generate the minimal format which fits both @p format0 and @p format1
def largest_format(format0, format1):
    if is_fixed_point(format0) and is_fixed_point(format1):
        #
        unsigned_int_size = format0.get_integer_size() if format1.get_signed() \
                            else format1.get_integer_size()
        signed_int_size = format1.get_integer_size() if format1.get_signed() \
                            else format0.get_integer_size()
        xor_signed = (format0.get_signed() != format1.get_signed()) and \
                    (unsigned_int_size >= signed_int_size)
        int_size = max(
            format0.get_integer_size(),
            format1.get_integer_size()
        ) + (1 if xor_signed else 0)
        frac_size = max(
            format0.get_frac_size(),
            format1.get_frac_size()
        )
        return fixed_point(
            int_size,
            frac_size,
            signed = format0.get_signed() or format1.get_signed()
        )
    elif format0 is None or format0 is ML_Integer: # TODO: fix for abstract format
        return format1
    elif format1 is None or format1 is ML_Integer: # TODO: fix for abstract format
        return format0
    elif format0.get_bit_size() == format1.get_bit_size():
        return format0
    else:
        Log.report(
            Log.Error,
            "unable to determine largest format between {} and {}".format(
                format0, format1
            ), error=NotImplementedError)

## determine if there is a common format
#  to unify format_list
def solve_equal_formats(optree_list):
    precision_list = [op.get_precision() for op in optree_list]

    format_reduced = reduce(
        lambda f0, f1: largest_format(f0, f1),
        precision_list,
        precision_list[0]
    )

    if format_reduced is None:
        Log.report(
            Log.Info,
            " precision of every item in list: {} is None:\n\t".format(
                ", \n\t".join(
                    op.get_str(display_precision = True) for op in optree_list
                )
            )
        )
        return None
    else:
        # test_format_equality_list(format_reduced, precision_list)
        return format_reduced
