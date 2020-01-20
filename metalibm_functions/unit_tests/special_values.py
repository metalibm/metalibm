# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          Jan 19th, 2020
# last-modified:    Jan 19th, 2020
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
# Instances (see valid/unit_test.py
#

import sys
import collections
import operator

import sollya
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.special_values import (
    Unordered, FP_PlusOmega, FP_MinusOmega, FP_SNaN, FP_QNaN,
    FP_PlusZero, FP_MinusZero, FP_PlusInfty, FP_MinusInfty
)
from metalibm_core.core.ml_formats import ML_Binary64, ML_Binary32

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)

from metalibm_functions.unit_tests.utils import TestRunner

op_map = collections.OrderedDict([
    ("+", operator.__add__),
    ("-", operator.__sub__),
    ("*", operator.__mul__),
    ("<", operator.__lt__),
    (">", operator.__gt__),
    ("<=", operator.__le__),
    ("==", operator.__eq__),
    ("!=", operator.__ne__),
])

class ML_UT_SpecialValue(TestRunner):
    def __init__(self, args=None):
        pass

    @staticmethod
    def __call__(args):
        for PRECISION in [ML_Binary32, ML_Binary64]:
            TEST_CASE = [
                (FP_PlusOmega(PRECISION), "<", FP_SNaN(PRECISION), Unordered),
                (FP_PlusOmega(PRECISION), "<=", FP_SNaN(PRECISION), Unordered),
                (FP_PlusInfty(PRECISION), "==", FP_PlusInfty(PRECISION), True),
                (FP_PlusInfty(PRECISION), "==", FP_QNaN(PRECISION), False),
                (FP_PlusZero(PRECISION), "==", FP_MinusZero(PRECISION), True),
                (FP_PlusZero(PRECISION), "!=", FP_MinusZero(PRECISION), False),
                (FP_QNaN(PRECISION), "==", FP_QNaN(PRECISION), False),
                (FP_PlusInfty(PRECISION), "==", -FP_MinusInfty(PRECISION), True),
                (FP_MinusInfty(PRECISION), ">", 0, False),
                (FP_MinusInfty(PRECISION), ">", FP_PlusZero(PRECISION), False),
                (FP_MinusInfty(PRECISION), ">", FP_MinusZero(PRECISION), False),
            ]
            for lhs, op, rhs, expected in TEST_CASE:
                result = op_map[op](lhs, rhs)
                print( "{} {} {} = ".format(lhs, op, rhs))
                print("{} vs expected {} ".format(result, expected))
                assert result == expected, "test failure"
        print("Test success")

# principal test runner
run_test = ML_UT_SpecialValue

if __name__ == "__main__":
    args = {}

    if run_test.__call__(args):
        exit(0)
    else:
        exit(1)


