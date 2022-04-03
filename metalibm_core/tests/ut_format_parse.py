# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2022 Nicolas Brunie
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
# created:              Mar    20th, 2022
#
# author(s):       Nicolas Brunie (nibrunie@gmail.com)
# description:    unit-tests for Metalibm format parsing
###############################################################################
import unittest
import sollya

from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_DoubleDouble, ML_SingleSingle)

class UT_FormatParse(unittest.TestCase):
    """ unit test for metalibm format string parsing functionality """
    def test_parse_format(self):
        # checking string matching
        self.assertTrue(ML_Binary32.matchInStr("0x1.ap-6") is not None)
        self.assertTrue(ML_Binary64.matchInStr("17d3") is None)

        # checking string conversion
        self.assertTrue(ML_Binary32.parseFromStr("0x1.8p1") == 3.0)
        self.assertTrue(ML_DoubleDouble.parseFromStr("{.hi=0x1.0p+2, .lo=0x1p+0}") == 5.0)
        self.assertTrue(ML_SingleSingle.parseFromStr("{.hi=0x1.0p+2, .lo=0x1p+0}") == 5.0)
        self.assertTrue(ML_SingleSingle.parseFromStr("{.hi=0x1.0p+2, .lo=0x1p-53}") != None)
        self.assertTrue(ML_DoubleDouble.parseFromStr("{.hi=0x1.df0b8b5388ea7p-1, .lo=-0x1p-56}") == sollya.parse("0x1.df0b8b5388ea7p-1 + -0x1p-56 "))


if __name__ == '__main__':
    unittest.main()
