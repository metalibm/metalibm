# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Kalray
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
# created:              Jan   20th, 2021
# last-modified:        Jan   20th, 2021
#
# author(s):       Nicolas Brunie (nicolas.brunie@kalray.eu)
# desciprition:    unit-tests for Random Generator
###############################################################################
import unittest

import sollya

from metalibm_core.utility.axf_utils import (
    AbsoluteApproxError, RelativeApproxError)

class UT_ApproxError(unittest.TestCase):
    ErrorClass = AbsoluteApproxError

    def test_basic_comp(self):
        """ test basic comparison operations on AbsoluteApproxError
            objects """
        a0 = self.ErrorClass(0.0)
        a1 = self.ErrorClass(1.0)
        a2 = self.ErrorClass(2.0)

        self.assertTrue(a0 < a1)
        self.assertTrue(a0 < a2)
        self.assertTrue(a1 < a2)
        self.assertTrue(a1 > a0)
        self.assertTrue(a2 > a0)
        self.assertTrue(a2 > a1)
        self.assertTrue(a1 == a1)

    def test_min_max(self):
        """ test min/max on self.ErrorClass objects """
        a0 = self.ErrorClass(0.0)
        a1 = self.ErrorClass(sollya.parse("[2.0036359077330092187553823271953202730943921494599e-14;2.00363602342695305403488761992545079477728128673956e-14]"))
        a2 = self.ErrorClass(sollya.parse("[2.7402990785775088478657193769016942042071605101228e-9;2.7402992368078572186539178700197208406596124738864e-9]"))

        self.assertTrue(max([a0, a1, a2]) == a2)
        self.assertTrue(min([a0, a1, a2]) == a0)

class UT_RelativeApproxError(UT_ApproxError):
    ErrorClass = RelativeApproxError


if __name__ == '__main__':
    unittest.main()
