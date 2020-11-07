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
# created:              Nov    7th, 2020
# last-modified:        Nov    7th, 2020
#
# author(s):       Nicolas Brunie (nicolas.brunie@kalray.eu)
# desciprition:    unit-tests for Random Generator
###############################################################################
import unittest

from metalibm_core.core.random_gen import UniformInterval
from metalibm_core.utility.ml_template import rng_mode_parser

class UT_StaticVectorizer(unittest.TestCase):
    def test_ref_assign(self):
        """ test behavior of StaticVectorizer on predicated ReferenceAssign """
        rnd_mode_list = rng_mode_list_parser ("UniformInterval(0, 1):UniformInterval(-1, 2)")
        self.assertTrue(isinstance(rnd_mode_list[0], UniformInterval))
        self.assertTrue(isinstance(rnd_mode_list[1], UniformInterval))
        self.assertEqual(rnd_mode_list[0].interval, Interval(0, 1))
        self.assertEqual(rnd_mode_list[1].interval, Interval(-1, 2))


if __name__ == '__main__':
    unittest.main()
