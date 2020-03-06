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
# created:              Mar    6th, 2020
# last-modified:        Mar    6th, 2020
#
# author(s):       Nicolas Brunie (nicolas.brunie@kalray.eu)
# desciprition:    unit-tests for Static Vectorizer
###############################################################################
import unittest

from metalibm_core.core.ml_vectorizer import (
    StaticVectorizer, fallback_policy, instanciate_variable)
from metalibm_core.core.ml_operations import (
    Variable, Statement, ReferenceAssign, Constant, ConditionBlock,
    Return)

class UT_StaticVectorizer(unittest.TestCase):
    def test_ref_assign(self):
        """ test behavior of StaticVectorizer on predicated ReferenceAssign """
        va = Variable("a")
        vb = Variable("b")
        vc = Variable("c")
        scheme = Statement(
            ReferenceAssign(va, Constant(3)),
            ConditionBlock(
                (va > vb).modify_attributes(likely=True),
                Statement(
                    ReferenceAssign(vb, va),
                    ReferenceAssign(va, Constant(11)),
                    Return(va)
                ),
            ),
            ReferenceAssign(va, Constant(7)),
            Return(vb)
        )
        vectorized_path = StaticVectorizer().extract_vectorizable_path(scheme, fallback_policy)

        linearized_most_likely_path = instanciate_variable(vectorized_path.linearized_optree, vectorized_path.variable_mapping)
        test_result = (isinstance(linearized_most_likely_path, Constant) and linearized_most_likely_path.get_value() == 11)
        if not test_result:
            print("test UT_StaticVectorizer failure")
            print("scheme: {}".format(scheme.get_str()))
            print("linearized_most_likely_path: {}".format(linearized_most_likely_path))
        self.assertTrue(test_result)


if __name__ == '__main__':
    unittest.main()
