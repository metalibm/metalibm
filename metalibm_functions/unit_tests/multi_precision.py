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
# created:              Oct    8th, 2018
# last-modified:        Oct   20th, 2018
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
# description: unit test for multi-precision format expansion
###############################################################################


import sys

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import (
    Return, Statement, Conversion,
    Addition, Multiplication, Subtraction,
    FMA,
    SpecificOperation,
    Constant,
)
from metalibm_core.core.ml_formats import (
    ML_Int32,
    ML_Binary32, ML_Binary64,
    ML_SingleSingle, ML_DoubleDouble
)

from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_functions.unit_tests.utils import TestRunner


from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)


class ML_UT_MultiPrecision(ML_FunctionBasis, TestRunner):
    function_name = "ml_ut_multi_precision"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
                builtin from a default argument mapping overloaded with @p kw """
        default_args = {
                "output_file": "ut_multi_precision.c",
                "function_name": "ut_multi_precision",
                "precision": ML_Binary32,
                "target": GenericProcessor(),
                "language": C_Code,
                "arity": 2,
                "input_precisions": [ML_Binary32, ML_Binary32],
                "fast_path_extract": True,
                "fuse_fma": False,
                "libm_compliant": True
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)


    def generate_scheme(self):
        # declaring function input variable
        v_x = [self.implementation.add_input_variable("x%d" % index, self.get_input_precision(index)) for index in range(self.arity)]

        double_format = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble
        }[self.precision]

        # testing Add211
        exact_add = Addition(v_x[0], v_x[1], precision=double_format, tag="exact_add")
        # testing Mul211
        exact_mul = Multiplication(v_x[0], v_x[1], precision=double_format, tag="exact_mul")
        # testing Sub211
        exact_sub = Subtraction(v_x[1], v_x[0], precision=double_format, tag="exact_sub")
        # testing Add222
        multi_add = Addition(exact_add, exact_sub, precision=double_format, tag="multi_add")
        # testing Mul222
        multi_mul = Multiplication(multi_add, exact_mul, precision=double_format, tag="multi_mul")
        # testing Add221 and Add212 and Sub222
        multi_sub = Subtraction(
            Addition(exact_sub, v_x[1], precision=double_format, tag="add221"),
            Addition(v_x[0], multi_mul, precision=double_format, tag="add212"),
            precision=double_format,
            tag="sub222"
        )
        # testing Mul212 and Mul221
        mul212 = Multiplication(multi_sub, v_x[0], precision=double_format, tag="mul212")
        mul221 = Multiplication(exact_mul, v_x[1], precision=double_format, tag="mul221")
        # testing Sub221 and Sub212
        sub221 = Subtraction(mul212, mul221.hi, precision=double_format, tag="sub221")
        sub212 = Subtraction(sub221, mul212.lo, precision=double_format, tag="sub212")
        # testing FMA2111
        fma2111 = FMA(sub221.lo, sub212.hi, mul221.hi, precision=double_format, tag="fma2111")
        # testing FMA2112
        fma2112 = FMA(fma2111.lo, fma2111.hi, fma2111, precision=double_format, tag="fma2112")
        # testing FMA2212
        fma2212 = FMA(fma2112, fma2112.hi, fma2112, precision=double_format, tag="fma2212")
        # testing FMA2122
        fma2122 = FMA(fma2212.lo, fma2212, fma2212, precision=double_format, tag="fma2122")
        # testing FMA22222
        fma2222 = FMA(fma2122, fma2212, fma2111, precision=double_format, tag="fma2222")
        # testing Add122
        add122 = Addition(fma2222, fma2222, precision=self.precision, tag="add122")
        # testing Add112
        add112 = Addition(add122, fma2222, precision=self.precision, tag="add112")
        # testing Add121
        add121 = Addition(fma2222, add112, precision=self.precision, tag="add121")
        # testing subnormalization
        multi_subnormalize = SpecificOperation(
            Addition(add121, add112, precision=double_format),
            Constant(3, precision=self.precision.get_integer_format()),
            specifier=SpecificOperation.Subnormalize,
            precision=double_format,
            tag="multi_subnormalize")
        result = Conversion(multi_subnormalize, precision=self.precision)

        scheme = Statement(
            Return(result)
        )

        return scheme

    def numeric_emulate(self, *args):
        acc = 0.0
        for i in range(self.arity):
            acc += args[i]
        return acc


    @staticmethod
    def __call__(args):
        ml_ut_llvm_code = ML_UT_MultiPrecision(args)
        ml_ut_llvm_code.gen_implementation()
        return True


run_test = ML_UT_MultiPrecision

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MultiPrecision.get_default_args())
    args = arg_template.arg_extraction()

    if ML_UT_MultiPrecision.__call__(args):
        exit(0)
    else:
        exit(1)



