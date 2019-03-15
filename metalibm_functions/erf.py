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
#
# created:            Mar   11th, 2016
# last-modified:      Mar    3rd, 2019
#
# description: meta-implementation of error-function erf
#              erf(x) = 2 / pi * integral(0, x, e^(-t^2), dt)
###############################################################################

import sollya

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, log1p,
    guessdegree
)

from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.ml_table import ML_NewTable, generic_mantissa_msb_index_fct
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.special_values import (
        FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.debug_utils import debug_multi


# static constant for numerical value 2
S2 = sollya.SollyaObject(2)


class Indexing:
    pass

class SubMantissaIndexing:
    """ Indexing class using upper bits of the floating-point mantissa
        to build an index """
    def __init__(self, field_bit_num):
        self.field_bit_num = field_bit_num
        self.split_num = 2**self.field_bit_num

    def get_index_node(self, vx):
        return generic_mantissa_msb_index_fct(self.field_bit_num, vx)

    def get_sub_interval(self, index):
        assert index >= 0 and index < self.split_num
        lo_bound = 1.0 + index * 2**(-self.field_bit_num)
        hi_bound = 1.0 + (index+1) * 2**(-self.field_bit_num)
        return Interval(lo_bound, hi_bound)

class SubFPIndexing:
    """ Indexation based on a sub-field of a fp-number
        e bits are extracted from the LSB of exponent
        f bits are extract from the MSB of mantissa
        exponent is offset by l """
    def __init__(self, exp_bits, field_bits, low_exp_value, precision):
        assert hi_index >= lo_index
        self.exp_bits = exp_bits
        self.field_bits = field_bits
        self.split_num = 2**(self.exp_bits + self.field_bits)
        self.low_exp_value = low_exp_value
        self.precision

    def get_index_node(self, vx):
        assert vx.precision is self.precision
        int_precision = vx.precision.get_integer_format()
        index_size = self.exp_bits + self.field_bits
        # building an index mask from the index_size
        index_mask   = Constant(2**index_size - 1, precision=int_precision)
        shift_amount = Constant(
            variable.get_precision().get_field_size() - self.field_bits, precision=int_precision
        )
        exp_offset = Constant(
            self.precision.get_integer_coding(S2**self.low_exp_value),
            precision=int_precision
        )
        return BitLogicAnd(
            BitLogicRightShift(
                Subtraction(
                    TypeCast(variable, precision=int_precision),
                    exp_offset,
                    precision=int_precision
                ),
                shift_amount, precision=int_precision
            ),
            index_mask, precision=int_precision)

    def get_sub_lo_bound(self, index):
        """ return the lower bound of the sub-interval
            of index @p index """
        assert index >= 0 and index < self.split_num
        field_index = index % self.field_bits
        exp_index = int(index / self.field_bits)
        exp_value = exp_index + self.low_exp_value
        lo_bound = (1.0 + field_index * 2**(-self.field_bits)) * S2**exp_value
        return lo_bound

    def get_sub_hi_bound(self, index):
        """ return the upper bound of the sub-interval
            of index @p index """
        assert index >= 0 and index < self.split_num
        field_index = index % self.field_bits
        exp_index = int(index / self.field_bits)
        exp_value = exp_index + self.low_exp_value
        hi_bound = (1.0 + (field_index+1) * 2**(-self.field_bits)) * S2**exp_value
        return hi_bound

    def get_min_bound(self):
        return self.get_sub_lo_bound(0)
    def get_max_bound(self):
        return self.get_sub_hi_bound(self.split_num - 1)

    def get_sub_interval(self, index):
        assert index >= 0 and index < self.split_num
        lo_bound = self.get_sub_lo_bound(index)
        hi_bound = self.get_sub_hi_bound(index)
        return Interval(lo_bound, hi_bound)


def uniform_poly_split(fct, interval, split_num, target_eps, coeff_precision, vx):
    """ generate the meta approximation for @p fct over
        @p interval by splitting the full range into @p split_num
        sub-interval and computing a polynomial approximation with
        maximal_error @p target_eps on each of them """
    low_bound = inf(interval)
    step_size = (sup(interval) - low_bound) / step_size

    def sub_interval(index):
        assert index < split_num and index >= 0
        sub_low_bound = inf(interval) + index * step_size
        # computing upper bound to make sure last interval covers the upper
        # bound of the global interval
        sub_upper_bound = sub_low_bound + step_size if index != split_num else sup(interval)
        return Interval(sub_low_bound, sub_upper_bound)


    poly_max_degree = max(guessdegree(approx_fct, sub_interval(index), target_eps) for index in range(split_num)) 

    # tabulating polynomial coefficients on split_num sub-interval of interval
    poly_table = ML_NewTable(dimensions=[split_num, poly_max_degree+1], storage_precision=coeff_precision)

    for sub_index in range(split_num):
        poly_degree = poly_max_degree
        approx_interval = sub_interval(sub_index)
        poly_object = Polynomial.build_from_approximation(
            approx_fct, poly_degree, [self.precision]*(poly_degree+1),
            approx_interval, sollya.absolute)

        for monomial_index in range(poly_max_degree+1):
            poly_table[sub_index][monomial_index] = poly_object.coeff_map[monomial_index] 

    # indexing function: derive index from input @p vx value
    poly_index = (vx - low_bound) / step_size

    # building polynomial evaluation scheme
    poly = TableLoad(poly_table, poly_index, poly_max_degree, precision=self.precision)
    for monomial_index in range(poly_max_degree, -1, -1):
        coeff = TableLoad(poly_table, poly_index, monomial_index, precision=self.precision)
        poly = FMA(vx, poly, coeff)

    return poly


class ML_Erf(ML_FunctionBasis):
    function_name = "ml_erf"
    def __init__(self, args):
        ML_FunctionBasis.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Erf,
                builtin from a default argument mapping overloaded with @p kw """
        default_args_erf = {
                "output_file": "my_erf.c",
                "function_name": "my_erf",
                "precision": ML_Binary32,
                "accuracy": ML_Faithful,
                "target": GenericProcessor(),
                "passes": [("start:instantiate_abstract_prec"), ("start:instantiate_prec"), ("start:basic_legalization"), ("start:expand_multi_precision")],
        }
        default_args_erf.update(kw)
        return DefaultArgTemplate(**default_args_erf)

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("x", self.precision)

        upper_approx_bound = 10

        eps = 1

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(0, eps)
        approx_fct = sollya.erf(sollya.x) / (sollya.x)
        poly_degree = sup(guessdegree(approx_fct, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        Log.report(Log.Debug, "poly_degree is {}", poly_degree)
        global_poly_object = Polynomial.build_from_approximation(approx_fct, poly_degree, [self.precision]*(poly_degree+1), approx_interval, sollya.absolute)
        poly_object = global_poly_object # .sub_poly(start_index=1)

        poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, vx, unified_precision=self.precision)
        result = vx * poly

        scheme = Return(vx * poly)
        return scheme

    def numeric_emulate(self, input_value):
        return sollya.erf(input_value)

    standard_test_cases = [
        (4.0, None),
        (0.5, None),
        (1.0, None),
        (1.5, None),
        (1024.0, None),
        (sollya.parse("0x1.13b2c6p-2"), None),
        (sollya.parse("0x1.2cb10ap-5"), None),
        (0.0, None),
    ]



if __name__ == "__main__":
        # auto-test
        arg_template = ML_NewArgTemplate(default_arg=ML_Erf.get_default_args())
        args = arg_template.arg_extraction()

        ml_erf = ML_Erf(args)
        ml_erf.gen_implementation()
