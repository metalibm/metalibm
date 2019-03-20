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
    dirtyinfnorm,
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

    def get_sub_list(self):
        return [self.get_sub_interval(index) for index in range(self.split_num)]

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
        assert exp_bits >= 0 and field_bits >= 0 and (exp_bits + field_bits) > 0
        self.exp_bits = exp_bits
        self.field_bits = field_bits
        self.split_num = 2**(self.exp_bits + self.field_bits)
        self.low_exp_value = low_exp_value
        self.precision = precision

    def get_index_node(self, vx):
        assert vx.precision is self.precision
        int_precision = vx.precision.get_integer_format()
        index_size = self.exp_bits + self.field_bits
        # building an index mask from the index_size
        index_mask   = Constant(2**index_size - 1, precision=int_precision)
        shift_amount = Constant(
            vx.get_precision().get_field_size() - self.field_bits, precision=int_precision
        )
        exp_offset = Constant(
            self.precision.get_integer_coding(S2**self.low_exp_value),
            precision=int_precision
        )
        return BitLogicAnd(
            BitLogicRightShift(
                Subtraction(
                    TypeCast(vx, precision=int_precision),
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
        field_index = index % 2**self.field_bits
        exp_index = int(index / 2**self.field_bits)
        exp_value = exp_index + self.low_exp_value
        lo_bound = (1.0 + field_index * 2**(-self.field_bits)) * S2**exp_value
        return lo_bound

    def get_sub_hi_bound(self, index):
        """ return the upper bound of the sub-interval
            of index @p index """
        assert index >= 0 and index < self.split_num
        field_index = index % 2**self.field_bits
        exp_index = int(index / 2**self.field_bits)
        exp_value = exp_index + self.low_exp_value
        hi_bound = (1.0 + (field_index+1) * 2**(-self.field_bits)) * S2**exp_value
        return hi_bound

    def get_sub_list(self):
        return [self.get_sub_interval(index) for index in range(self.split_num)]
    def get_offseted_sub_list(self):
        return [self.get_offseted_sub_interval(index) for index in range(self.split_num)]

    def get_min_bound(self):
        return self.get_sub_lo_bound(0)
    def get_max_bound(self):
        return self.get_sub_hi_bound(self.split_num - 1)

    def get_offseted_sub_interval(self, index):
        assert index >= 0 and index < self.split_num
        lo_bound = self.get_sub_lo_bound(index)
        hi_bound = self.get_sub_hi_bound(index)
        return lo_bound, Interval(0, hi_bound - lo_bound)

    def get_sub_interval(self, index):
        assert index >= 0 and index < self.split_num
        lo_bound = self.get_sub_lo_bound(index)
        hi_bound = self.get_sub_hi_bound(index)
        return Interval(lo_bound, hi_bound)


def generic_poly_split(offset_fct, indexing, target_eps, coeff_precision, vx):
    """ generate the meta approximation for @p over several intervals
        defined by @p indexing object
        For each sub-interval, a polynomial approximation with
        maximal_error @p target_eps is tabulated """
    poly_degree_list = [int(sup(guessdegree(offset_fct(offset), sub_interval, target_eps))) for offset, sub_interval in indexing.get_offseted_sub_list()]
    poly_max_degree = max(poly_degree_list)

    # tabulating polynomial coefficients on split_num sub-interval of interval
    poly_table = ML_NewTable(dimensions=[indexing.split_num, poly_max_degree+1], storage_precision=coeff_precision)
    offset_table = ML_NewTable(dimensions=[indexing.split_num], storage_precision=coeff_precision)
    max_error = 0.0

    for sub_index in range(indexing.split_num):
        poly_degree = poly_degree_list[sub_index]
        offset, approx_interval = indexing.get_offseted_sub_interval(sub_index)
        offset_table[sub_index] = offset
        if poly_degree == 0:
            # managing constant approximation separately since it seems
            # to break sollya
            local_approx = coeff_precision.round_sollya_object(offset_fct(offset)(inf(approx_interval)))
            poly_table[sub_index][0] = local_approx
            for monomial_index in range(1, poly_max_degree+1):
                poly_table[sub_index][monomial_index] = 0
            approx_error = sollya.infnorm(offset_fct(offset) - local_approx, approx_interval) 

        else:
            poly_object, approx_error = Polynomial.build_from_approximation_with_error(
                offset_fct(offset), poly_degree, [coeff_precision]*(poly_degree+1),
                approx_interval, sollya.relative)


            for monomial_index in range(poly_max_degree+1):
                if monomial_index <= poly_degree:
                    poly_table[sub_index][monomial_index] = poly_object.coeff_map[monomial_index] 
                else:
                    poly_table[sub_index][monomial_index] = 0
        max_error = max(approx_error, max_error)

    Log.report(Log.Debug, "max_error is {}", max_error)

    # indexing function: derive index from input @p vx value
    poly_index = indexing.get_index_node(vx)
    poly_index.set_attributes(tag="poly_index", debug=debug_multi)

    ext_precision = {
        ML_Binary32: ML_SingleSingle,
        ML_Binary64: ML_DoubleDouble,
    }[coeff_precision]

    # building polynomial evaluation scheme
    offset = TableLoad(offset_table, poly_index, precision=coeff_precision, tag="offset", debug=debug_multi) 
    poly = TableLoad(poly_table, poly_index, poly_max_degree, precision=coeff_precision, tag="poly_init", debug=debug_multi)
    red_vx = Subtraction(vx, offset, precision=vx.precision, tag="red_vx", debug=debug_multi)
    for monomial_index in range(poly_max_degree, -1, -1):
        coeff = TableLoad(poly_table, poly_index, monomial_index, precision=coeff_precision, tag="poly_%d" % monomial_index, debug=debug_multi)
        fma_precision = coeff_precision if monomial_index > 1 else ext_precision
        poly = FMA(red_vx, poly, coeff, precision=fma_precision)

    #return Conversion(poly, precision=coeff_precision)
    return poly.hi


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
        abs_vx = Abs(vx, precision=self.precision)

        upper_approx_bound = 10

        eps_exp = -3
        eps = S2**eps_exp

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(0, eps)
        # fonction to approximate is erf(x) / x
        # it is an even function erf(x) / x = erf(-x) / (-x)
        approx_fct = sollya.erf(sollya.x) - (sollya.x)
        poly_degree = int(sup(guessdegree(approx_fct, approx_interval, S2**-(self.precision.get_field_size()+5)))) + 1

        poly_degree_list = list(range(1, poly_degree, 2))
        Log.report(Log.Debug, "poly_degree is {} and list {}", poly_degree, poly_degree_list)
        global_poly_object = Polynomial.build_from_approximation(approx_fct, poly_degree_list, [self.precision]*len(poly_degree_list), approx_interval, sollya.relative)
        Log.report(Log.Debug, "inform is {}", dirtyinfnorm(approx_fct - global_poly_object.get_sollya_object(), approx_interval))
        poly_object = global_poly_object.sub_poly(start_index=1, offset=1)

        ext_precision = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble,
        }[self.precision]

        pre_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, abs_vx, unified_precision=self.precision)
        #poly = FMA(vx, pre_poly, global_poly_object.coeff_map[0], precision=ext_precision, tag="poly", debug=debug_multi)
        #pre_result = Multiplication(vx, poly, precision=ext_precision, tag="pre_result", debug=debug_multi)
        #result = Conversion(pre_result, precision=self.precision)
        result = FMA(pre_poly, abs_vx, abs_vx)
        result.set_attributes(tag="result", debug=debug_multi)

        eps_target = S2**-(self.precision.get_field_size() +5)
        def offset_div_function(fct):
            return lambda offset: fct(sollya.x + offset) 

        near_indexing = SubFPIndexing(2, 6, eps_exp, self.precision)
        near_approx = generic_poly_split(offset_div_function(sollya.erf), near_indexing, eps_target, self.precision, abs_vx)
        near_approx.set_attributes(tag="near_approx", debug=debug_multi)

        def offset_function(fct):
            return lambda offset: fct(sollya.x + offset)
        medium_indexing = SubFPIndexing(2, 7, 1, self.precision)

        medium_approx = generic_poly_split(offset_function(sollya.erf), medium_indexing, eps_target, self.precision, abs_vx)
        medium_approx.set_attributes(tag="medium_approx", debug=debug_multi)

        # approximation for positive values
        scheme = ConditionBlock(
            abs_vx < eps,
            Return(result),
            ConditionBlock(
                abs_vx < near_indexing.get_max_bound(),
                Return(near_approx),
                ConditionBlock(
                    abs_vx < medium_indexing.get_max_bound(),
                    Return(medium_approx),
                    Return(Constant(1.0, precision=self.precision))
                )
            )
        )
        return scheme

    def numeric_emulate(self, input_value):
        return sollya.erf(input_value)

    standard_test_cases = [
        (1.0, None),
        (4.0, None),
        (0.5, None),
        (1.5, None),
        (1024.0, None),
        (sollya.parse("0x1.13b2c6p-2"), None),
        (sollya.parse("0x1.2cb10ap-5"), None),
        (0.0, None),
        (sollya.parse("0x1.07e08ep+1"), None),
    ]



if __name__ == "__main__":
        # auto-test
        arg_template = ML_NewArgTemplate(default_arg=ML_Erf.get_default_args())
        args = arg_template.arg_extraction()

        ml_erf = ML_Erf(args)
        ml_erf.gen_implementation()
