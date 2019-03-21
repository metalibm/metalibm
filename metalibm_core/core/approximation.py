# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2019 Kalray
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
# created:            Mar   21st, 2019
# last-modified:      Mar   21st, 2019
#
# description: toolbox for approximation construct
#
###############################################################################

import sollya
from sollya import Interval, sup, inf, guessdegree

S2 = sollya.SollyaObject(2)


from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.utility.num_utils import fp_next
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.ml_operations import (
    Constant, FMA, TableLoad, BitLogicAnd, Subtraction, BitLogicRightShift, 
    TypeCast,
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_SingleSingle, ML_DoubleDouble
)

from metalibm_core.utility.debug_utils import debug_multi
from metalibm_core.utility.log_report import Log

def get_extended_fp_precision(precision):
    """ return the extended counterart of @p precision """
    ext_precision = {
        ML_Binary32: ML_SingleSingle,
        ML_Binary64: ML_DoubleDouble,
    }[precision]
    return ext_precision


class Indexing:
    """ generic class for expressing value indexing (in a table) """
    def get_index_node(self, vx):
        """ return the meta graph to implement index calculation
            from input @p vx """
        raise NotImplementedError
    def get_sub_interval(self, index):
        """ return the sub-interval numbered @p index """
        raise NotImplementedError
    def get_sub_list(self):
        """ return the list of sub-intervals ordered by index """
        raise NotImplementedError

class SubMantissaIndexing(Indexing):
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
    # computing degree for a different polynomial approximation on each
    # sub-interval
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

    Log.report(Log.Debug, "max approx error is {}", max_error)

    # indexing function: derive index from input @p vx value
    poly_index = indexing.get_index_node(vx)
    poly_index.set_attributes(tag="poly_index", debug=debug_multi)

    ext_precision = get_extended_fp_precision(coeff_precision)

    # building polynomial evaluation scheme
    offset = TableLoad(offset_table, poly_index, precision=coeff_precision, tag="offset", debug=debug_multi) 
    poly = TableLoad(poly_table, poly_index, poly_max_degree, precision=coeff_precision, tag="poly_init", debug=debug_multi)
    red_vx = Subtraction(vx, offset, precision=vx.precision, tag="red_vx", debug=debug_multi)
    for monomial_index in range(poly_max_degree, -1, -1):
        coeff = TableLoad(poly_table, poly_index, monomial_index, precision=coeff_precision, tag="poly_%d" % monomial_index, debug=debug_multi)
        #fma_precision = coeff_precision if monomial_index > 1 else ext_precision
        fma_precision = coeff_precision
        poly = FMA(red_vx, poly, coeff, precision=fma_precision)

    #return Conversion(poly, precision=coeff_precision)
    #return poly.hi
    return poly

def search_bound_threshold(fct, limit, start_point, end_point, precision):
    """ search by dichotomy the minimal x, floating-point number in
        @p precision, such that x >= start_point and x <= end_point
        and round(fct(x)) = limit """
    assert precision.round_sollya_object(fct(start_point)) < limit
    assert precision.round_sollya_object(fct(end_point)) >= limit
    assert start_point < end_point
    left_bound = start_point
    right_bound = end_point
    while left_bound != right_bound and fp_next(left_bound, precision) != right_bound:
        print(left_bound, right_bound, fp_next(left_bound, precision))
        mid_point = precision.round_sollya_object((left_bound + right_bound) / S2, round_mode=sollya.RU)
        mid_point_img = precision.round_sollya_object(fct(mid_point), round_mode=sollya.RU)
        if mid_point_img >= limit:
            right_bound = mid_point
        elif mid_point_img < limit:
            left_bound = mid_point
        else:
            Log.report(Log.Error, "function must be increasing in search_bound_threshold")
    return left_bound

