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
#
# created:            Nov   13th, 2020
# last-modified:      Nov   13th, 2020
#
# description: utility to construct advanced indexing from fp numbers
#
###############################################################################
import sollya
from sollya import Interval
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64
from metalibm_core.core.ml_operations import (
    Constant, BitLogicRightShift, BitLogicAnd, Subtraction, TypeCast)

from metalibm_core.utility.log_report import Log

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
    @property
    def interval(self):
        """ return the whole interval covered by the indexing function """
        raise NotImplementedError
    @staticmethod
    def parse(s):
        return eval(s) #, globals + {"SubFPIndexing": SubFPIndexing})

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

class SubFPIndexing(Indexing):
    """ Indexation based on a sub-field of a fp-number
        SubFPIndexing(l, h, f) 
        e bits are extracted from the LSB of exponent
        f bits are extracted from the MSB of mantissa
        exponent is offset by l """
    def __init__(self, low_exp_value, max_exp_value, field_bits, precision):
        self.field_bits = field_bits
        self.low_exp_value = low_exp_value
        self.max_exp_value = max_exp_value
        exp_bits = int(sollya.ceil(sollya.log2(max_exp_value - low_exp_value + 1)))
        assert exp_bits >= 0 and field_bits >= 0 and (exp_bits + field_bits) > 0
        self.exp_bits = exp_bits
        self.split_num = (self.max_exp_value - self.low_exp_value + 1) * 2**(self.field_bits)
        Log.report(Log.Debug, "split_num={}", self.split_num)
        self.precision = precision

    def __repr__(self):
        precision_to_str = {
            ML_Binary32: "ML_Binary32",
            ML_Binary64: "ML_Binary64"
        }
        return "SubFPIndexing(%s, %s, %s, %s)" % (self.low_exp_value, self.max_exp_value, self.field_bits, precision_to_str[self.precision])

    def get_index_node(self, vx):
        """ generation an operation sub-graph to compute the
            indexing from input vx

            :param vx: input operand
            :type vx: ML_Operation

        """
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

    @property
    def min_bound(self):
        return self.get_sub_lo_bound(0)
    @property
    def max_bound(self):
        return self.get_sub_hi_bound(self.split_num - 1)

    def get_offseted_sub_interval(self, index):
        """ return a pair (offset, [0; size]) """
        assert index >= 0 and index < self.split_num
        lo_bound = self.get_sub_lo_bound(index)
        hi_bound = self.get_sub_hi_bound(index)
        return lo_bound, Interval(0, hi_bound - lo_bound)

    def get_sub_interval(self, index):
        assert index >= 0 and index < self.split_num
        lo_bound = self.get_sub_lo_bound(index)
        hi_bound = self.get_sub_hi_bound(index)
        return Interval(lo_bound, hi_bound)
    @property
    def interval(self):
        return Interval(self.min_bound, self.max_bound)


class SubIntervalIndexing(Indexing):
    def __init__(self, interval, split_num):
        # overall interval
        self.interval = interval
        # number of sub-intervals
        self.split_num = split_num

    @property
    def bound_low(self):
        return inf(self.interval)
    @property
    def bound_high(self):
        return sup(self.interval)
        
    def get_index_node(self, vx):
        """ return the meta graph to implement index calculation
            from input @p vx """
        precision = vx.get_precision()
        bound_low = inf(self.interval)
        bound_high = sup(self.interval)
        num_intervals = self.split_num 

        int_prec = precision.get_integer_format()

        diff = Subtraction(
            vx,
            Constant(bound_low, precision=precision),
            tag="diff",
            precision=precision
        )

        # delta = bound_high - bound_low
        delta_ratio = Constant(num_intervals / (bound_high - bound_low), precision=precision)
        # computing table index
        # index = nearestint(diff / delta * <num_intervals>)
        index = Max(0,
            Min(
                NearestInteger(
                    Multiplication(
                        diff,
                        delta_ratio,
                        precision=precision
                    ),
                    precision=int_prec,
                ),
                num_intervals - 1
            ),
            tag="index",
            precision=int_prec
        )
        return index
    def get_sub_interval(self, index):
        """ return the sub-interval numbered @p index """
        subint_low = self.bound_low + i * interval_size
        subint_high = self.bound_low + (i+1) * interval_size
        return Interval(subint_low, subint_high)

    def get_sub_list(self):
        return [self.get_sub_interval(index) for index in range(self.split_num)]
