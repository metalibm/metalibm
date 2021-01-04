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
from metalibm_core.core.polynomials import (
    Polynomial, PolynomialSchemeEvaluator, SollyaError)
from metalibm_core.core.ml_operations import (
    Constant, FMA, TableLoad, BitLogicAnd, BitLogicRightShift,
    Multiplication, Subtraction,
    TypeCast, Conversion,
    Max, Min,
    NearestInteger,
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_SingleSingle, ML_DoubleDouble)

from metalibm_core.utility.debug_utils import debug_multi
from metalibm_core.utility.log_report import Log
from metalibm_core.utility.axf_utils import (
    AXF_SimplePolyApprox, AXF_UniformPiecewiseApprox,
    AbsoluteApproxError, AXF_ApproxError,
    RelativeApproxError,
    AXF_Polynomial,
    AXF_GenericPolynomialSplit)

from metalibm_core.utility.ml_template import precision_parser

def get_extended_fp_precision(precision):
    """ return the extended counterart of @p precision """
    ext_precision = {
        ML_Binary32: ML_SingleSingle,
        ML_Binary64: ML_DoubleDouble,
    }[precision]
    return ext_precision


def generic_poly_split_param_from_axf(axf_approx, indexing):
    """ load paramater for a generic polynomial split from an AXF structure """
    # indexing = eval(axf_approx.indexing)
    max_degree = axf_approx.max_degree
    coeff_precision = axf_approx.precision

    poly_table = ML_NewTable(dimensions=[indexing.split_num, max_degree+1], storage_precision=coeff_precision, const=True)
    offset_table = ML_NewTable(dimensions=[indexing.split_num], storage_precision=coeff_precision, const=True)
    # TODO/FIXME: means to select and/or compare between relative and absolute errors
    max_error = RelativeApproxError(0.0)

    for sub_index in range(indexing.split_num):
        offset, approx_interval = indexing.get_offseted_sub_interval(sub_index)
        offset_table[sub_index] = offset

        local_approx = axf_approx.approx_list[sub_index]

        poly_object = local_approx.poly
        approx_error = local_approx.approx_error

        for monomial_index in range(max_degree+1):
            if monomial_index in poly_object.coeff_map:
                poly_table[sub_index][monomial_index] = poly_object.coeff_map[monomial_index] 
            else:
                poly_table[sub_index][monomial_index] = 0

                # TODO/FIXME: must implement proper error storage/comparaison
                #             mechanism to process cases when we want to
                #             compare a relative to an absolute error
                if approx_error.error_type != max_error.error_type:
                    Log.report(Log.Warning, "comparing two errors of different types")
                max_error = approx_error if approx_error.value > max_error.value else max_error# max(approx_error, max_error)

    return offset_table, max_degree, poly_table, max_error


def generic_poly_split_paramgen(offset_fct, indexing, target_eps, coeff_precision, axf_export=False):
    # computing degree for a different polynomial approximation on each
    # sub-interval
    poly_degree_list = [int(sup(guessdegree(offset_fct(offset), sub_interval, target_eps))) for offset, sub_interval in indexing.get_offseted_sub_list()]
    max_degree = max(poly_degree_list)

    # tabulating polynomial coefficients on split_num sub-interval of interval
    poly_table = ML_NewTable(dimensions=[indexing.split_num, max_degree+1], storage_precision=coeff_precision, const=True)
    offset_table = ML_NewTable(dimensions=[indexing.split_num], storage_precision=coeff_precision, const=True)
    max_error = 0.0

    # object for AXF export
    if axf_export:
        # TODO/FIXME/ using offset_fct evaluation at 0 to provide a dumpable
        #             function. We may prefer an non-evaluated offset_fct
        #             transcription
        axf_error = AXF_ApproxError.from_AE(AbsoluteApproxError(target_eps))
        axf_approx = AXF_GenericPolynomialSplit(offset_fct(0), coeff_precision, indexing.interval, indexing, max_degree, axf_error)
    else:
        axf_approx = None

    for sub_index in range(indexing.split_num):
        poly_degree = poly_degree_list[sub_index]
        offset, approx_interval = indexing.get_offseted_sub_interval(sub_index)
        offset_table[sub_index] = offset
        if poly_degree == 0:
            # managing constant approximation separately since it seems
            # to break sollya
            local_approx = coeff_precision.round_sollya_object(offset_fct(offset)(inf(approx_interval)))
            poly_table[sub_index][0] = local_approx
            for monomial_index in range(1, max_degree+1):
                poly_table[sub_index][monomial_index] = 0
            approx_error = sollya.infnorm(offset_fct(offset) - local_approx, approx_interval) 

            if axf_export:
                axf_poly = AXF_Polynomial.from_poly(Polynomial({0: local_approx}))
                axf_error = AXF_ApproxError.from_AE(AbsoluteApproxError(approx_error))
                axf_approx.approx_list.append(
                    AXF_SimplePolyApprox(axf_poly,
                                         offset_fct(offset), [0], [coeff_precision],
                                         approx_interval,
                                         approx_error=axf_error)) 

        else:
            poly_object, approx_error = Polynomial.build_from_approximation_with_error(
                offset_fct(offset), poly_degree, [coeff_precision]*(poly_degree+1),
                approx_interval, sollya.relative)

            for monomial_index in range(max_degree+1):
                if monomial_index <= poly_degree:
                    poly_table[sub_index][monomial_index] = poly_object.coeff_map[monomial_index] 
                else:
                    poly_table[sub_index][monomial_index] = 0

            if axf_export:
                axf_poly = AXF_Polynomial.from_poly(poly_object)
                axf_error = AXF_ApproxError.from_AE(RelativeApproxError(approx_error))
                axf_approx.approx_list.append(
                    AXF_SimplePolyApprox(axf_poly,
                                         offset_fct(offset), list(range(poly_degree+1)),
                                         [coeff_precision]*(poly_degree+1),
                                         approx_interval,
                                         approx_error=axf_error)) 
        max_error = max(approx_error, max_error)

    return offset_table, max_degree, poly_table, max_error, axf_approx


def generic_poly_split_from_params(offset_table, max_degree, poly_table, indexing, coeff_precision, vx):
    # indexing function: derive index from input @p vx value
    poly_index = indexing.get_index_node(vx)
    poly_index.set_attributes(tag="poly_index", debug=debug_multi)

    ext_precision = get_extended_fp_precision(coeff_precision)

    # building polynomial evaluation scheme
    offset = TableLoad(offset_table, poly_index, precision=coeff_precision, tag="offset", debug=debug_multi) 
    poly = TableLoad(poly_table, poly_index, max_degree, precision=coeff_precision, tag="poly_init", debug=debug_multi)
    red_vx = Subtraction(vx, offset, precision=vx.precision, tag="red_vx", debug=debug_multi)
    for monomial_index in range(max_degree, -1, -1):
        coeff = TableLoad(poly_table, poly_index, monomial_index, precision=coeff_precision, tag="poly_%d" % monomial_index, debug=debug_multi)
        #fma_precision = coeff_precision if monomial_index > 1 else ext_precision
        fma_precision = coeff_precision
        # TODO/FIXME: only using Horner evaluation scheme
        poly = FMA(red_vx, poly, coeff, precision=fma_precision)

    #return Conversion(poly, precision=coeff_precision)
    #return poly.hi
    return poly

def generic_poly_split(offset_fct, indexing, target_eps, coeff_precision, vx, axf_export=False):
    """ generate the meta approximation for @p offset_fct over several
        intervals defined by @p indexing object
        For each sub-interval, a polynomial approximation with
        maximal_error @p target_eps is tabulated, and evaluated using format
        @p coeff_precision.
        The input variable is @p vx """

    offset_table, max_degree, poly_table, max_error, axf_approx = generic_poly_split_paramgen(offset_fct, indexing,
                                                                    target_eps, coeff_precision,
                                                                    axf_export=axf_export)
    Log.report(Log.Debug, "max approx error is {}", max_error)

    poly = generic_poly_split_from_params(offset_table, max_degree, poly_table, indexing, coeff_precision, vx)

    return poly, axf_approx

def search_bound_threshold(fct, limit, start_point, end_point, precision):
    """ This function assume that <fct> is monotonic and increasing
        search by dichotomy the minimal x, floating-point number in
        @p precision, such that x >= start_point and x <= end_point
        and round(fct(x)) = limit.  """
    assert precision.round_sollya_object(fct(start_point)) < limit
    assert precision.round_sollya_object(fct(end_point)) >= limit
    assert start_point < end_point
    left_bound = start_point
    right_bound = end_point
    while left_bound != right_bound and fp_next(left_bound, precision) != right_bound:
        mid_point = precision.round_sollya_object((left_bound + right_bound) / S2, round_mode=sollya.RU)
        mid_point_img = precision.round_sollya_object(fct(mid_point), round_mode=sollya.RU)
        if mid_point_img >= limit:
            right_bound = mid_point
        elif mid_point_img < limit:
            left_bound = mid_point
        else:
            Log.report(Log.Error, "function must be increasing in search_bound_threshold")
    return left_bound


def search_bound_threshold_mirror(fct, limit, start_point, end_point, precision):
    """ This function assume that <fct> is monotonic and decreasing
        search by dichotomy the maximal x, floating-point number in
        @p precision, such that x >= start_point and x <= end_point
        and round(fct(x)) >= limit.  """
    assert precision.round_sollya_object(fct(start_point)) >= limit
    assert precision.round_sollya_object(fct(end_point)) < limit
    assert start_point < end_point
    left_bound = start_point
    right_bound = end_point
    while left_bound != right_bound and fp_next(left_bound, precision) != right_bound:
        mid_point = precision.round_sollya_object((left_bound + right_bound) / S2, round_mode=sollya.RU)
        mid_point_img = precision.round_sollya_object(fct(mid_point), round_mode=sollya.RU)
        if mid_point_img >= limit:
            left_bound = mid_point
        elif mid_point_img < limit:
            right_bound = mid_point
        else:
            Log.report(Log.Error, "function must be increasing in search_bound_threshold")
    return left_bound


def piecewise_approximation_degree_generator(
        function,
        bound_low=-1.0,
        bound_high=1.0,
        num_intervals=16,
        max_degree=2,
        error_threshold=S2**-24):
    """ """
    interval_size = (bound_high - bound_low) / num_intervals
    for i in range(num_intervals):
        subint_low = bound_low + i * interval_size
        subint_high = bound_low + (i+1) * interval_size

        local_function = function(sollya.x + subint_low)
        local_interval = Interval(-interval_size, interval_size)

        local_degree = sollya.guessdegree(local_function, local_interval, error_threshold) 
        yield int(sollya.sup(local_degree))



def piecewise_approximation_paramgen(
        function,
        variable,
        coeff_precision,
        bound_low=-1.0,
        bound_high=1.0,
        num_intervals=16,
        max_degree=2,
        error_threshold=S2**-24,
        odd=False,
        even=False,
        axf_export=False):
    """ Generate the parameters of a piecewise approximation

        :param function: function to be approximated
        :type function: SollyaObject
        :param variable: input variable
        :type variable: Variable
        :param coeff_precision: format used to store polynomial coefficients
        :type coeff_precision: ML_Format
        :param bound_low: lower bound for the approximation interval
        :param bound_high: upper bound for the approximation interval
        :param num_intervals: number of sub-interval / sub-division of the main interval
        :param max_degree: maximum degree for an approximation on any sub-interval
        :param error_threshold: error bound for an approximation on any sub-interval

        :return: pair (scheme, error) where scheme is a graph node for an
            approximation scheme of function evaluated at variable, and error
            is the maximum approximation error encountered
        :rtype tuple(ML_Operation, SollyaObject): """
    degree_generator = piecewise_approximation_degree_generator(
        function, bound_low, bound_high,
        num_intervals=num_intervals,
        error_threshold=error_threshold,
    )
    degree_list = list(degree_generator)

    if axf_export:
        axf_error = AXF_ApproxError.from_AE(AbsoluteApproxError(error_threshold))
        axf_approx = AXF_UniformPiecewiseApprox(
            function(sollya.x), coeff_precision, Interval(bound_low, bound_high), num_intervals, max_degree, axf_error)
    else:
        axf_approx = None

    # if max_degree is None then we determine it locally
    if max_degree is None:
        max_degree = max(degree_list)
    # table to store coefficients of the approximation on each segment
    coeff_table = ML_NewTable(
        dimensions=[num_intervals,max_degree+1],
        storage_precision=coeff_precision,
        tag="coeff_table",
        const=True # by default all approximation coeff table are const
    )

    error_function = lambda p, f, ai, mod, t: sollya.dirtyinfnorm(p - f, ai)
    max_approx_error = 0.0
    interval_size = (bound_high - bound_low) / num_intervals

    for i in range(num_intervals):
        subint_low = bound_low + i * interval_size
        subint_high = bound_low + (i+1) * interval_size

        local_function = function(sollya.x + subint_low)
        local_interval = Interval(-interval_size, interval_size)

        local_degree = degree_list[i]
        if local_degree > max_degree:
            Log.report(Log.Warning, "local_degree {} exceeds max_degree bound ({}) in piecewise_approximation", local_degree, max_degree)
        # as max_degree defines the size of the table we can use
        # it as the degree for each sub-interval polynomial
        # as there is nothing to gain (yet) by using a smaller polynomial
        degree = max_degree # min(max_degree, local_degree)

        if function(subint_low) == 0.0:
            # if the lower bound is a zero to the function, we
            # need to force value=0 for the constant coefficient
            # and extend the approximation interval
            local_poly_degree_list = list(range(1 if even else 0, degree+1, 2 if odd or even else 1))
            format_list = [coeff_precision] * len(local_poly_degree_list)
            poly_object, approx_error = Polynomial.build_from_approximation_with_error(
                function(sollya.x) / sollya.x,
                local_poly_degree_list,
                format_list,
                Interval(-subint_high * 0.95,subint_high),
                sollya.absolute,
                error_function=error_function
            )
            # multiply by sollya.x
            poly_object = poly_object.sub_poly(offset=-1)
            if axf_export:
                axf_approx.approx_list.append(
                    AXF_SimplePolyApprox(poly_object, function(sollya.x), [d+1 for d in local_poly_degree_list], format_list, Interval(subint_low, subint_high), absolute=True, approx_error=approx_error)) 
        else:
            try:
                poly_object, approx_error = Polynomial.build_from_approximation_with_error(
                    local_function,
                    degree,
                    [coeff_precision] * (degree + 1),
                    local_interval,
                    sollya.absolute,
                    error_function=error_function
                )
            except SollyaError as err:
                # try to see if function is constant on the interval (possible
                # failure cause for fpminmax)
                cst_value = coeff_precision.round_sollya_object(function(subint_low), sollya.RN)
                accuracy = error_threshold
                diff_with_cst_range = sollya.supnorm(cst_value, local_function, local_interval, sollya.absolute, accuracy)
                diff_with_cst = sup(abs(diff_with_cst_range))
                if diff_with_cst < error_threshold:
                    Log.report(Log.Info, "constant polynomial detected")
                    poly_object = Polynomial([function(subint_low)] + [0] * degree)
                    approx_error = diff_with_cst
                else:
                    Log.report(Log.error, "degree: {} for index {}, diff_with_cst={} (vs error_threshold={}) ", degree, i, diff_with_cst, error_threshold, error=err)
            if axf_export:
                axf_error = AXF_ApproxError.from_AE(AbsoluteApproxError(approx_error))
                axf_poly = AXF_Polynomial.from_poly(poly_object)
                axf_approx.approx_list.append(
                    
                    AXF_SimplePolyApprox(axf_poly, local_function, range(degree+1), [coeff_precision] * (degree+1), Interval(subint_low, subint_high), approx_error=axf_error)) 
        for ci in range(max_degree+1):
            if ci in poly_object.coeff_map:
                coeff_table[i][ci] = poly_object.coeff_map[ci]
            else:
                coeff_table[i][ci] = 0.0

        if approx_error > error_threshold:
            Log.report(Log.Warning, "piecewise_approximation on index {} exceeds error threshold: {} > {}", i, approx_error, error_threshold)
        max_approx_error = max(max_approx_error,abs(approx_error))


    return interval_size, coeff_table, max_approx_error, max_degree, axf_approx

def piecewise_param_from_axf(axf_approx):
    """ load a piecewise approximation parameter from an
        Approximation eXchange Format (AXF) storage """

    max_degree = axf_approx.max_degree
    num_intervals = axf_approx.num_intervals
    coeff_precision = axf_approx.precision
    bound_high = sup(axf_approx.interval)
    bound_low = inf(axf_approx.interval)
    error_threshold = axf_approx.error_bound

    # table to store coefficients of the approximation on each segment
    coeff_table = ML_NewTable(
        dimensions=[num_intervals, max_degree+1],
        storage_precision=coeff_precision,
        tag="coeff_table",
        const=True # by default all approximation coeff table are const
    )

    error_function = lambda p, f, ai, mod, t: sollya.dirtyinfnorm(p - f, ai)
    max_approx_error = AbsoluteApproxError(0.0)
    interval_size = (bound_high - bound_low) / num_intervals

    for i in range(num_intervals):
        subint_low = bound_low + i * interval_size
        subint_high = bound_low + (i+1) * interval_size

        local_interval = Interval(-interval_size, interval_size)

        local_approx = axf_approx.approx_list[i]
        assert isinstance(local_approx.approx_error, AbsoluteApproxError)
        approx_error = local_approx.approx_error

        poly_object = local_approx.poly
        axf_interval = local_approx.interval

        # enforcing validity checks on local approximation
        assert subint_low == sollya.inf(axf_interval)
        assert subint_high == sollya.sup(axf_interval)

        for ci in range(max_degree+1):
            if ci in poly_object.coeff_map:
                coeff_table[i][ci] = poly_object.coeff_map[ci]
            else:
                coeff_table[i][ci] = 0.0

        if approx_error > error_threshold:
            Log.report(Log.Warning, "piecewise_approximation on index {} exceeds error threshold: {} > {}", i, approx_error, error_threshold)
        assert approx_error.value >= 0
        max_approx_error = max(max_approx_error,approx_error)

    return interval_size, coeff_table, max_approx_error, max_degree

def piecewise_approximation(
        function,
        variable,
        coeff_precision,
        bound_low=-1.0,
        bound_high=1.0,
        num_intervals=16,
        max_degree=2,
        error_threshold=S2**-24,
        odd=False,
        even=False):
    """ Generate a piecewise approximation

        :param function: function to be approximated
        :type function: SollyaObject
        :param variable: input variable
        :type variable: Variable
        :param coeff_precision: format used to store polynomial approximation coefficient
        :type coeff_precision: ML_Format
        :param bound_low: lower bound for the approximation interval
        :param bound_high: upper bound for the approximation interval
        :param num_intervals: number of sub-interval / sub-division of the main interval
        :param max_degree: maximum degree for an approximation on any sub-interval
        :param error_threshold: error bound for an approximation on any sub-interval

        :return: pair (scheme, error) where scheme is a graph node for an
            approximation scheme of function evaluated at variable, and error
            is the maximum approximation error encountered
        :rtype tuple(ML_Operation, SollyaObject): """


    # NOTES: max_degree may-be updated (if pre-defined as None) when
    #        returning from piecewise_approximation_paramgen
    interval_size, coeff_table, max_approx_error, max_degree, axf_export = piecewise_approximation_paramgen(
        function,
        variable,
        coeff_precision,
        bound_low=bound_low,
        bound_high=bound_high,
        num_intervals=num_intervals,
        max_degree=max_degree,
        error_threshold=error_threshold,
        odd=odd,
        even=even)

    return piecewise_evaluation_from_param(variable, coeff_precision, bound_low, bound_high, max_degree, num_intervals, interval_size, coeff_table), max_approx_error

def piecewise_evaluation_from_param(variable, precision, bound_low, bound_high, max_degree, num_intervals, interval_size, coeff_table):
    """ Generate a piecewise evaluation scheme from a pre-determined coefficient table 

        :param variable: input variable
        :type variable: Variable
        :param precision: format used to store polynomial approximation coefficient and to perform
                          intermediary computation
        :type precision: ML_Format
        :param bound_low: lower bound for the approximation interval
        :param bound_high: upper bound for the approximation interval
        :param num_intervals: number of sub-interval / sub-division of the main interval
        :type num_intervals: int
        :param interval_size: size of the full approximation interval
        :param coeff_table: pre-computed table of polynomial coefficients
        :type coeff_table: ML_NewTable

        :return: a graph node for an approximation scheme of function evaluated at variable
        :rtype ML_Operation: """
    # computing offset
    # TODO/FIXME variable's precision could be used when evaluating
    #            expression depending on variable node,
    #            or a specific precision could be specified
    #            as an extra function parameter
    diff = Subtraction(
        variable,
        Constant(bound_low, precision=precision),
        tag="diff",
        debug=debug_multi,
        precision=precision
    )
    int_prec = precision.get_integer_format()

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
        debug=debug_multi,
        precision=int_prec
    )
    poly_var = Subtraction(
        diff,
        Multiplication(
            Conversion(index, precision=precision),
            Constant(interval_size, precision=precision)
        ),
        precision=precision,
        tag="poly_var",
        debug=debug_multi
    )
    # generating indexed polynomial
    coeffs = [(ci, TableLoad(coeff_table, index, ci)) for ci in range(max_degree+1)][::-1]
    poly_scheme = PolynomialSchemeEvaluator.generate_horner_scheme2(
        coeffs,
        poly_var,
        precision, {}, precision
    )
    return poly_scheme

