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


def load_piecewese_poly_params_from_axf(axf_approx, indexing):
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


def generate_parameters_piecewise_poly_approx(offset_fct, indexing, target_eps, coeff_precision, max_degree=None, axf_export=False, error_target_type=sollya.relative):
    """ generate the parameters (table) for a generic piecewise polynomial
        approximation """
    # computing degree for a different polynomial approximation on each
    # sub-interval
    if max_degree is None:
        poly_degree_list = [int(sup(guessdegree(offset_fct(offset), sub_interval, target_eps))) for offset, sub_interval in indexing.get_offseted_sub_list()]
        max_degree = max(poly_degree_list)
    else:
        poly_degree_list = [max_degree for index in range(indexing.split_num)]
    Log.report(Log.Debug, "generate_parameters_piecewise_poly_approx max_degree={}", max_degree)

    # tabulating polynomial coefficients on split_num sub-interval of interval
    poly_table = ML_NewTable(dimensions=[indexing.split_num, max_degree+1], storage_precision=coeff_precision, const=True)
    offset_table = ML_NewTable(dimensions=[indexing.split_num], storage_precision=coeff_precision, const=True)

    ErrorCtor = AbsoluteApproxError if error_target_type is sollya.absolute else RelativeApproxError
    max_error = ErrorCtor(0.0)


    # object for AXF export
    if axf_export:
        # TODO/FIXME/ using offset_fct evaluation at 0 to provide a dumpable
        #             function. We may prefer an non-evaluated offset_fct
        #             transcription
        axf_error = AXF_ApproxError.from_AE(ErrorCtor(target_eps))
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
            if error_target_type is sollya.absolute:
                approx_error_value = sup(abs(sollya.infnorm(offset_fct(offset) - local_approx, approx_interval)))
            elif error_target_type is sollya.relative:
                approx_error_value = sup(abs(sollya.infnorm((offset_fct(offset) - local_approx) / offset_fct(offset_fct), approx_interval)))
            else:
                raise NotImplementedError
            approx_error = ErrorCtor(approx_error_value)

            if axf_export:
                axf_poly = AXF_Polynomial.from_poly(Polynomial({0: local_approx}))
                # axf_error = AXF_ApproxError.from_AE(AbsoluteApproxError(approx_error))
                axf_error = AXF_ApproxError.from_AE(approx_error)
                axf_approx.approx_list.append(
                    AXF_SimplePolyApprox(axf_poly,
                                         offset_fct(offset), [0], [coeff_precision],
                                         approx_interval,
                                         approx_error=axf_error))

        else:
            try:
                if 0 in approx_interval and offset_fct(offset)(0) == 0.0:
                    # if 0 is within the local interval and that the function has a zero at zero,
                    # we force the first coefficient to be 0
                    # NOTES: having a zero makes it difficult to target relative error
                    assert error_target_type is sollya.absolute
                    poly_object, approx_error_value = Polynomial.build_from_approximation_with_error(
                        offset_fct(offset), list(range(1,poly_degree+1)), [coeff_precision]*(poly_degree),
                        approx_interval, error_target_type)
                    approx_error = AbsoluteApproxError(approx_error_value)
                else:
                    poly_object, approx_error_value = Polynomial.build_from_approximation_with_error(
                        offset_fct(offset), poly_degree, [coeff_precision]*(poly_degree+1),
                        approx_interval, error_target_type)
                    # TODO/FIXME: not sure build_from_approximation_with_error
                    #             is returning an error of the proper
                    #             <error_target_type> type
                    approx_error = ErrorCtor(approx_error_value)
            except SollyaError as err:
                # try to see if function is constant on the interval (possible
                # failure cause for fpminmax)
                subint_low = inf(approx_interval)
                local_function = offset_fct(offset)
                local_interval = approx_interval
                #import pdb; pdb.set_trace()
                cst_value = coeff_precision.round_sollya_object(local_function(subint_low), sollya.RN)
                accuracy = target_eps
                diff_with_cst_range = sollya.supnorm(cst_value, local_function, local_interval, sollya.absolute, accuracy)
                diff_with_cst = sup(abs(diff_with_cst_range))
                if diff_with_cst < target_eps:
                    Log.report(Log.Info, "constant polynomial detected")
                    poly_object = Polynomial([cst_value] + [0] * poly_degree)
                    if error_target_type is sollya.absolute:
                        approx_error_value = diff_with_cst
                    elif error_target_type is sollya.relative:
                        approx_error_value = diff_with_cst / sollya.infnorm(local_function, local_interval)

                    else:
                        raise NotImplementedError
                    approx_error = ErrorCtor(approx_error_value)
                else:
                    Log.report(Log.Error, "degree: {} for index {}, diff_with_cst={} (vs error_threshold={}) ", poly_degree, sub_index, diff_with_cst, target_eps, error=err)

            for monomial_index in range(max_degree+1):
                if monomial_index <= poly_degree:
                    if monomial_index in poly_object.coeff_map:
                        coeff_value = poly_object.coeff_map[monomial_index] 
                    else:
                        coeff_value = 0
                    poly_table[sub_index][monomial_index] = coeff_value
                else:
                    poly_table[sub_index][monomial_index] = 0

            if axf_export:
                axf_poly = AXF_Polynomial.from_poly(poly_object)
                # axf_error = AXF_ApproxError.from_AE(RelativeApproxError(approx_error))
                axf_error = AXF_ApproxError.from_AE(approx_error)
                axf_approx.approx_list.append(
                    AXF_SimplePolyApprox(axf_poly,
                                         offset_fct(offset), list(range(poly_degree+1)),
                                         [coeff_precision]*(poly_degree+1),
                                         approx_interval,
                                         approx_error=axf_error)) 
        max_error = max(approx_error, max_error)

    # if an axf approx is being exported, we need to update the stored
    # approximation error
    if not axf_approx is None:
        axf_approx.approx_error = AXF_ApproxError.from_AE(max_error)

    return offset_table, max_degree, poly_table, max_error, axf_approx


def generate_piecewise_poly_approx_from_params(offset_table, max_degree, poly_table, indexing, coeff_precision, vx):
    """ generate the ML node graph of approximation of a function
        from the parameter of a generic piecewise polynomial approximation """
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

def generate_piecewise_poly_approx(offset_fct, indexing, target_eps, coeff_precision, vx, max_degree=None, error_target_type=sollya.relative, axf_export=False):
    """ generate the meta approximation for @p offset_fct over several
        intervals defined by @p indexing object
        For each sub-interval, a polynomial approximation with
        maximal_error @p target_eps is tabulated, and evaluated using format
        @p coeff_precision.
        The input variable is @p vx """

    offset_table, max_degree, poly_table, max_error, axf_approx = generate_parameters_piecewise_poly_approx(offset_fct, indexing,
                                                                    target_eps, coeff_precision,
                                                                    max_degree=max_degree,
                                                                    error_target_type=error_target_type,
                                                                    axf_export=axf_export)
    Log.report(Log.Debug, "max approx error is {}", max_error)

    poly = generate_piecewise_poly_approx_from_params(offset_table, max_degree, poly_table, indexing, coeff_precision, vx)

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


