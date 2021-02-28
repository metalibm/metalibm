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

###############################################################################
# This file is part of Metalibm tool
# created:          Apr 16th, 2017
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################

import sollya
from .ml_operations import (
    LogicalAnd, LogicalNot,
    LogicalOr, NotEqual,
    Select, Equal,
    Comparison, FunctionObject, Min, Abs, Subtraction, Division,
    TypeCast, ComponentSelection)
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.core.ml_formats import (
    is_floating_format, ML_FP_MultiElementFormat)

from metalibm_core.core.special_values import FP_SpecialValue
from metalibm_core.opt.opt_utils import (logical_or_reduce, logical_and_reduce)

ml_infty = sollya.parse("infty")

## Parent class for output precision indication/constraint
class ML_FunctionPrecision(object):
  def __init__(self, precision):
    self.precision = precision
  ## return the number of output values required
  #  for each test input
  def get_num_output_value(self):
    raise NotImplementedError
  def get_precision(self):
    return self.precision
  ## return a tuple of output values required to check a test
  # @param exact_value exact (possibly unevaluated expression) representation of the value
  #        to be checked
  def get_output_check_value(self, exact_value):
    """ return the reference values required to check result """
    raise NotImplementedError
  ## return an Operation graph for testing if test_result
  #  fails numeric test defined by @p self accuracy and @p stored_outputs
  #  numeric output values
  def get_output_check_test(self, test_result, stored_outputs):
    raise NotImplementedError
  def get_error_print_function(self, function_name):
    raise NotImplementedError
  def get_error_print_call(self, function_name, input_value, local_result, output_values):
    raise NotImplementedError
  ## generate an optree to get evaluation error between local_result 
  # and output_values
  # @param relative select the computation of a relative error (default: absolute)
  def compute_error(self, local_result, output_values, relative = False):
    raise NotImplementedError
    
class ML_TwoFactorPrecision(ML_FunctionPrecision):
    """ Precision defined by two bounds (low and high) """
    def get_num_output_value(self):
        return 2
    def get_check_value_low_bound(self, exact_value):
        """ return the low bound of the expected interval """
        raise NotImplementedError
    def get_check_value_high_bound(self, exact_value):
        """ return the high bound of the expected interval """
        raise NotImplementedError
    def get_output_check_value(self, exact_value):
        """ return the reference values required to check result """
        if FP_SpecialValue.is_special_value(exact_value):
            return (exact_value, exact_value)
        else:
            low_bound = self.get_check_value_low_bound(exact_value)
            high_bound = self.get_check_value_high_bound(exact_value)
            return low_bound, high_bound

    def compute_error(self, local_result, stored_outputs, relative = False):
        """ return MDL expression to compute error between local_result and 
            stored_outputs """
        precision = local_result.get_precision()
        low_bound, high_bound = stored_outputs
        error = Min(
          Abs(Subtraction(local_result, low_bound, precision = precision), precision = precision),
          Abs(Subtraction(local_result, high_bound, precision = precision), precision = precision),
          precision = precision
        )
        if relative:
            # to avoid NaN when dividing 0 / 0
            error = Select(
                LogicalAnd(Equal(error, 0), Equal(local_result, 0)),
                local_result,
                Abs(Division(error, local_result, precision = precision), precision = precision),
                precision=precision
            )
        return error

    def get_output_check_test(self, test_result, stored_outputs):
        low_bound, high_bound = stored_outputs
        # to circumvent issue #25: failure to detected unpexcted NaNs
        # check for failure was changed for an inverted check for success
        # such that the test succeed only if the value is within the bound
        # and not fails if the value lies outside the bound (which fails to
        # raise unexpected NaNs as failure)
        success_test = LogicalOr(
            LogicalAnd(
                Comparison(test_result, low_bound, specifier=Comparison.GreaterOrEqual),
                Comparison(test_result, high_bound, specifier=Comparison.LessOrEqual)
            ),
            # NaN comparison
            LogicalAnd(
                LogicalOr(
                    NotEqual(low_bound, low_bound),
                    NotEqual(high_bound, high_bound),
                ),
                NotEqual(test_result, test_result)
            )
            # exact zero comparison
        )
        failure_test = LogicalNot(success_test)
        return failure_test
    def get_output_print_function(self, function_name, footer="\\n"):
        printf_template = "printf(\"[%s;%s]%s\", %s, %s)" % (
            self.precision.get_display_format(C_Code).format_string,
            self.precision.get_display_format(C_Code).format_string,
            footer,
            self.precision.get_display_format(C_Code).pre_process_fct("{0}"),
            self.precision.get_display_format(C_Code).pre_process_fct("{1}"),
        )
        printf_op = TemplateOperatorFormat(printf_template, arity=2, void_function=True)
        printf_function = FunctionObject("printf", [self.precision] * 2, ML_Void, printf_op)
        return printf_function

    def get_output_print_call(self, function_name, output_values, footer = "\\n"):
        low, high = output_values
        print_function = self.get_output_print_function(function_name, footer)
        return print_function(low, high)

## Correctly Rounded (error <= 0.5 ulp) rounding output precision indication
class ML_CorrectlyRounded(ML_FunctionPrecision):
  def get_num_output_value(self):
    return 1
  def get_output_check_value(self, exact_value):
    if FP_SpecialValue.is_special_value(exact_value):
        return (exact_value, )
    else:
        expected_value = self.precision.round_sollya_object(exact_value, sollya.RN)
        return (expected_value,)

  def compute_error(self, local_result, output_values, relative = False):
    precision = local_result.get_precision()
    expected_value,  = output_values
    error = Subtraction(local_result, expected_value, precision = precision)
    if relative:
      # to avoid NaN when dividing 0 / 0
      error = Select(
        LogicalAnd(Equal(error, 0), Equal(local_result, 0)),
        local_result,
        Abs(Division(error, local_result, precision = precision), precision = precision),
        precision=precision
      )
    else:
      error = Abs(error, precision = precision)
    return error

  def get_output_check_test(self, test_result, stored_outputs):
    """ generate operation graph to implement correctly rounded equality test
        of <test_result> vs <stored_outputs> """
    expected_value,  = stored_outputs
    expected_format = expected_value.get_precision()
    assert not expected_format is None
    if is_floating_format(expected_format):
        if ML_FP_MultiElementFormat.is_fp_multi_elt_format(expected_format):
            # specific processing for multi-precision floating-point values
            limb_num = expected_format.limb_num
            expected_limbs = [ComponentSelection(expected_value, ComponentSelection.Field(limb_index)) for limb_index in range(limb_num)]
            result_limbs = [ComponentSelection(test_result, ComponentSelection.Field(limb_index)) for limb_index in range(limb_num)]

            zero_expected = logical_and_reduce(Equal(expected_limbs[limb_index], 0) for limb_index in range(limb_num)) 
            nan_expected = logical_or_reduce(NotEqual(expected_limbs[limb_index], expected_limbs[limb_index]) for limb_index in range(limb_num)) 
            nan_detected = logical_or_reduce(NotEqual(result_limbs[limb_index], result_limbs[limb_index]) for limb_index in range(limb_num)) 

            # TODO/FIXME only extracting first limb precision
            int_format = test_result.get_precision().get_limb_precision(0).get_integer_format()
            casted_expected = [TypeCast(expected_limbs[limb_index], precision=int_format) for limb_index in range(limb_num)]
            casted_result = [TypeCast(result_limbs[limb_index], precision=int_format) for limb_index in range(limb_num)]
            bitexact_comparison = logical_and_reduce(Equal(casted_expected[limb_index], casted_result[limb_index]) for limb_index in range(limb_num))
        else:
            nan_expected = NotEqual(expected_value, expected_value, tag="nan_expected")
            nan_detected = NotEqual(test_result, test_result, tag="nan_detected")
            # zero expected use cases
            zero_expected = Equal(expected_value, 0, tag="zero_expected")

            # implementing bit-exact test through bitcast from floating-point
            # to integer and integer equality checking
            int_format = test_result.get_precision().get_integer_format()
            bitexact_comparison = Equal(
                TypeCast(test_result, precision=int_format),
                TypeCast(expected_value, precision=int_format),
            )
        failure_test = LogicalOr(
            LogicalOr(
                LogicalAnd(
                    LogicalNot(nan_expected),
                    Comparison(
                      test_result,
                      expected_value,
                      specifier=Comparison.NotEqual,
                    ),
                   tag="value_failure",
                ),
                LogicalAnd(nan_expected, LogicalNot(nan_detected), tag="nan_failure")
            ),
            # bit exact zero comparison
            LogicalAnd(zero_expected, LogicalNot(bitexact_comparison))
        )
    else:
        failure_test = NotEqual(test_result, expected_value)
    return failure_test
  def get_output_print_function(self, function_name, footer="\\n"):
    printf_template = "printf(\"%s%s\", %s)" % (
        self.precision.get_display_format().format_string,
        footer,
        self.precision.get_display_format().pre_process_fct("{0}")
    )
    printf_op = TemplateOperatorFormat(printf_template, arity=1, void_function=True)
    printf_function = FunctionObject("printf", [self.precision], ML_Void, printf_op)
    return printf_function

  def get_output_print_call(self, function_name, output_values, footer = "\\n"):
    expected_value, = output_values
    print_function = self.get_output_print_function(function_name, footer)
    return print_function(expected_value)

## Faithful (error <= 1 ulp) rounding output precision indication
class ML_Faithful(ML_TwoFactorPrecision):
    """ Faithful (abs(error) < 1 ulp) precision """ 
    def get_check_value_low_bound(self, exact_value):
        low_bound  = self.precision.round_sollya_object(exact_value, sollya.RD)
        return low_bound
    def get_check_value_high_bound(self, exact_value):
        high_bound = self.precision.round_sollya_object(exact_value, sollya.RU)
        return high_bound

## Degraded accuracy function output precision indication
class ML_DegradedAccuracy(ML_TwoFactorPrecision):
    def __init__(self, goal):
        # maxium error goal
        self.goal = goal

    def get_value_error_goal(self, value):
        raise NotImplementedError

    def get_check_value_cr(self, exact_value):
        """ return correctly rounded value """
        return self.precision.round_sollya_object(exact_value, sollya.RN)
    def get_check_value_low_bound(self, exact_value):
        cr_value = self.get_check_value_cr(exact_value)
        value_goal = self.get_value_error_goal(cr_value)
        if abs(cr_value) == ml_infty:
            return cr_value
        return self.precision.saturate(cr_value - value_goal)
    def get_check_value_high_bound(self, exact_value):
        cr_value = self.get_check_value_cr(exact_value)
        value_goal = self.get_value_error_goal(cr_value)
        if abs(cr_value) == ml_infty:
            return cr_value
        return self.precision.saturate(cr_value + value_goal)

    def set_precision(self, precision):
        self.precision = precision
        return self

## Degraded accuracy with absoute error function output precision indication
class ML_DegradedAccuracyAbsolute(ML_DegradedAccuracy, ):
    """ absolute error accuracy """
    def __init__(self, absolute_goal):
        ML_DegradedAccuracy.__init__(self, absolute_goal)

    def get_value_error_goal(self, value):
        return abs(self.goal)

    def __str__(self):
        return "ML_DegradedAccuracyAbsolute(%s)" % self.goal

## Degraded accuracy with relative error function output precision indication
class ML_DegradedAccuracyRelative(ML_DegradedAccuracy):
    """ relative error accuracy """
    def __init__(self, relative_goal):
        ML_DegradedAccuracy.__init__(self, relative_goal)

    def get_value_error_goal(self, value):
        return abs(self.goal * value)

    def __str__(self):
        return "ML_DegradedAccuracyRelative(%s)" % self.goal


## degraded absolute accuracy alias
def daa(*args, **kwords):
    return lambda precision: ML_DegradedAccuracyAbsolute(*args, **kwords).set_precision(precision)
## degraded relative accuracy alias
def dar(*args, **kwords):
    return lambda precision: ML_DegradedAccuracyRelative(*args, **kwords).set_precision(precision)
