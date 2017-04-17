# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Metalibm tool
# Copyright (2017)
# All rights reserved
# created:          Apr 16th, 2017
# last-modified:    Apr 16th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################

import sollya
from .ml_operations import LogicalOr, Comparison, FunctionObject 
from metalibm_core.code_generation.generator_utility import *

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
  def get_output_check_value(self, emulated_function, input_value):
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
    
## Faithful (error <= 1 ulp) rounding output precision indication
class ML_Faithful(ML_FunctionPrecision):
  def get_num_output_value(self):
    return 2
  def get_output_check_value(self, numeric_emulate, input_value):
    low_bound  = self.precision.round_sollya_object(numeric_emulate(input_value), sollya.RD)
    high_bound = self.precision.round_sollya_object(numeric_emulate(input_value), sollya.RU)
    return low_bound, high_bound

  def get_output_check_test(self, test_result, stored_outputs):
    low_bound, high_bound = stored_outputs
    failure_test = LogicalOr(
      Comparison(test_result, low_bound, specifier = Comparison.Less),
      Comparison(test_result, high_bound, specifier = Comparison.Greater)
    )
    return failure_test
  def get_output_print_function(self, function_name, footer = "\\n"):
    printf_op = FunctionOperator(
      "printf", 
      arg_map = {
        0: "\"[{display_format};{display_format}]{footer}\"".format(display_format = self.precision.get_display_format(), footer = footer), 
        1: FO_Arg(0), 
        2: FO_Arg(1) 
      }, void_function = True) 
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
  def get_output_check_value(self, numeric_emulate, input_value):
    expected_value = self.precision.round_sollya_object(numeric_emulate(input_value), sollya.RN)
    return (expected_value,)

  def get_output_check_test(self, test_result, stored_outputs):
    expected_value,  = stored_outputs
    failure_test = Comparison(
      test_result, 
      expected_value, 
      specifier = Comparison.NotEqual
    )
    return failure_test
  def get_output_print_function(self, function_name, footer = "\\n"):
    printf_op = FunctionOperator(
      "printf", 
      arg_map = {
        0: "\"{display_format}{footer}\"".format(display_format = self.precision.get_display_format(), footer = footer), 
        1: FO_Arg(0) 
      }, void_function = True) 
    printf_function = FunctionObject("printf", [self.precision], ML_Void, printf_op)
    return printf_function

  def get_output_print_call(self, function_name, output_values, footer = "\\n"):
    expected_value, = output_values
    print_function = self.get_output_print_function(function_name, footer)
    return print_function(expected_value)

## Degraded accuracy function output precision indication
class ML_DegradedAccuracy(ML_FunctionPrecision):
  def __init__(self, goal):
    self.goal = goal

  ## return the absolute or relative goal assocaited
  #  with the accuracy object
  def get_goal(self):
    return self.goal

## Degraded accuracy with absoute error function output precision indication
class ML_DegradedAccuracyAbsolute(ML_DegradedAccuracy):
  """ absolute error accuracy """
  def __init__(self, absolute_goal):
    ML_DegradedAccuracy.__init__(self, absolute_goal)

  def __str__(self):
    return "ML_DegradedAccuracyAbsolute(%s)" % self.goal

## Degraded accuracy with relative error function output precision indication
class ML_DegradedAccuracyRelative(ML_DegradedAccuracy):
  """ relative error accuracy """
  def __init__(self, relative_goal):
    ML_DegradedAccuracy.__init__(self, relative_goal)

  def __str__(self):
    return "ML_DegradedAccuracyRelative(%s)" % self.goal


## degraded absolute accuracy alias
def daa(*args, **kwords):
    return ML_DegradedAccuracyAbsolute(*args, **kwords)
## degraded relative accuracy alias
def dar(*args, **kwords):
    return ML_DegradedAccuracyRelative(*args, **kwords)
