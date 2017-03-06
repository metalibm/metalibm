# -*- coding: utf-8 -*-

# This file is part of the Metalibm project
# author(s): Nicolas Brunie (nibrunie@gmail.com)
#
# Description: Utility for Metalibm validation
# Created:     March 6th, 2017

from metalibm_core.core.ml_function import DefaultArgTemplate

class TestResult:
  ## @param result boolean indicating success (True) or failure (False)
  #  @param details string with test information
  def __init__(self, result, details):
    self.result = result
    self.details = details

  def get_result(self):
    return self.result

  def get_details(self):
    return self.details

# Test object for new type meta function
class NewSchemeTest:
  ## @param title name of the test
  #  @param ctor MetaFunction constructor
  #  @param argument_tc list of argument tests cases (dict)
  def __init__(self, title, ctor, argument_tc):
    self.title = title
    self.ctor = ctor
    self.argument_tc = argument_tc

  ## @return test object title
  def get_title(self):
    return self.title

  def single_test(self, arg_tc):
    function_name = self.get_title()
    test_desc = "{}/{}".format(function_name, str(arg_tc))
    arg_template = DefaultArgTemplate(**arg_tc) 
    try:
      fct = self.ctor(arg_template)
    except:
      return TestResult(False, "{} ctor failed".format(test_desc))
    try:
      fct.gen_implementation()
    except:
      return TestResult(False, "{} gen_implementation failed".format(test_desc))

    return TestResult(True, "{} succeed".format(test_desc))

  def perform_all_test(self):
    result_list = [self.single_test(tc) for tc in self.argument_tc]
    success_count = [r.get_result() for r in result_list].count(True)
    failure_count = len(result_list) - success_count
    overall_success = (success_count > 0) and (failure_count == 0)
    function_name = self.get_title()

    if overall_success:
      return TestResult(True, "{} success ! ({}/{})".format(function_name, success_count, len(result_list)))
    else:
      return TestResult(False, "{} success ! ({}/{})\n {}".format(function_name, success_count, len(result_list), "\n".join(r.get_details() for r in result_list)))
