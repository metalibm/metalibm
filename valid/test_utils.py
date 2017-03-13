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

class CommonTestScheme:
  ## @param title name of the test
  #  @param argument_tc list of argument tests cases (dict)
  def __init__(self, title, argument_tc):
    self.title = title
    self.argument_tc = argument_tc

  ## @return test object title
  def get_title(self):
    return self.title

  def perform_all_test(self, debug = False):
    result_list = [self.single_test(tc, debug = debug) for tc in self.argument_tc]
    success_count = [r.get_result() for r in result_list].count(True)
    failure_count = len(result_list) - success_count
    overall_success = (success_count > 0) and (failure_count == 0)
    function_name = self.get_title()

    if overall_success:
      return TestResult(True, "{} success ! ({}/{})".format(function_name, success_count, len(result_list)))
    else:
      return TestResult(False, "{} success ! ({}/{})\n {}".format(function_name, success_count, len(result_list), "\n".join(r.get_details() for r in result_list)))

# Test object for new type meta function
class NewSchemeTest(CommonTestScheme):
  #  @param ctor MetaFunction constructor
  def __init__(self, title, ctor, argument_tc):
    CommonTestScheme.__init__(self, title, argument_tc)
    self.ctor = ctor

  def single_test(self, arg_tc, debug = False):
    function_name = self.get_title()
    test_desc = "{}/{}".format(function_name, str(arg_tc))
    arg_template = DefaultArgTemplate(**arg_tc) 
    if debug:
      fct = self.ctor(arg_template)
      fct.gen_implementation()

    else:
      try:
        fct = self.ctor(arg_template)
      except:
        return TestResult(False, "{} ctor failed".format(test_desc))
      try:
        fct.gen_implementation()
      except:
        return TestResult(False, "{} gen_implementation failed".format(test_desc))
      
    return TestResult(True, "{} succeed".format(test_desc))

