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
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
#
# Description: Utility for Metalibm validation
# Created:           March 6th, 2017
# Last Modified:     March 6th, 2018
###############################################################################

from metalibm_core.core.ml_function import (
    DefaultArgTemplate, BuildError, ValidError
)

class GenerationError(Exception):
    """ Exception indicating that an error occured during code generation """
    pass

class TestResult:
  ## @param result boolean indicating success (True) or failure (False)
  #  @param details string with test information
  #  @param test_object CommonTestScheme object defining the test
  #  @param test_case specific test parameters used in the test
  def __init__(self, result, details, test_object=None, test_case=None, error=None):
    self.result = result
    self.details = details
    self.test_object = test_object
    self.test_case = test_case
    self.error = error

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

  @property
  def num_test(self):
    return len(self.argument_tc)

  ## get a transform version of title with no space
  def get_tag_title(self):
    return self.title.replace(" ", "_")

  def perform_all_test_no_reduce(self, debug=False):
    """ perform all test in CommonTestScheme and returns
        raw TestResult list """
    return [self.single_test(tc, debug = debug) for tc in self.argument_tc]

  def get_success_count(self, result_list):
    success_count = [r.get_result() for r in result_list].count(True)
    return success_count

  def reduce_test_result(self, result_list):
    """ Reduce a test result list to a single agglomerated test result """
    success_count = self.get_success_count(result_list)
    failure_count = len(result_list) - success_count
    overall_success = (success_count >= 0) and (failure_count == 0)
    function_name = self.get_title()

    if overall_success:
      result_msg = "{} success ! ({}/{})".format(function_name, success_count, len(result_list))
      return TestResult(True, result_msg)
    else:
      result_msg = "{} success ! ({}/{})\n {}".format(
            function_name, success_count, len(result_list), 
            "\n".join(r.get_details() for r in result_list))
      return TestResult(False, result_msg)

  def perform_all_test(self, debug=False):
    """ Perform all test of the scheme ahd then reduce test results
        to a single object """
    result_list = self.perform_all_test_no_reduce(debug)
    return self.reduce_test_result(result_list)

# Test object for new type meta function
class NewSchemeTest(CommonTestScheme):
    #    @param ctor MetaFunction constructor
    def __init__(self, title, ctor, argument_tc):
        CommonTestScheme.__init__(self, title, argument_tc)
        self.ctor = ctor

    ## Build an argument template from dict
    def build_arg_template(self, **kw):
        return self.ctor.get_default_args(**kw)

    def single_test(self, arg_tc, debug = False):
        function_name = self.get_title()
        test_desc = "{}/{}".format(function_name, str(arg_tc))
        arg_template = self.build_arg_template(**arg_tc)

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
            except (BuildError, ValidError) as e:
                return TestResult(False, "{} gen_implementation failed".format(test_desc), error=e)
            except:
                return TestResult(False, "{} gen_implementation failed".format(test_desc), error=GenerationError())
            
        return TestResult(True, "{} succeed".format(test_desc))

