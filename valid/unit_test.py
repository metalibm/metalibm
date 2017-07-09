# -*- coding: utf-8 -*-
""" Utilities for unit-testing """

###############################################################################
# This file is part of New Metalibm tool
# Copyrights Nicolas Brunie (2016-)
# All rights reserved
# created:          
# last-modified:    Jul  9th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
# description: utilities for unit testing
###############################################################################

import argparse
import inspect


from valid.test_utils import *

from metalibm_core.core.ml_formats import ML_Int32, ML_Int16, ML_Int64


from metalibm_functions.unit_tests.utils import TestRunner



## Object to describe a unit-test
class UnitTestScheme(CommonTestScheme):
  ## Constructor
  #  @param title test's title
  #  @param module Python module implementing the test (
  #     should provide run_test method accepting ArgTemplate 
  #     arg)
  #   @param argument_tc list of dict test case (arg_name -> arg_value)
  def __init__(self, title, module, argument_tc):
    CommonTestScheme.__init__(self, title, argument_tc)
    self.module  = module

  ## @return test object title
  def get_title(self):
    return self.title

  def single_test(self, arg_tc, debug = False):
    runner = self.module.run_test
    test_desc = self.get_title()
    print isinstance(runner, TestRunner)
    if inspect.isclass(runner) and TestRunner in runner.__bases__:
      arg_template = runner.build_default_args(**arg_tc)
    else:
      arg_template = DefaultArgTemplate(**arg_tc) 

    if debug:
      runner(arg_template)
      return TestResult(True, "{} succeed".format(test_desc))
    else:
      try:
        runner(arg_template)
        return TestResult(True, "{} succeed".format(test_desc))
      except:
        return TestResult(False, "{} failed".format(test_desc))

## Command line action to set break on error in load module
class ListUnitTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  unit_test_list:
          print test.get_tag_title()
        exit(0)


# generate list of test object from string 
# of comma separated test's tag
def parse_unit_test_list(test_list):
  test_tags = test_list.split(",")
  return [unit_test_tag_map[tag] for tag in test_tags]


