# -*- coding: utf-8 -*-
""" Utilities for unit-testing """

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
# created:
# last-modified:    Mar  6th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
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
    if inspect.isclass(runner) and TestRunner in runner.__bases__:
      arg_template = runner.get_default_args(**arg_tc)
      runner = runner.__call__
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


