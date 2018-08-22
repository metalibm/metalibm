# -*- coding: utf-8 -*-
""" Software source code unit testing """

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
# created:          Jul  9th, 2017
# last-modified:    Mar  6th, 201_
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
# description: software source code unit testing
###############################################################################

import sys
import argparse

from sollya import Interval

from metalibm_core.targets import *
import metalibm_core.code_generation.mpfr_backend

from metalibm_core.utility.ml_template import target_instanciate
from metalibm_core.core.ml_formats import ML_Int32, ML_Int16, ML_Int64

from valid.unit_test import (
    UnitTestScheme
)

import metalibm_hw_blocks.unit_tests.adaptative_size as ut_adaptative_size
import metalibm_hw_blocks.unit_tests.report_test as ut_report_test
import metalibm_hw_blocks.unit_tests.range_eval as ut_range_eval
import metalibm_hw_blocks.unit_tests.fixed_point_position as ut_fixed_point_position
import metalibm_hw_blocks.unit_tests.ut_special_values as ut_special_values
import metalibm_hw_blocks.unit_tests.min_max_select as ut_min_max_select
import metalibm_hw_blocks.unit_tests.unify_pipeline as ut_unify_pipeline
import metalibm_hw_blocks.unit_tests.sub_components as ut_sub_components
import metalibm_hw_blocks.unit_tests.hw_shifts as ut_hw_shifts
import metalibm_hw_blocks.unit_tests.cast_constant as ut_cast_constant

unit_test_list = [
  UnitTestScheme(
    "basic min max select entity",
    ut_min_max_select,
    [{}]
  ),
  UnitTestScheme(
    "basic special values arithmetic",
    ut_special_values,
    [{}]
  ),
  UnitTestScheme(
    "basic fixed-point legalization pass",
    ut_adaptative_size,
    [{}]
  ),
  UnitTestScheme(
    "basic fixed-point position ",
    ut_fixed_point_position,
    [{"width": 17}, {"width": 32}, {"width": 63}]
  ),
  UnitTestScheme(
    "basic message report test",
    ut_report_test,
    [{}]
  ),
  UnitTestScheme(
    "basic range evaluate test",
    ut_range_eval,
    [{}]
  ),
  UnitTestScheme(
    "pipeline unification pass test",
    ut_unify_pipeline,
    [{}]
  ),
  UnitTestScheme(
    "sub-component instanciation",
    ut_sub_components,
    [{}]
  ),
  UnitTestScheme(
    "dynamic and static shifts generation",
    ut_hw_shifts,
    [{}]
  ),
  UnitTestScheme(
    "cast constant test with CodeExpression memoization",
    ut_hw_shifts,
    [{}]
  ),
]

## Command line action to set break on error in load module
class ListUnitTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  unit_test_list:
          print(test.get_tag_title())
        exit(0)


# generate list of test object from string
# of comma separated test's tag
def parse_unit_test_list(test_list):
  test_tags = test_list.split(",")
  return [unit_test_tag_map[tag] for tag in test_tags]


# TODO: factorize / encapsulate in object/function

# filling unit-test tag map
unit_test_tag_map = {}
for test in unit_test_list:
  unit_test_tag_map[test.get_tag_title()] = test

arg_parser = argparse.ArgumentParser(" Metalibm rtl unit tests")
arg_parser.add_argument("--debug", dest = "debug", action = "store_const", 
                        default = False, const = True, 
                        help = "enable debug mode")
# listing available tests
arg_parser.add_argument("--list", action = ListUnitTestAction, help = "list available unit tests", nargs = 0) 
# select list of tests to be executed
arg_parser.add_argument("--execute", dest = "test_list", type = parse_unit_test_list, default = unit_test_list, help = "list of comma separated test to be executed") 


args = arg_parser.parse_args(sys.argv[1:])

success = True
debug_flag = args.debug

# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

for test_scheme in args.test_list:
  test_result = test_scheme.perform_all_test(debug = debug_flag)
  result_details.append(test_result)
  if not test_result.get_result(): 
    success = False

# Printing test summary for new scheme
for result in result_details:
  print(result.get_details())

if success:
  print("OVERALL SUCCESS")
  exit(0)
else:
  print("OVERALL FAILURE")
  exit(1)
