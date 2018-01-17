# -*- coding: utf-8 -*-
""" Software source code unit testing """

###############################################################################
# This file is part of New Metalibm tool
# Copyrights Nicolas Brunie (2017-)
# All rights reserved
# created:          Jul  9th, 2017
# last-modified:    Jul  9th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
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

import metalibm_functions.unit_tests.new_arg_template as ut_new_arg_template
import metalibm_functions.unit_tests.block_lzcnt as ut_block_lzcnt
import metalibm_functions.unit_tests.fixed_point as ut_fixed_point
import metalibm_functions.unit_tests.function_emulate  as ut_function_emulate
import metalibm_functions.unit_tests.function_formats as ut_function_formats
import metalibm_functions.unit_tests.gappa_code as ut_gappa_code
import metalibm_functions.unit_tests.loop_operation as ut_loop_operation
import metalibm_functions.unit_tests.opencl_code as ut_opencl_code
import metalibm_functions.unit_tests.pointer_manipulation as ut_pointer_manipulation
import metalibm_functions.unit_tests.static_vectorization as ut_static_vectorization
import metalibm_functions.unit_tests.vector_code as ut_vector_code
import metalibm_functions.unit_tests.call_externalization as ut_call_externalization
import metalibm_functions.unit_tests.auto_test as ut_auto_test
import metalibm_functions.unit_tests.m128_conversion as ut_m128_conversion
import metalibm_functions.unit_tests.new_table as ut_new_table
import metalibm_functions.unit_tests.multi_ary_function as ut_multi_ary_function
import metalibm_functions.unit_tests.entity_pass as ut_entity_pass
import metalibm_functions.unit_tests.implicit_interval_eval as ut_implicit_interval_eval

unit_test_list = [
  UnitTestScheme(
    "implicit interval eval test",
    ut_implicit_interval_eval,
    [{}]
  ),
  UnitTestScheme(
    "basic new arg template test",
    ut_new_arg_template,
    [{"target": target_instanciate("x86_avx2")}]
  ),
  UnitTestScheme(
    "basic block LZCNT test",
    ut_block_lzcnt,
    [
      {"precision": ML_Int32, "auto_test_execute": 100}, 
      # {"precision": ML_Int64, "auto_test_execute": 100}, 
    ]
  ),
  UnitTestScheme(
    "basic fixed-point",
    ut_fixed_point,
    [{"target": target_instanciate("fixed_point")}]
  ),
  UnitTestScheme(
    "basic function emulation test",
    ut_function_emulate,
    [{"target": target_instanciate("mpfr_backend")}]
  ),
  UnitTestScheme(
    "basic function format test",
    ut_function_formats,
    [{"target": target_instanciate("mpfr_backend")}]
  ),
  UnitTestScheme(
    "basic gappa code generation test",
    ut_gappa_code,
    [{}]
  ),
  UnitTestScheme(
    "basic loop operation support test",
    ut_loop_operation,
    [{}]
  ),
  UnitTestScheme(
    "basic opencl code generation test",
    ut_opencl_code,
    [{}]
  ),
  UnitTestScheme(
    "basic pointer manipulation test",
    ut_pointer_manipulation,
    [{}]
  ),
  UnitTestScheme(
    "basic static vectorization test",
    ut_static_vectorization,
    [{"target": target_instanciate("vector")}]
  ),
  UnitTestScheme(
    "basic vector code generation test",
    ut_vector_code,
    [{"target": target_instanciate("vector")}]
  ),
  UnitTestScheme(
    "basic call externalization test",
    ut_call_externalization,
    [{}]
  ),
  UnitTestScheme(
    "basic auto test",
    ut_auto_test,
    [{"auto_test": 10}]
  ),
  UnitTestScheme(
    "m128 conversion test",
    ut_m128_conversion,
    [{"pre_gen_passes": ["m128_promotion"], "target": target_instanciate("x86_avx2"), "vector_size": 4, "auto_test_execute": 100}],
  ),
  UnitTestScheme(
    "new table test",
    ut_new_table,
    [{"auto_test_range": Interval(0, 100), "precision": ML_Int32, "auto_test_execute": 10}],
  ),
  UnitTestScheme(
    "perf bench test",
    ut_new_table,
    [{"bench_range": Interval(0, 100), "precision": ML_Int32, "bench_execute": 100, "target": target_instanciate("x86")}],
  ),
  UnitTestScheme(
    "multi ary function",
    ut_multi_ary_function,
    [{"input_precisions": [ML_Int32, ML_Int32, ML_Int32], "precision": ML_Int32, "bench_execute": 100, "target": target_instanciate("x86")}],
  ),
  UnitTestScheme(
    "entity pass scheduling",
    ut_entity_pass,
    [{}],
  ),
]

# TODO: factorize / encapsulate in object/function
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


# filling unit-test tag map
unit_test_tag_map = {}
for test in unit_test_list:
  unit_test_tag_map[test.get_tag_title()] = test

arg_parser = argparse.ArgumentParser(" Metalibm unit tests")
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
  print result.get_details()

if success:
  print "OVERALL SUCCESS"
  exit(0)
else:
  print "OVERALL FAILURE"
  exit(1)
