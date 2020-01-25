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
# created:          Jul  9th, 2017
# last-modified:    Mar  7th, 2018
#
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
# description: software source code unit testing
###############################################################################

import sys
import argparse

from sollya import Interval, SollyaObject

S2 = SollyaObject(2)

from metalibm_core.targets import *
import metalibm_core.code_generation.mpfr_backend

from metalibm_core.utility.ml_template import target_instanciate
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Int16, ML_Int64,
    ML_Binary32, ML_Binary64,
    ML_SingleSingle, ML_DoubleDouble,
)
from metalibm_core.core.precisions import (
    ML_CorrectlyRounded, ML_Faithful, dar, daa
)

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
import metalibm_functions.unit_tests.m128_boolean as ut_m128_boolean
import metalibm_functions.unit_tests.m128_debug as ut_m128_debug
import metalibm_functions.unit_tests.new_table as ut_new_table
import metalibm_functions.unit_tests.multi_ary_function as ut_multi_ary_function
import metalibm_functions.unit_tests.entity_pass as ut_entity_pass
import metalibm_functions.unit_tests.implicit_interval_eval as ut_implicit_interval_eval
import metalibm_functions.unit_tests.legalize_sqrt as ut_legalize_sqrt
import metalibm_functions.unit_tests.accuracies as ut_accuracies
import metalibm_functions.unit_tests.legalize_reciprocal_seed as ut_legalize_reciprocal_seed
import metalibm_functions.unit_tests.fuse_fma as ut_fuse_fma
import metalibm_functions.unit_tests.llvm_code as ut_llvm_code
import metalibm_functions.unit_tests.multi_precision as ut_multi_precision
import metalibm_functions.unit_tests.function_ptr as ut_function_ptr
import metalibm_functions.unit_tests.multi_precision_vectorization as ut_mp_vectorization
import metalibm_functions.unit_tests.embedded_bin as ut_embedded_bin
import metalibm_functions.unit_tests.bfloat16 as ut_bfloat16
import metalibm_functions.unit_tests.ut_eval_error as ut_eval_error
import metalibm_functions.unit_tests.special_values as ut_special_values

unit_test_list = [
  UnitTestScheme(
    "legalize_reciprocal_seed",
    ut_legalize_reciprocal_seed,
    [{"auto_test": 1024, "execute_trigger": True, "accuracy": dar(S2**-6)}]
  ),
  UnitTestScheme(
    "fuse_fma pass test",
    ut_fuse_fma,
    [{"passes": ["beforecodegen:fuse_fma"]}]
  ),
  UnitTestScheme(
    "implicit interval eval test",
    ut_implicit_interval_eval,
    [{}]
  ),
  UnitTestScheme(
    "legalization of InvSquareRoot operation",
    ut_legalize_sqrt,
    [{"auto_test": 100, "execute": True}],
  ),
  UnitTestScheme(
    "accuracy option",
    ut_legalize_sqrt,
    [
        {"auto_test": 100, "execute": True, "accuracy": ML_CorrectlyRounded},
        {"auto_test": 100, "execute": True, "accuracy": ML_Faithful},
        {"auto_test": 100, "execute": True, "accuracy": dar(S2**-8)},
        {"auto_test": 100, "execute": True, "accuracy": daa(S2**-8)},
    ],
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
    [{"passes": ["beforecodegen:m128_promotion"], "target": target_instanciate("x86_avx2"), "vector_size": 4, "auto_test_execute": 100}],
  ),
  UnitTestScheme(
    "m128 boolean test",
    ut_m128_boolean,
    [{"passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization", "beforecodegen:m128_promotion"], "target": target_instanciate("x86_sse2"), "vector_size": 4, "auto_test_execute": 100, "precision": ML_Int32}],
  ),
  UnitTestScheme(
    "m128 debug test",
    ut_m128_debug,
    [{"passes": ["beforecodegen:m128_promotion"], "target": target_instanciate("x86_avx2"), "vector_size": 4, "auto_test_execute": 10, "auto_test": 10, "precision": ML_Binary32, "debug": True}],
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
  UnitTestScheme(
    "llvm code generation test",
    ut_llvm_code,
    [{"passes": ["beforecodegen:gen_basic_block", "beforecodegen:ssa_translation"]}]
  ),
  UnitTestScheme(
    "multi precision expansion",
    ut_multi_precision,
    [
        {
            "precision": ML_Binary32, "input_precisions": [ML_Binary32]*2,
            "arity": 2, "passes": ["beforecodegen:expand_multi_precision"]
        },
        {
            "precision": ML_Binary64, "passes": ["beforecodegen:expand_multi_precision"],
            "arity": 2, "input_precisions": [ML_Binary64]*2,
        },
    ]
  ),
  UnitTestScheme(
    "function pointer argument",
    ut_function_ptr,
    [{}],
  ),
  UnitTestScheme(
    "multi-precision expansion",
    ut_mp_vectorization,
    [
        {"precision": ML_DoubleDouble, "passes": ["start:basic_legalization", "start:expand_multi_precision"]},
        {"precision": ML_SingleSingle, "passes": ["start:basic_legalization", "start:expand_multi_precision"]},
    ],
  ),
  UnitTestScheme(
    "multi-precision vectorization",
    ut_mp_vectorization,
    [
        {"precision": ML_DoubleDouble, "vector_size": 4, "target": target_instanciate("vector"), "passes": ["start:basic_legalization", "start:expand_multi_precision"]},
        {"precision": ML_SingleSingle, "vector_size": 4, "target": target_instanciate("vector"), "passes": ["start:basic_legalization", "start:expand_multi_precision"]},
    ],
  ),
  UnitTestScheme(
    "embedded binary",
    ut_embedded_bin,
    [
        {"embedded_bin": True},
        {"embedded_bin": False},
    ],
  ),
  UnitTestScheme(
    "bfloat16",
    ut_bfloat16,
    [{}]
  ),
  UnitTestScheme(
    "runtime error eval",
    ut_eval_error,
    [{"target": target_instanciate("fixed_point")}]
  ),
  UnitTestScheme(
    "special values",
    ut_special_values,
    [{}]
  )
]

# TODO: factorize / encapsulate in object/function
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
  print(result.get_details())

if success:
  print("OVERALL SUCCESS")
  exit(0)
else:
  print("OVERALL FAILURE")
  exit(1)
