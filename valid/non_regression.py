# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/kalray/metalibm)

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


import argparse
import sys
import re
import os

import sollya
from sollya import Interval, SollyaObject

# numeric value 2 as a sollya object
S2 = SollyaObject(2)

# import meta-function script from metalibm_functions directory
import metalibm_functions.ml_log1p
import metalibm_functions.ml_exp
import metalibm_functions.ml_expm1
import metalibm_functions.ml_exp2
import metalibm_functions.ml_cbrt
import metalibm_functions.ml_sqrt
import metalibm_functions.ml_isqrt
import metalibm_functions.ml_vectorizable_log
import metalibm_functions.ml_cosh
import metalibm_functions.ml_sinh
import metalibm_functions.ml_sincos
import metalibm_functions.ml_atan
import metalibm_functions.external_bench
import metalibm_functions.ml_tanh
import metalibm_functions.ml_div
import metalibm_functions.rootn
import metalibm_functions.remquo

import metalibm_functions.softmax
import metalibm_functions.vectorial_function
import metalibm_functions.erf

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Int32, ML_Int64
from metalibm_core.core.precisions import dar, ML_CorrectlyRounded

from metalibm_core.code_generation.code_constant import LLVM_IR_Code
from metalibm_core.code_generation.generator_utility import LibraryDependency

from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.targets.common.llvm_ir import LLVMBackend

from metalibm_core.targets.intel.x86_processor import (
        X86_Processor, X86_SSE_Processor, X86_SSE2_Processor,
        X86_SSE3_Processor, X86_SSSE3_Processor, X86_SSE41_Processor,
        X86_AVX_Processor, X86_AVX2_Processor
        )


from metalibm_core.targets.intel.m128_promotion import Pass_M128_Promotion
from metalibm_core.targets.intel.m256_promotion import Pass_M256_Promotion
from metalibm_core.utility.ml_template import (
    target_instanciate, VerboseAction, ExitOnErrorAction)

from valid.test_utils import *

try:
    from metalibm_core.targets.kalray.k1b_processor import K1B_Processor
    k1b_defined = True
    k1b = K1B_Processor()
except ImportError:
    k1b_defined = False
    k1b = None


# default directory to load AXF file from
AXF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "metalibm_functions", "axf")

# target instanciation
x86_processor = X86_Processor.get_target_instance()
x86_avx2_processor = X86_AVX2_Processor.get_target_instance()
avx2_pass_m128_promotion = Pass_M128_Promotion(x86_avx2_processor)
avx2_pass_m256_promotion = Pass_M256_Promotion(x86_avx2_processor)

# list of non-regression tests
# details on NewSchemeTest object can be found in valid.test_utils module
#   Each object requires a title, a function constructor and a list
#   of test cases (each is a dictionnary of parameters -> values)
new_scheme_function_list = [
  NewSchemeTest(
    "basic hyperbolic cosine gen test",
    metalibm_functions.ml_cosh.ML_HyperbolicCosine,
    [{"precision": ML_Binary32, "auto_test": 128, "execute_trigger": True},
     {"precision": ML_Binary64, "auto_test": 128, "execute_trigger": True}
     ]
  ),
  NewSchemeTest(
    "vector hyperbolic cosine gen test",
    metalibm_functions.ml_cosh.ML_HyperbolicCosine,
    [{"precision": ML_Binary32, "vector_size": 4, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance(),  "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
    {"precision": ML_Binary32, "vector_size": 8, "auto_test": 128,
    "execute_trigger": True, "target": VectorBackend.get_target_instance(),  "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
     ]
  ),
  NewSchemeTest(
    "basic hyperbolic sine gen test",
    metalibm_functions.ml_sinh.ML_HyperbolicSine,
    [
        {"precision": ML_Binary32, "function_name": "my_sinhf"}, # disabled valid test
        {"precision": ML_Binary64, "function_name": "my_sinh"}, # disabled valid test
    ]
  ),
  NewSchemeTest(
    # test also AXF export
    "basic hyperbolic tangent gen test",
    metalibm_functions.ml_tanh.ML_HyperbolicTangent,
    [{"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True,
      "dump_axf_approx": "./dump-tanh-fp32.axf"},
    {"precision": ML_Binary64, "dump_axf_approx": "./dump-tanh-fp64.axf"}]
  ),
  NewSchemeTest(
    # test also AXF export
    "basic error function test",
    metalibm_functions.erf.ML_Erf,
    [{"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True,
      "dump_axf_approx": "./dump-erf-fp32.axf"},
     {"precision": ML_Binary64, "auto_test": 1000, "execute_trigger": True,
      "dump_axf_approx": "./dump-erf-fp64.axf"}]
  ),
  NewSchemeTest(
    "tanh axf load test",
    metalibm_functions.ml_tanh.ML_HyperbolicTangent,
    [{"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True,
      "load_axf_approx": os.path.join(AXF_DIR, "tanh-fp32.json.axf")},
    {"precision": ML_Binary64, "auto_test": 1000, "execute_trigger": True,
     "load_axf_approx": os.path.join(AXF_DIR, "tanh-fp64.json.axf")}]
  ),
  NewSchemeTest(
    "erf axf load test",
    metalibm_functions.erf.ML_Erf,
    [{"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True,
      "load_axf_approx": os.path.join(AXF_DIR, "erf-fp32.json.axf")},
     {"precision": ML_Binary64, "auto_test": 1000, "execute_trigger": True,
      "load_axf_approx": os.path.join(AXF_DIR, "erf-fp64.json.axf") }]
  ),
  NewSchemeTest(
    "atan axf load test",
    metalibm_functions.ml_atan.MetaAtan,
    [{"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True,
      "load_axf_approx": os.path.join(AXF_DIR, "atan-fp32.json.axf")},
    {"precision": ML_Binary64, "auto_test": 1000, "execute_trigger": True,
     "load_axf_approx": os.path.join(AXF_DIR, "atan-fp64.json.axf")}]
  ),

  NewSchemeTest(
    "auto test hyperbolic cosine",
    metalibm_functions.ml_cosh.ML_HyperbolicCosine,
    [
        {"function_name": "my_cosh", "precision": ML_Binary32,
         "auto_test": 100, "auto_test": 100, "execute_trigger": True},
        {"function_name": "my_cosh", "precision": ML_Binary64,
        "auto_test": 100, "auto_test": 100, "execute_trigger": True},
    ]
  ),
  NewSchemeTest(
    "basic log test",
    metalibm_functions.generic_log.ML_GenericLog,
    [
        {"precision": ML_Binary32, "function_name": "my_logf", "basis": sollya.exp(1),
         "auto_test": 1000, "execute_trigger": True, "expected_to_fail": True},
        {"precision": ML_Binary64, "function_name": "my_log", "basis": sollya.exp(1),
         "auto_test": 1000, "execute_trigger": True, "expected_to_fail": True}
    ]
  ),
  NewSchemeTest(
    "basic log1p test",
    metalibm_functions.ml_log1p.ML_Log1p,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "generic log2 test",
    metalibm_functions.generic_log.ML_GenericLog,
    [
        {"precision": ML_Binary32, "basis": 2},
        {"precision": ML_Binary64, "basis": 2, "auto_test": 10000, "execute_trigger": True,
         "expected_to_fail": True}
    ]
  ),
  NewSchemeTest(
    "x86 log2 test",
    metalibm_functions.generic_log.ML_GenericLog,
    [
        {"precision": ML_Binary32, "basis": 2, "target": x86_processor},
        {"precision": ML_Binary64, "basis": 2, "target": x86_processor},
    ]
  ),
  NewSchemeTest(
    "basic log10 test",
    metalibm_functions.generic_log.ML_GenericLog,
    [{"precision": ML_Binary32, "basis": 10},
     {"precision": ML_Binary64, "basis": 10}]
  ),
  NewSchemeTest(
    "basic exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "llvm-ir exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [
        {"precision": ML_Binary32, "target": LLVMBackend.get_target_instance(), "language": LLVM_IR_Code, "extra_passes": ["start:instantiate_abstract_prec", "start:instantiate_prec", "start:basic_legalization"]} ,
    {"precision": ML_Binary64, "target": LLVMBackend.get_target_instance(), "language": LLVM_IR_Code, "extra_passes": ["start:instantiate_abstract_prec", "start:instantiate_prec", "start:basic_legalization"]}]
  ),
  NewSchemeTest(
    "exp value test",
    metalibm_functions.ml_exp.ML_Exponential,
    [{"precision": ML_Binary32, "value_test": [(1.0,),(0.0,)], "execute_trigger": True}]
  ),
  NewSchemeTest(
    "auto execute exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [
        {"precision": ML_Binary32, "function_name": "my_exp",
         "auto_test": 1000, "execute_trigger": True},
        {"precision": ML_Binary64, "function_name": "my_exp",
         "auto_test": 1000, "execute_trigger": True},
        {"precision": ML_Binary32, "function_name": "my_exp",
         "target": x86_avx2_processor,
         "auto_test": 1000, "execute_trigger": True},
    ]
  ),
  NewSchemeTest(
    "auto execute exp2 test",
    metalibm_functions.ml_exp2.ML_Exp2,
    [
        {"precision": ML_Binary32, "function_name": "my_exp2", "auto_test": 100,
        "auto_test_execute": 100},
        {"precision": ML_Binary64, "function_name": "my_exp2", "auto_test": 100,
         "auto_test_execute": 100},
    ]
  ),
  NewSchemeTest(
    "basic cubic square test",
    metalibm_functions.ml_cbrt.ML_Cbrt,
    [
    ]
  ),
  NewSchemeTest(
    "basic square root test",
    metalibm_functions.ml_sqrt.MetalibmSqrt,
    [
    ]
  ),
  NewSchemeTest(
    "auto execute sqrt test",
    metalibm_functions.ml_sqrt.MetalibmSqrt,
    [
    ]
  ),
  NewSchemeTest(
    "basic inverse square root test",
    metalibm_functions.ml_isqrt.ML_Isqrt,
    [
    ]
  ),
  NewSchemeTest(
    "auto execute isqrt test",
    metalibm_functions.ml_isqrt.ML_Isqrt,
    [
    ]
  ),
  NewSchemeTest(
    "basic cosine test",
    metalibm_functions.ml_sincos.ML_SinCos,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "basic sine test",
    metalibm_functions.ml_sincos.ML_SinCos,
    [{"precision": ML_Binary32, "sin_output" : True}, {"precision": ML_Binary64, "sin_output" : True}]
  ),
  NewSchemeTest(
    "basic arctangent test",
    metalibm_functions.ml_atan.MetaAtan,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "basic vectorizable_log tests",
    metalibm_functions.ml_vectorizable_log.ML_Log,
    [
      {"precision": ML_Binary32},
      # disabled pending bugfix
      #{"precision": ML_Binary64},
      {
        "precision": ML_Binary32,
        "target": x86_avx2_processor,
        "vector-size": 4,
        "pre-gen-pass": avx2_pass_m128_promotion,
      },
      {
        "precision": ML_Binary32,
        "target": x86_avx2_processor,
        "vector-size": 8,
        "pre-gen-pass": avx2_pass_m256_promotion,
      },
    ]
  ),
  NewSchemeTest(
    "vector exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [
        {"precision": ML_Binary32, "vector_size": 2, "target": VectorBackend.get_target_instance(),
         "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]
        },
    ]
  ),
  NewSchemeTest(
    "external bench test",
    metalibm_functions.external_bench.ML_ExternalBench,
    [{"precision": ML_Binary32,
      "bench_function_name": "tanf",
      "target": target_instanciate("x86"),
      "input_precisions": [ML_Binary32],
      "bench_test_number": 1000,
      "execute_trigger": True,
      "headers": [],
      "libraries": [LibraryDependency("math.h","-lm")],
      "bench_test_range": [Interval(-1, 1)]
    },
    # testing vector function codegen
   {"precision": ML_Binary32,
     "bench_function_name": "dummy_v4",
     "target": target_instanciate("x86"),
     "input_precisions": [ML_Binary32],
     "vector_size": 4,
     "function_input_vector_size": 4,
     "bench_test_number": 1000,
     "bench_test_range": [Interval(-1, 1)]
   },
    ]
  ),
  NewSchemeTest(
    "basic division test",
    metalibm_functions.ml_div.ML_Division,
    [
        {"precision": ML_Binary32, "auto_test": 1000, "execute_trigger": True, "accuracy": ML_CorrectlyRounded},
        {"precision": ML_Binary64, "auto_test": 1000, "execute_trigger": True, "accuracy": ML_CorrectlyRounded},
    ],
  ),
  NewSchemeTest(
    "basic rootn test",
    metalibm_functions.rootn.MetaRootN,
    [
        {"precision": ML_Binary32, "input_precisions": [ML_Binary32, ML_Int32], "auto_test_range": [Interval(-S2**127, S2**127), Interval(0, 255)], "auto_test": 1000, "execute_trigger": True, "accuracy": dar(S2**-22)},
        {"precision": ML_Binary64, "input_precisions": [ML_Binary64, ML_Int64], "auto_test_range": [Interval(-S2**1023, S2**1023), Interval(0, 255)], "auto_test": 1000, "execute_trigger": True, "accuracy": dar(S2**-50)},
    ],
  ),
  NewSchemeTest(
    "basic remquo test",
    metalibm_functions.remquo.MetaRemQuo,
    [
        # 64-bit
        # remainder mode
        {"precision": ML_Binary64, "input_precisions": [ML_Binary64, ML_Binary64], "auto_test_std": True, "execute_trigger": True, "accuracy": ML_CorrectlyRounded, "mode": "remainder"},
        # quotient mode
        {"precision": ML_Binary64, "input_precisions": [ML_Binary64, ML_Binary64], "auto_test_std": True, "execute_trigger": True, "accuracy": ML_CorrectlyRounded, "mode": "quotient"},
        # full mode: codegen only
        {"precision": ML_Binary64, "input_precisions": [ML_Binary64, ML_Binary64], "accuracy": ML_CorrectlyRounded, "mode": "full"},

        # 32-bit
        {"precision": ML_Binary32, "input_precisions": [ML_Binary32, ML_Binary32], "accuracy": ML_CorrectlyRounded, "mode": "full"},
        {"precision": ML_Binary32, "input_precisions": [ML_Binary32, ML_Binary32], "auto_test": 100, "execute_trigger": True, "accuracy": ML_CorrectlyRounded, "mode": "quotient"},
        {"precision": ML_Binary32, "input_precisions": [ML_Binary32, ML_Binary32], "auto_test": 100, "execute_trigger": True, "accuracy": ML_CorrectlyRounded, "mode": "remainder"},
    ],
  ),
  NewSchemeTest(
    "basic softmax test",
    metalibm_functions.softmax.ML_SoftMax,
    [
        {"precision": ML_Binary32, "auto_test": 10, "execute_trigger": True,
         "accuracy": dar(S2**-22)},
    ],
  ),
  NewSchemeTest(
    "basic vector function test",
    metalibm_functions.vectorial_function.ML_VectorialFunction,
    [
        {"precision": ML_Binary32, "auto_test": 10, "execute_trigger": True,  "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
        {"precision": ML_Binary32, "use_libm_function": "expf", "auto_test": 10, "execute_trigger": True, "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
        {"precision": ML_Binary32, "multi_elt_num": 4, "target": VectorBackend.get_target_instance(), "auto_test": 10, "index_test_range": [16, 32], "execute_trigger": True,  "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
        {"precision": ML_Binary64, "multi_elt_num": 4, "target": VectorBackend.get_target_instance(), "auto_test": 10,  "index_test_range": [16, 32], "execute_trigger": True, "expected_to_fail": True,  "passes": ["beforecodegen:virtual_vector_bool_legalization", "beforecodegen:vector_mask_test_legalization"]},
    ],
  ),
  NewSchemeTest(
    "vector exp with sub_vector_size 1",
    metalibm_functions.ml_exp.ML_Exponential,
    [
    # 2-element vectors
    {"precision": ML_Binary32, "vector_size": 2, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    {"precision": ML_Binary64, "vector_size": 2, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    # 4-element vectors
    {"precision": ML_Binary32, "vector_size": 4, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    {"precision": ML_Binary64, "vector_size": 4, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    # 8-element vectors
    {"precision": ML_Binary32, "vector_size": 8, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    {"precision": ML_Binary64, "vector_size": 8, "sub_vector_size": 1, "auto_test": 128,
      "execute_trigger": True, "target": VectorBackend.get_target_instance() },
    ]
  )
]

test_tag_map = {}
for test in new_scheme_function_list:
  test_tag_map[test.get_tag_title()] = test

## Command line action to set break on error in load module
class ListTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  new_scheme_function_list:
          print(test.get_tag_title())
        exit(0)

# generate list of test object from string
# of comma separated test's tag
def parse_test_list(test_list):
  test_tags = test_list.split(",")
  return [test_tag_map[tag] for tag in test_tags]

arg_parser = argparse.ArgumentParser(" Metalibm non-regression tests")
# enable debug mode
arg_parser.add_argument("--debug", dest = "debug", action = "store_const",
                        default = False, const = True,
                        help = "enable debug mode")
# listing available tests
arg_parser.add_argument("--list", action = ListTestAction, help = "list available test", nargs = 0)

# select list of tests to be executed
arg_parser.add_argument("--execute", dest = "test_list", type = parse_test_list, default = new_scheme_function_list, help = "list of comma separated test to be executed")

arg_parser.add_argument("--match", dest = "match_regex", type = str, default = ".*", help = "list of comma separated match regexp to be used for test selection")
arg_parser.add_argument(
    "--exit-on-error", dest="exit_on_error",
    action=ExitOnErrorAction, const=True,
    default=False,
    nargs=0,
    help="convert Fatal error to sys exit rather than exception")


arg_parser.add_argument(
    "--verbose", dest="verbose_enable", action=VerboseAction,
    const=True, default=False,
    help="enable Verbose log level")


args = arg_parser.parse_args(sys.argv[1:])

success = True
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

for test_scheme in args.test_list:
  if re.search(args.match_regex, test_scheme.get_tag_title()) != None:
    print("executing test {}".format(test_scheme.title))
    test_result = test_scheme.perform_all_test(debug = args.debug)
    result_details.append(test_result)
    if not test_result.get_result():
      success = False

unexpected_failure_count = 0

# Printing test summary for new scheme
for result in result_details:
  print(result.get_details())
  unexpected_failure_count += result.unexpected_count

print(" {} unexpected failure(s)".format(unexpected_failure_count))

if success:
  print("OVERALL SUCCESS")
  exit(0)
else:
  print("OVERALL FAILURE")
  exit(1)
