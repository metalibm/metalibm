# -*- coding: utf-8 -*-

import argparse
import sys
import re

from sollya import Interval

# import meta-function script from metalibm_functions directory
import metalibm_functions.ml_log10
import metalibm_functions.ml_log1p
import metalibm_functions.ml_log2
import metalibm_functions.ml_log
import metalibm_functions.ml_exp
import metalibm_functions.ml_expm1
import metalibm_functions.ml_exp2_bis
import metalibm_functions.ml_cbrt
import metalibm_functions.ml_sqrt
import metalibm_functions.ml_isqrt
import metalibm_functions.ml_vectorizable_log
import metalibm_functions.ml_cosh
import metalibm_functions.ml_sinh
import metalibm_functions.ml_tanh
import metalibm_functions.ml_sincos
import metalibm_functions.ml_atan
import metalibm_functions.external_bench

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Int32
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.targets.intel.x86_processor import (
        X86_Processor, X86_SSE_Processor, X86_SSE2_Processor,
        X86_SSE3_Processor, X86_SSSE3_Processor, X86_SSE41_Processor,
        X86_AVX_Processor, X86_AVX2_Processor
        )
        

from metalibm_core.targets.intel.m128_promotion import Pass_M128_Promotion
from metalibm_core.targets.intel.m256_promotion import Pass_M256_Promotion
from metalibm_core.utility.ml_template import target_instanciate

from valid.test_utils import *

try:
    from metalibm_core.targets.kalray.k1b_processor import K1B_Processor
    k1b_defined = True
    k1b = K1B_Processor()
except ImportError:
    k1b_defined = False
    k1b = None

# target instanciation
x86_processor = X86_Processor()
x86_avx2_processor = X86_AVX2_Processor()
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
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "basic hyperbolic sine gen test",
    metalibm_functions.ml_sinh.ML_HyperbolicSine,
    [
        {"precision": ML_Binary32, "function_name": "my_sinhf",
         "auto_test_execute": 1000},
        {"precision": ML_Binary64, "function_name": "my_sinh",
         "auto_test_execute": 1000}
    ]
  ),
  NewSchemeTest(
    "basic hyperbolic tangent gen test",
    metalibm_functions.ml_tanh.ML_HyperbolicTangent,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "auto test hyperbolic cosine",
    metalibm_functions.ml_cosh.ML_HyperbolicCosine,
    [
        {"function_name": "my_cosh", "precision": ML_Binary32,
         "auto_test": 100, "auto_test_execute": 100},
        {"function_name": "my_cosh", "precision": ML_Binary64,
        "auto_test": 100, "auto_test_execute": 100},
    ]
  ),
  NewSchemeTest(
    "basic log test",
    metalibm_functions.ml_log.ML_Log,
    [
        {"precision": ML_Binary32, "function_name": "my_logf",
         "auto_test_execute": 1000},
        {"precision": ML_Binary64, "function_name": "my_log",
         "auto_test_execute": 1000}
    ]
  ),
  NewSchemeTest(
    "basic log1p test",
    metalibm_functions.ml_log1p.ML_Log1p,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "generic log2 test",
    metalibm_functions.ml_log2.ML_Log2,
    [
        {"precision": ML_Binary32},
        {"precision": ML_Binary64, "auto_test_execute": 10000}
    ]
  ),
  NewSchemeTest(
    "x86 log2 test",
    metalibm_functions.ml_log2.ML_Log2,
    [
        {"precision": ML_Binary32, "target": x86_processor,
         "auto_test_execute": 10000},
        {"precision": ML_Binary64, "target": x86_processor,
         "auto_test_execute": 10000}
    ]
  ),
  NewSchemeTest(
    "basic log10 test",
    metalibm_functions.ml_log10.ML_Log10,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "basic exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64}]
  ),
  NewSchemeTest(
    "auto execute exp test",
    metalibm_functions.ml_exp.ML_Exponential,
    [
        {"precision": ML_Binary32, "function_name": "my_exp", "auto_test": 100,
         "auto_test_execute": 1000},
        {"precision": ML_Binary64, "function_name": "my_exp", "auto_test": 100,
         "auto_test_execute": 1000},
    ]
  ), 
  NewSchemeTest(
    "auto execute exp2 test",
    metalibm_functions.ml_exp2_bis.ML_Exp2,
    [
        {"precision": ML_Binary32, "function_name": "my_exp2", "auto_test": 100,
        "auto_test_execute": 100},
        {"precision": ML_Binary64, "function_name": "my_exp2", "auto_test": 100,
         "auto_test_execute": 100},
    #{"precision": ML_Binary32, "target": k1b, "function_name": "my_exp2", "auto_test": 100, "auto_test_execute": 100},
    #{"precision": ML_Binary64, "target": k1b, "function_name": "my_exp2", "auto_test": 100, "auto_test_execute": 100},
    ]
  ), 
  NewSchemeTest(
    "basic cubic square test",
    metalibm_functions.ml_cbrt.ML_Cbrt,
    [
        #{"precision": ML_Binary32, "target" : k1b},
        #{"precision": ML_Binary64, "target" : k1b}
    ]
  ),
  NewSchemeTest(
    "basic square root test",
    metalibm_functions.ml_sqrt.ML_Sqrt,
    [
        #{"precision": ML_Binary32, "target" : k1b, "num_iter" : 1},
        #{"precision": ML_Binary64, "target" : k1b, "num_iter" : 1}
    ]
  ),
  NewSchemeTest(
    "auto execute sqrt test",
    metalibm_functions.ml_sqrt.ML_Sqrt,
    [
    #{"precision": ML_Binary32, "target": k1b, "function_name": "my_sqrt", "auto_test": 100, "auto_test_execute": 100, "num_iter" : 1},
    #{"precision": ML_Binary64, "target": k1b, "function_name": "my_sqrt", "auto_test": 100, "auto_test_execute": 100, "num_iter" : 1},
    ]
  ),
  NewSchemeTest(
    "basic inverse square root test",
    metalibm_functions.ml_isqrt.ML_Isqrt,
    [
        #{"precision": ML_Binary32, "target" : k1b, "num_iter" : 3}, {"precision": ML_Binary64, "target" : k1b, "num_iter" : 3}
    ]
  ),
  NewSchemeTest(
    "auto execute isqrt test",
    metalibm_functions.ml_isqrt.ML_Isqrt,
    [
    #{"precision": ML_Binary32, "target": k1b, "function_name": "my_isqrt", "auto_test": 100, "auto_test_execute": 100, "num_iter" : 3},
    #{"precision": ML_Binary64, "target": k1b, "function_name": "my_isqrt", "auto_test": 100, "auto_test_execute": 100, "num_iter" : 3},
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
    metalibm_functions.ml_atan.ML_Atan,
    [
        #{"precision": ML_Binary32, "target": k1b}, {"precision": ML_Binary64, "target": k1b}
    ]
  ),
  NewSchemeTest(
    "auto execute atan test",
    metalibm_functions.ml_atan.ML_Atan,
    [
        #{"precision": ML_Binary32, "target": k1b, "function_name": "my_atan", "auto_test_execute": 100, "auto_test_std" : True},
    # {"precision": ML_Binary64, "target": k1b, "function_name": "my_atan", "auto_test_execute": 100, "auto_test_std" : True},
    ]
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
    [{"precision": ML_Binary32, "vector_size": 2, "target": VectorBackend()}, ]
  ),
  NewSchemeTest(
    "external bench test",
    metalibm_functions.external_bench.ML_ExternalBench,
    [{"precision": ML_Binary32,
      "bench_function_name": "tanf",
      "target": target_instanciate("x86"),
      "input_precisions": [ML_Binary32],
      "bench_execute": 1000,
      "bench_test_range": Interval(-1, 1)
    }, ]
  ),
]

test_tag_map = {}
for test in new_scheme_function_list:
  test_tag_map[test.get_tag_title()] = test

## Command line action to set break on error in load module
class ListTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  new_scheme_function_list:
          print test.get_tag_title()
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




args = arg_parser.parse_args(sys.argv[1:])

success = True
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

for test_scheme in args.test_list:
  if re.search(args.match_regex, test_scheme.get_tag_title()) != None:
    test_result = test_scheme.perform_all_test(debug = args.debug)
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
