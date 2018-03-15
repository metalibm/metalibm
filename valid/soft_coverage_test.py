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
import metalibm_functions.ml_sincos
import metalibm_functions.ml_atan
import metalibm_functions.external_bench
import metalibm_functions.ml_tanh

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Int32
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.targets.intel.x86_processor import (
        X86_Processor, X86_SSE_Processor, X86_SSE2_Processor,
        X86_SSE3_Processor, X86_SSSE3_Processor, X86_SSE41_Processor,
        X86_AVX_Processor, X86_AVX2_Processor
        )
from metalibm_core.code_generation.generic_processor import GenericProcessor


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

NUM_AUTO_TEST = 1024 # to be divisible by standard vector length

FUNCTION_LIST = [
    metalibm_functions.ml_cosh.ML_HyperbolicCosine,
    metalibm_functions.ml_sinh.ML_HyperbolicSine,
]
_ = [
    metalibm_functions.ml_tanh.ML_HyperbolicTangent,
    metalibm_functions.ml_exp.ML_Exponential,
    metalibm_functions.ml_log.ML_Log,
    metalibm_functions.ml_log1p.ML_Log1p,
    metalibm_functions.ml_log2.ML_Log2,
    metalibm_functions.ml_log10.ML_Log10,
    metalibm_functions.ml_exp2_bis.ML_Exp2,
    metalibm_functions.ml_cbrt.ML_Cbrt,
    metalibm_functions.ml_sqrt.ML_Sqrt,
    metalibm_functions.ml_isqrt.ML_Isqrt,
    metalibm_functions.ml_sincos.ML_SinCos,
    metalibm_functions.ml_atan.ML_Atan,
    metalibm_functions.ml_vectorizable_log.ML_Log,
]

global_test_list = []

# list of all possible test for a single function
test_list = []
for scalar_target in [GenericProcessor(), X86_Processor(), X86_AVX2_Processor()]:
    for precision in [ML_Binary32, ML_Binary64]:
        test_list.append({
            "precision": precision,
            "target": scalar_target,
        })
for vector_target in [X86_AVX2_Processor(), VectorBackend()]:
    for precision in [ML_Binary32, ML_Binary64]:
        for vector_size in [4, 8]:
            test_list.append({
                "precision": precision,
                "target": vector_target,
                "vector_size": vector_size
            })

for function in FUNCTION_LIST:
    test_case = NewSchemeTest(
        "test %s" % function,
        function,
        test_list
    )
    global_test_list.append(test_case)

arg_parser = argparse.ArgumentParser(" Metalibm non-regression tests")
# enable debug mode
arg_parser.add_argument("--debug", dest = "debug", action = "store_const",
                        default = False, const = True,
                        help = "enable debug mode")
arg_parser.add_argument("--report-only", dest = "report_only", action = "store_const",
                        default = False, const = True,
                        help = "limit display to final report")
arg_parser.add_argument("--output", dest = "output", action = "store",
                        default = "report.html",
                        help = "define output file")

args = arg_parser.parse_args(sys.argv[1:])

success = True
success_count = 0
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

NUM_TESTS = reduce(lambda acc, v: acc + v.num_test, global_test_list, 0)

RESULT_MAP = {}

for test_scheme in global_test_list:
    test_results = test_scheme.perform_all_test_no_reduce(debug=args.debug)
    RESULT_MAP[test_scheme] = test_results

for test_scheme in RESULT_MAP:
    result_list = RESULT_MAP[test_scheme]
    test_result = test_scheme.reduce_test_result(result_list)
    success_count += test_scheme.get_success_count(result_list)
    result_details.append(test_result)
    if not test_result.get_result():
        success = False

if not args.report_only:
    # Printing test summary for new scheme
    for result in result_details:
        print(result.get_details())

OUTPUT_FILE = open(args.output, "w")

def print_report(msg):
    OUTPUT_FILE.write(msg)


print_report("<html><body><div>")
print_report("<p><ul>\n")
for index, test_case in enumerate(test_list):
    nice_str = "; ".join("{}: {}".format(option, str(test_case[option])) for option in test_case)
    print_report("<li>({}): {}</li>\n".format(index, nice_str))

print_report("</ul></p>\n\n")
header = "<table border='1'>"
header += "<th>{:10}</th>".format("function "[:10])
header += "".join(["\t\t<th> {:2} </th>\n".format(i) for i in range(len(test_list))])
header += "\n"
#header += ("----------"+ "|----" * len(test_list))
#header += "|"
print_report(header)

def color_cell(msg, color="red", markup="td", indent="\t\t"):
    return """{indent}<{markup} style="color:{color}">{msg}</{markup}>\n""".format(
        indent=indent, color=color, msg=msg, markup=markup
    )

# report dusplay
for test_scheme in RESULT_MAP:
    name = test_scheme.ctor.get_default_args().function_name
    msg = "<tr>\n\t\t<td>{:10}</td>\n".format(name[:10])
    for result in RESULT_MAP[test_scheme]:
        if result.get_result():
            msg += color_cell(" OK ", "orange")
        else:
            msg += color_cell(" KO ", "red")
    msg += "</tr>"
    print_report(msg)

print_report("</table>")
print_report("<p>\n")
print_report("Total {}/{}/{} test(s)".format(
            color_cell("+%d" % success_count, color="green", markup="span"),
            color_cell("-%d" % (NUM_TESTS - success_count), color="red", markup="span"),
            NUM_TESTS)
     )
print_report("      Success: {:.1f}%".format(float(success_count) / NUM_TESTS * 100))

print_report("</p>\n<p>\n")

if success:
    print_report("OVERALL SUCCESS")
    exit(0)
else:
    print_report("OVERALL FAILURE")
    exit(1)
print_report("</p>\n")

print_report("</div></body></html>")

OUTPUT_FILE.close()
