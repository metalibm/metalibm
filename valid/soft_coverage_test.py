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


import datetime
import argparse
import sys
import re
import functools

import sollya
from sollya import Interval

# import meta-function script from metalibm_functions directory
import metalibm_functions.ml_log10
import metalibm_functions.ml_log1p
import metalibm_functions.ml_log2
import metalibm_functions.ml_log
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
import metalibm_functions.generic_log
import metalibm_functions.erf

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Int32
from metalibm_core.core.ml_function import (
    BuildError, ValidError
)
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

class VerboseAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(VerboseAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for level_str in values.split(","):
            if ":" in level_str:
                level, sub_level = level_str.split(":")
            else:
                level, sub_level = level_str, None
            Log.enable_level(level, sub_level=sub_level)


NUM_AUTO_TEST = 1024 # to be divisible by standard vector length

class FunctionTest:
    def __init__(self, ctor, arg_map_list, title=None):
        """ FunctionTest constructor:

            Args:
                ctor(class): constructor of the meta-function
                arg_map_list: list of dictionnaries

        """
        self.ctor = ctor
        self.arg_map_list = arg_map_list
        self.title = title if not title is None else ctor.function_name

GEN_LOG_ARGS = {"basis": sollya.exp(1), "function_name": "ml_genlog", "extra_passes" : ["beforecodegen:fuse_fma"]}
GEN_LOG2_ARGS =  {"basis": 2, "function_name": "ml_genlog2", "extra_passes" : ["beforecodegen:fuse_fma"]}
GEN_LOG10_ARGS =  {"basis": 10, "function_name": "ml_genlog10", "extra_passes" : ["beforecodegen:fuse_fma"]}

FUNCTION_LIST = [
    # FunctionTest(metalibm_functions.ml_tanh.ML_HyperbolicTangent, [{}])
    # FunctionTest(metalibm_functions.ml_atan.ML_Atan, [{}])

  FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG_ARGS], title="ml_genlog"),
  FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG2_ARGS], title="ml_genlog2"),
  FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG10_ARGS], title="ml_genlog10"),

  FunctionTest(metalibm_functions.ml_cosh.ML_HyperbolicCosine, [{}]),
  FunctionTest(metalibm_functions.ml_sinh.ML_HyperbolicSine, [{}]),
  FunctionTest(metalibm_functions.ml_exp.ML_Exponential, [{}]),
  FunctionTest(metalibm_functions.ml_log1p.ML_Log1p, [{}]),

  FunctionTest(metalibm_functions.ml_div.ML_Division, [{}]),

   # superseeded by ML_GenericLog
   # FunctionTest(metalibm_functions.ml_log10.ML_Log10, [{"passes": ["beforecodegen:fuse_fma"]}]),
   # FunctionTest(metalibm_functions.ml_log.ML_Log, [{}]),
   # FunctionTest(metalibm_functions.ml_log2.ML_Log2, [{}]),

  FunctionTest(metalibm_functions.ml_exp2.ML_Exp2, [{}]),
  FunctionTest(metalibm_functions.ml_cbrt.ML_Cbrt, [{}]),
  FunctionTest(metalibm_functions.ml_sqrt.MetalibmSqrt, [{}]),
  FunctionTest(metalibm_functions.ml_isqrt.ML_Isqrt, [{}]),
  FunctionTest(metalibm_functions.ml_vectorizable_log.ML_Log, [{}]),

  FunctionTest(metalibm_functions.ml_sincos.ML_SinCos, [{}]),

  FunctionTest(metalibm_functions.erf.ML_Erf, [{}]),
]

global_test_list = []

# instantiating target objects
X86_AVX2 = X86_AVX2_Processor()
GENERIC_PROCESSOR = GenericProcessor()
X86_PROCESSOR = X86_Processor()
VECTOR_BACKEND = VectorBackend()

TARGET_OPTIONS_MAP = {
    GENERIC_PROCESSOR: {},
    X86_AVX2: {
        "extra_passes": [
            "beforecodegen:basic_legalization",
            "beforecodegen:expand_multi_precision",
            "beforecodegen:m128_promotion",
            "beforecodegen:m256_promotion"
        ]
    },
    VECTOR_BACKEND: {},
    X86_PROCESSOR: {},
}

SCALAR_TARGET_LIST = [GENERIC_PROCESSOR, X86_PROCESSOR, X86_AVX2]
SCALAR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

VECTOR_TARGET_LIST = [VECTOR_BACKEND, X86_AVX2]
VECTOR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

# list of all possible test for a single function
test_list = []
for scalar_target in SCALAR_TARGET_LIST:
    for precision in SCALAR_PRECISION_LIST:
        options = {
            "precision": precision,
            "target": scalar_target,
            "auto_test": NUM_AUTO_TEST,
            "auto_test_std": True,
            "execute_trigger": True,
            "output_name": "{}_{}.c".format(precision, scalar_target.target_name),
            "function_name": "{}_{}".format(precision, scalar_target.target_name),
        }
        options.update(TARGET_OPTIONS_MAP[scalar_target])
        test_list.append(options)
for vector_target in VECTOR_TARGET_LIST:
    for precision in VECTOR_PRECISION_LIST:
        for vector_size in [4, 8]:
            options = {
                "precision": precision,
                "target": vector_target,
                "vector_size": vector_size,
                "auto_test": NUM_AUTO_TEST,
                "auto_test_std": True,
                "execute_trigger": True,
                "output_name": "v{}-{}_{}.c".format(vector_size, precision, vector_target.target_name),
                "function_name": "v{}_{}_{}".format(vector_size, precision, vector_target.target_name),
            }
            options.update(TARGET_OPTIONS_MAP[vector_target])
            test_list.append(options)

for function_test in FUNCTION_LIST:
    function = function_test.ctor
    local_test_list = []
    # updating copy
    for test in test_list:
        for sub_test in function_test.arg_map_list:
            option = test.copy()
            opt_fname = option["function_name"]
            opt_oname = option["output_name"]
            option.update(sub_test)
            fname = sub_test["function_name"] if "function_name" in sub_test else function.function_name
            option["function_name"] = fname + "_" + opt_fname
            option["output_name"] = fname + "_" + opt_oname
            local_test_list.append(option)
    test_case = NewSchemeTest(
        function_test.title,
        function, # class / constructor
        local_test_list
    )
    global_test_list.append(test_case)

arg_parser = argparse.ArgumentParser(" Metalibm non-regression tests")
# enable debug mode
arg_parser.add_argument("--debug", dest="debug", action="store_const",
                        default=False, const=True,
                        help="enable debug mode")
arg_parser.add_argument("--report-only", dest="report_only", action="store_const",
                        default=False, const=True,
                        help="limit display to final report")
arg_parser.add_argument("--output", dest="output", action="store",
                        default="report.html",
                        help="define output file")
arg_parser.add_argument(
    "--verbose", dest="verbose_enable", action=VerboseAction,
    const=True, default=False,
    help="enable Verbose log level")

args = arg_parser.parse_args(sys.argv[1:])

success = True
success_count = 0
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

NUM_TESTS = functools.reduce(lambda acc, v: acc + v.num_test, global_test_list, 0)

RESULT_MAP = {}


# forcing exception cause to be raised
Log.exit_on_error = False

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


def get_cmdline_option(option_list, option_value):
    OPTION_MAP = {
        "passes": lambda vlist: ("--passes " + ",".join(vlist)),
        "extra_passes": lambda vlist: ("--extra-passes " + ",".join(vlist)),
        "execute_trigger": lambda v: "--execute",
        "auto_test": lambda v: "--auto-test {}".format(v),
        "auto_test_std": lambda _: "--auto-test-std ",
        "target": lambda v: "--target {}".format(v),
        "precision": lambda v: "--precision {}".format(v),
        "vector_size": lambda v: "--vector-size {}".format(v),
        "function_name": lambda v: "--fname {}".format(v),
        "output_name": lambda v: "--output {}".format(v),
    }
    return " ".join(OPTION_MAP[option](option_value[option]) for option in option_list) 

print_report("<html><body><div>")
print_report("<b>Function generation test report:</b>")
print_report("<p><ul>\n")
for index, test_case in enumerate(test_list):
    nice_str = "; ".join("{}: {}".format(option, str(test_case[option])) for option in test_case)
    cmd_str = get_cmdline_option(test_case.keys(), test_case)
    print_report("<li><b>({}): {}</b></li>\n".format(index, nice_str))
    print_report("{}\n".format(cmd_str))

print_report("</ul></p>\n\n")
header = "<table border='1'>"
header += "<th>{:10}</th>".format("function "[:20])
header += "".join(["\t\t<th> {:2} </th>\n".format(i) for i in range(len(test_list))])
header += "\n"
#header += ("----------"+ "|----" * len(test_list))
#header += "|"
print_report(header)

def color_cell(msg, color="red", markup="td", indent="\t\t"):
    return """{indent}<{markup} style="color:{color}; text-align: center">{msg}</{markup}>\n""".format(
        indent=indent, color=color, msg=msg, markup=markup
    )

# report dusplay
for test_scheme in sorted(RESULT_MAP.keys(), key=(lambda ts: str.lower(ts.title))):
    # name = test_scheme.ctor.get_default_args().function_name
    name = test_scheme.title
    msg = "<tr>\n\t\t<td>{:15}</td>\n".format(name)
    for result in RESULT_MAP[test_scheme]:
        if result.get_result():
            msg += color_cell("  OK    ", "green")
        elif isinstance(result.error, GenerationError):
            msg += color_cell("  KO[G]  ", "red")
        elif isinstance(result.error, BuildError):
            msg += color_cell(" KO[B] ", "red")
        elif isinstance(result.error, ValidError):
            msg += color_cell(" KO[V] ", "orange")
        else:
            msg += color_cell(" KO[{}] ".format(result.error), "red")
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
else:
    print_report("OVERALL FAILURE")
print_report("</p>\n")
print_report("generated: {}".format(datetime.datetime.today()))

print_report("</div></body></html>")

OUTPUT_FILE.close()

exit(0)
