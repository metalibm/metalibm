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
import os

import sollya
from sollya import Interval

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
import metalibm_functions.generic_log
import metalibm_functions.erf
import metalibm_functions.ml_acos
import metalibm_functions.rootn

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.ml_template import target_parser

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Int32, ML_Int64
from metalibm_core.core.ml_function import (
    BuildError, ValidError
)
from metalibm_core.core.random_gen import UniformInterval

from metalibm_core.targets.common.vector_backend import VectorBackend

from metalibm_core.targets.intel.x86_processor import (
        X86_Processor, X86_SSE_Processor, X86_SSE2_Processor,
        X86_SSE3_Processor, X86_SSSE3_Processor, X86_SSE41_Processor,
        X86_AVX_Processor, X86_AVX2_Processor
        )
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_configuration import CodeConfiguration


from metalibm_core.targets.intel.m128_promotion import Pass_M128_Promotion
from metalibm_core.targets.intel.m256_promotion import Pass_M256_Promotion
from metalibm_core.utility.ml_template import target_instanciate
from metalibm_core.utility.build_utils import SourceFile

from valid.test_utils import *
from valid.test_summary import TestSummary


try:
    from metalibm_core.targets.kalray.k1b_processor import K1B_Processor
    k1b_defined = True
    k1b = K1B_Processor()
except ImportError:
    k1b_defined = False
    k1b = None


# default directory to load AXF file from
AXF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "metalibm_functions", "axf")

# common value threshold for error
# FIXME/TODO: should be define on a per-function / per-test basis
ERROR_ULP_THRESHOLD = 1.0

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



class FunctionTest:
    def __init__(self, ctor, arg_map_list, title=None, specific_opts_builder=lambda v: v, predicate=lambda _: True):
        """ FunctionTest constructor:

            Args:
                ctor(class): constructor of the meta-function
                arg_map_list: list of dictionnaries
                predicate: constraint of validity for the test

        """
        self.ctor = ctor
        self.arg_map_list = arg_map_list
        self.title = title if not title is None else ctor.function_name
        # callback(<option dict>) -> specialized <option dict>
        self.specific_opts_builder = specific_opts_builder
        self.predicate = predicate


    @property
    def tag(self):
        return self.title

# global bench test range
BENCH_TEST_RANGE = {
    "exp": [UniformInterval(0, 80)],
    "expm1": [Interval(-1, 1)],
    "log": [UniformInterval(0, 100)],
    "log1p": [Interval(-1, 1)],
    "trigo": [UniformInterval(-1e7, 1e7)],
}

GEN_LOG_ARGS = {"basis": sollya.exp(1), "function_name": "ml_genlog", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}
GEN_LOG2_ARGS =  {"basis": 2, "function_name": "ml_genlog2", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}
GEN_LOG10_ARGS =  {"basis": 10, "function_name": "ml_genlog10", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}

class LibmFunctionTest(FunctionTest):
    @property
    def tag(self):
        # NOTES/FIXME: 0-th element of self.arg_map_list is chosen
        # for tag determination without considering the others
        return self.title + "_" + self.arg_map_list[0]["bench_function_name"]

S2 = sollya.SollyaObject(2)
S10 = sollya.SollyaObject(10)
def emulate_exp2(v):
    return S2**v
def emulate_exp10(v):
    return S10**v

# predicate to limit libm test validity
BINARY32_FCT = lambda opts: (opts["precision"] == ML_Binary32)
BINARY64_FCT = lambda opts: (opts["precision"] == ML_Binary64)



# libm functions
LIBM_FUNCTION_LIST = [
    # single precision
    LibmFunctionTest(metalibm_functions.external_bench.ML_ExternalBench, [{"bench_function_name": fname, "emulate": emulate, "precision": ML_Binary32, "auto_test": 0, "bench_test_range": bench_range, "headers": ["math.h"]}], title="libm", predicate=BINARY32_FCT)
    for fname, emulate, bench_range in [
        ("expf", sollya.exp, BENCH_TEST_RANGE["exp"]),
        ("exp2f", emulate_exp2,   BENCH_TEST_RANGE["exp"]),
        ("exp10f", emulate_exp10, BENCH_TEST_RANGE["exp"]),
        ("expm1f", sollya.expm1, BENCH_TEST_RANGE["expm1"]),
        ("logf", sollya.log, BENCH_TEST_RANGE["log"]),
        ("log2f", sollya.log2, BENCH_TEST_RANGE["log"]),
        ("log10f", sollya.log10, BENCH_TEST_RANGE["log"]),
        ("log1p", sollya.log1p, BENCH_TEST_RANGE["log1p"]),
        ("cosf", sollya.cos, BENCH_TEST_RANGE["trigo"]),
        ("sinf", sollya.sin, BENCH_TEST_RANGE["trigo"]),
        ("tanf", sollya.tan, BENCH_TEST_RANGE["trigo"]),
        ("atanf", sollya.atan, [None, None]),
        ("coshf", sollya.cosh, [None]),
        ("sinhf", sollya.sinh, [None]),
        ("tanhf", sollya.tanh, [None]),
    ]
] + [
    LibmFunctionTest(metalibm_functions.external_bench.ML_ExternalBench, [{"bench_function_name": fname, "emulate": emulate, "input_formats": [ML_Binary64], "bench_test_range": bench_range, "precision": ML_Binary64, "auto_test": 0, "headers": ["math.h"]}], title="libm", predicate=BINARY64_FCT)
    for fname, emulate, bench_range in [
        ("exp", sollya.exp,      BENCH_TEST_RANGE["exp"]),
        ("exp2", emulate_exp2,   BENCH_TEST_RANGE["exp"]),
        ("exp10", emulate_exp10, BENCH_TEST_RANGE["exp"]),
        ("expm1", sollya.expm1, BENCH_TEST_RANGE["expm1"]),
        ("log", sollya.log,     BENCH_TEST_RANGE["log"]),
        ("log2", sollya.log2,   BENCH_TEST_RANGE["log"]),
        ("log10", sollya.log10, BENCH_TEST_RANGE["log"]),
        ("log1p", sollya.log1p, BENCH_TEST_RANGE["log1p"]),
        ("cos", sollya.cos, BENCH_TEST_RANGE["trigo"]),
        ("sin", sollya.sin, BENCH_TEST_RANGE["trigo"]),
        ("tan", sollya.tan, BENCH_TEST_RANGE["trigo"]),
        ("atan", sollya.atan, [None, None]),
        ("cosh", sollya.cosh, [None]),
        ("sinh", sollya.sinh, [None]),
        ("tanh", sollya.tanh, [None]),
    ]

]


def rootn_option_specialization(opt_dict):
    """ Option specilization callback for FunctionTest
        dedicated to rootn meta-function """
    precision = opt_dict["precision"]
    input_precisions = {
        ML_Binary32: [ML_Binary32, ML_Int32],
        ML_Binary64: [ML_Binary64, ML_Int64],
    }[precision]
    auto_test_range = {
        ML_Binary32: [Interval(-2.0**126, 2.0**126), Interval(0, 255)], 
        ML_Binary64:  [Interval(-2.0**1022, 2.0**1022), Interval(0, 255)], 
    }[precision]
    opt_dict["auto_test_range"] = auto_test_range
    opt_dict["input_precisions"] = input_precisions
    return opt_dict


def tanh_option_specialization(opt_dict):
    """ Option specilization callback for FunctionTest
        dedicated to tanh meta-function """
    precision = opt_dict["precision"]
    opt_dict["load_approx"] = {
        ML_Binary32: os.path.join(AXF_DIR, "tanh-fp32.axf"),
        ML_Binary64: os.path.join(AXF_DIR, "tanh-fp64.axf"),
    }[precision]
    return opt_dict

FUNCTION_LIST = LIBM_FUNCTION_LIST + [

    # meta-functions
    FunctionTest(metalibm_functions.ml_tanh.ML_HyperbolicTangent, [{}], specific_opts_builder=tanh_option_specialization, title="ml_tanh"),

    FunctionTest(metalibm_functions.ml_atan.MetaAtan, [{}], title="ml_atan"),

    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG_ARGS], title="ml_genlog"),
    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG2_ARGS], title="ml_genlog2"),
    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG10_ARGS], title="ml_genlog10"),

    FunctionTest(metalibm_functions.ml_cosh.ML_HyperbolicCosine, [{}]),
    FunctionTest(metalibm_functions.ml_sinh.ML_HyperbolicSine, [{}]),
    FunctionTest(metalibm_functions.ml_exp.ML_Exponential, [{"bench_test_range": BENCH_TEST_RANGE["exp"]}]),
    FunctionTest(metalibm_functions.ml_log1p.ML_Log1p, [{"bench_test_range": BENCH_TEST_RANGE["log1p"]}]),

    FunctionTest(metalibm_functions.ml_div.ML_Division, [{}]),

    FunctionTest(metalibm_functions.ml_exp2.ML_Exp2, [{"bench_test_range": BENCH_TEST_RANGE["exp"]}]),
    FunctionTest(metalibm_functions.ml_cbrt.ML_Cbrt, [{}]),
    FunctionTest(metalibm_functions.ml_sqrt.MetalibmSqrt, [{}]),
    FunctionTest(metalibm_functions.ml_isqrt.ML_Isqrt, [{}]),
    FunctionTest(metalibm_functions.ml_vectorizable_log.ML_Log, [{}], title="vectorizable_log"),

    FunctionTest(metalibm_functions.ml_sincos.ML_SinCos, [{"bench_test_range": BENCH_TEST_RANGE["trigo"]}]),

    FunctionTest(metalibm_functions.erf.ML_Erf, [{}]),

    FunctionTest(metalibm_functions.ml_acos.ML_Acos, [{}]),

    FunctionTest(metalibm_functions.rootn.MetaRootN, [{}], specific_opts_builder=rootn_option_specialization),
]


def get_cmdline_option(option_list, option_value):
    """ generate the command line equivalent to the list of options and their
        corresponding values """
    OPTION_MAP = {
        "passes": lambda vlist: ("--passes " + ",".join(vlist)),
        "extra_passes": lambda vlist: ("--extra-passes " + ",".join(vlist)),
        "execute_trigger": lambda v: "--execute",
        "auto_test": lambda v: "--auto-test {}".format(v),
        "bench_test_number": lambda v: "--bench {}".format(v),
        "bench_loop_num": lambda v: "--bench-loop-num {}".format(v),
        "auto_test_std": lambda _: "--auto-test-std ",
        "target": lambda v: "--target {}".format(v),
        "precision": lambda v: "--precision {}".format(v),
        "vector_size": lambda v: "--vector-size {}".format(v),
        "function_name": lambda v: "--fname {}".format(v),
        "output_file": lambda v: "--output {}".format(v),
        "compute_max_error": lambda v: "--max-error",
        # discarding target_specific_options
        "target_exec_options": lambda _: "",
    }
    return " ".join(OPTION_MAP[option](option_value[option]) for option in option_list)


def generate_pretty_report(filename, test_list, test_summary, evolution_map):
    """ generate a HTML pretty version of the test report

        :param filename: output file name
        :type filename: str
        :param test_summary: summary of test results
        !type test_summary: TestSummary
        :param evolution_map: dictionnary of changes between this test and
                              a reference report
        :type evolution_map: dict
    """
    # extracting summary properties (for compatibility with legacy print code)
    success_count = test_summary.success_count
    success = test_summary.success
    result_map = test_summary.result_map
    NUM_TESTS = test_summary.NUM_TESTS

    with open(args.output, "w") as OUTPUT_FILE:
        def print_report(msg):
            OUTPUT_FILE.write(msg)

        print_report("<html><body><div>")
        print_report("<b>Function generation test report:</b>")
        print_report("generated with metalibm: <br/>")
        print_report("<br/>".join(CodeConfiguration.get_git_comment().split("\n")))
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
        print_report(header)

        def color_cell(msg, submsg="", color="red", subcolor="black", markup="td", indent="\t\t"):
            return """{indent}<{markup} style="color:{color}; text-align: center">{msg}<font color={subcolor}>{submsg}</font></{markup}>\n""".format(
                indent=indent, color=color, msg=msg, markup=markup,
                submsg=submsg, subcolor=subcolor
            )

        # report display
        for test_scheme in sorted(result_map.keys(), key=(lambda ts: str.lower(ts.title))):
            name = test_scheme.title
            msg = "<tr>\n\t\t<td>{:15}</td>\n".format(name)
            for result in result_map[test_scheme]:
                evolution_summary = ""
                if result.title in evolution_map:
                    local_evolution = evolution_map[result.title]
                    evolution_summary = local_evolution.html_msg
                if result.get_result():
                    cpe_measure = "-"
                    max_error = "-"
                    color = "green"
                    result_sumup = " OK  "
                    if result.return_value != None:
                        if "cpe_measure" in result.return_value:
                            cpe_measure = "{:.2f}".format(result.return_value["cpe_measure"])
                        if "max_error" in result.return_value:
                            max_error_ulps = float(result.return_value["max_error"])
                            if max_error_ulps > 1000.0:
                                max_error = "{:.2e} ulp(s)".format(max_error_ulps)
                            else:
                                max_error = "{:.2f} ulp(s)".format(max_error_ulps)
                            is_nan = (result.return_value["max_error"] != result.return_value["max_error"])
                            if max_error_ulps > ERROR_ULP_THRESHOLD or is_nan:
                                color = "orange"
                                result_sumup = "KO[V]"
                    msg += color_cell(" %s   " % result_sumup, submsg="<br />[%s, %s]%s" % (cpe_measure, max_error, evolution_summary), color=color)
                elif isinstance(result.error, GenerationError):
                    msg += color_cell("  KO[G]  ", submsg=evolution_summary, color="red")
                elif isinstance(result.error, BuildError):
                    msg += color_cell(" KO[B] ", submsg=evolution_summary, color="red")
                elif isinstance(result.error, ValidError):
                    msg += color_cell(" KO[V] ", submsg=evolution_summary, color="orange")
                elif isinstance(result.error, DisabledTest):
                    msg += color_cell(" N/A ", submsg=evolution_summary, color="grey")
                else:
                    msg += color_cell(" KO[{}] ".format(result.error), submsg=evolution_summary, color="red")
            msg += "</tr>"
            print_report(msg)

        print_report("</table>")
        print_report("<p>\n")
        print_report("Total {}/{}/{} test(s)".format(
                    color_cell("+%d" % success_count, color="green", markup="span"),
                    color_cell("-%d" % (NUM_TESTS - success_count), color="red", markup="span"),
                    NUM_TESTS)
             )
        if NUM_TESTS != 0:
            print_report("      Success: {:.1f}%".format(float(success_count) / NUM_TESTS * 100))
        else:
            print_report("      Success: {:.1f}%".format(0))

        print_report("</p>\n<p>\n")

        if success:
            print_report("OVERALL SUCCESS")
        else:
            print_report("OVERALL FAILURE")
        print_report("</p>\n")
        print_report("generated: {}".format(datetime.datetime.today()))

        print_report("</div></body></html>")



global_test_list = []

# instantiating target objects
X86_AVX2 = X86_AVX2_Processor.get_target_instance()
GENERIC_PROCESSOR = GenericProcessor.get_target_instance()
X86_PROCESSOR = X86_Processor.get_target_instance()
VECTOR_BACKEND = VectorBackend.get_target_instance()

TARGET_OPTIONS_MAP = {
    GENERIC_PROCESSOR: {},
    X86_AVX2: {
        "extra_passes": [
            "beforecodegen:basic_legalization",
            "beforecodegen:expand_multi_precision",
            "beforecodegen:virtual_vector_bool_legalization",
            "beforecodegen:m128_promotion",
            "beforecodegen:m256_promotion",
            "beforecodegen:vector_mask_test_legalization"
        ]
    },
    VECTOR_BACKEND: {
        "extra_passes": [
            "beforecodegen:virtual_vector_bool_legalization",
            "beforecodegen:vector_mask_test_legalization"
        ]
    },
    X86_PROCESSOR: {},
}

TARGET_BY_NAME_MAP = {target.target_name: target for target in TARGET_OPTIONS_MAP}

#SCALAR_TARGET_LIST = [GENERIC_PROCESSOR, X86_PROCESSOR, X86_AVX2]
SCALAR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

#VECTOR_TARGET_LIST = [VECTOR_BACKEND, X86_AVX2]
VECTOR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

def cleanify_name(name):
    return name.replace("-", "_")


def generate_test_list(NUM_AUTO_TEST, NUM_BENCH_TEST, scalar_target_tag_list, vector_target_tag_list, ENANBLE_STD_TEST=True, MAX_ERROR_EVAL=False, NUM_BENCH_LOOP=1000):
    """ generate a list of test """
    # list of all possible test for a single function
    test_list = []

    def get_target_by_tag(target_tag):
        if target_tag in TARGET_BY_NAME_MAP:
            target = TARGET_BY_NAME_MAP[target_tag]
        else:
            target_specific_options = {}
            if ":" in target_tag:
                target_tag, platform = target_tag.split(':')
                target_specific_options["target_exec_options"]  = {"platform": platform}
            target = target_parser(target_tag).get_target_instance()
            TARGET_OPTIONS_MAP[target] = target_specific_options
        return target

    def get_target_option(target_obj):
        if target_obj in TARGET_OPTIONS_MAP:
            return TARGET_OPTIONS_MAP[target_obj]
        else:
            return {}

    # generating scalar tests and adding them to test_list
    for scalar_target_tag in scalar_target_tag_list:
        scalar_target = get_target_by_tag(scalar_target_tag)
        for precision in SCALAR_PRECISION_LIST:
            options = {
                "precision": precision,
                "target": scalar_target,
                "auto_test": NUM_AUTO_TEST,
                "auto_test_std": ENANBLE_STD_TEST,
                "compute_max_error": MAX_ERROR_EVAL,
                "execute_trigger": True,
                "bench_test_number": NUM_BENCH_TEST,
                "bench_loop_num": NUM_BENCH_LOOP,
                "output_file": "{}_{}.c".format(precision, cleanify_name(scalar_target.target_name)),
                "function_name": "{}_{}".format(precision, cleanify_name(scalar_target.target_name)),
            }
            options.update(get_target_option(scalar_target))
            test_list.append(options)
    # generating vector tests and adding them to test_list
    for vector_target_tag in vector_target_tag_list:
        vector_target = get_target_by_tag(vector_target_tag)
        for precision in VECTOR_PRECISION_LIST:
            for vector_size in [4, 8]:
                options = {
                    "precision": precision,
                    "target": vector_target,
                    "vector_size": vector_size,
                    "auto_test": NUM_AUTO_TEST,
                    "bench_test_number": NUM_BENCH_TEST,
                    "bench_loop_num": NUM_BENCH_LOOP,
                    "auto_test_std": ENANBLE_STD_TEST,
                    "compute_max_error": MAX_ERROR_EVAL,
                    "execute_trigger": True,
                    "output_file": "v{}-{}_{}.c".format(vector_size, precision, cleanify_name(vector_target.target_name)),
                    "function_name": "v{}_{}_{}".format(vector_size, precision, cleanify_name(vector_target.target_name)),
                }
                options.update(get_target_option(vector_target))
                test_list.append(options)
    return test_list



class SubFunctionTest(NewSchemeTest):
    def sub_case_title(self, arg_tc):
        """ method to generate sub-case title """
        return self.title + "_" + arg_tc["function_name"]

def execute_test_list(test_list):
    """ execute all the tests listed in test_list """
    result_map = {}
    # forcing exception cause to be raised
    Log.exit_on_error = False
    for test_scheme in test_list:
        test_results = test_scheme.perform_all_test_no_reduce(debug=args.debug)
        result_map[test_scheme] = test_results

    test_summary = GlobalTestResult.init_from_test_list(test_list, result_map)

    return test_summary

class GlobalTestResult:
    """ Summary of global test execution """
    def __init__(self):
        self.success_count = None
        self.success = None
        self.result_map = {}
        self.result_details = []
        self.NUM_TESTS = None

    @staticmethod
    def init_from_test_list(global_test_list, result_map):
        """ Generate a GlobalTestResult from a result_map """
        test_summary = GlobalTestResult()
        # reset success
        test_summary.success = True
        # reset success_count
        test_summary.success_count = 0
        # list of TestResult objects generated by execution
        # of new scheme tests
        test_summary.result_details = []
        test_summary.result_map = result_map

        test_summary.NUM_TESTS = functools.reduce(lambda acc, v: acc + v.num_test, global_test_list, 0)

        for test_scheme in test_summary.result_map:
            result_list = test_summary.result_map[test_scheme]
            test_result = test_scheme.reduce_test_result(result_list)
            test_summary.success_count += test_scheme.get_success_count(result_list)
            test_summary.result_details.append(test_result)
            if not test_result.get_result():
                test_summary.success = False
        return test_summary


    def summarize(self):
        """ convert a GlobalTestResult into a TestSummary """
        test_map = {}
        for test_scheme in sorted(self.result_map.keys(), key=(lambda ts: str.lower(ts.title))):
            for result in self.result_map[test_scheme]:
                name = result.title
                if result.get_result():
                    cpe_measure = "-"
                    max_error = "-"
                    summary_tag = "OK"
                    if result.return_value != None:
                        if "cpe_measure" in result.return_value:
                            cpe_measure = "{:.2f}".format(result.return_value["cpe_measure"])
                        if "max_error" in result.return_value:
                            max_error_value = result.return_value["max_error"]
                            max_error = "{}".format(max_error_value)
                            if abs(max_error_value) > ERROR_ULP_THRESHOLD or max_error_value != max_error_value:
                                summary_tag = "KO[V]"
                    summary = [summary_tag, cpe_measure, max_error]
                elif isinstance(result.error, GenerationError):
                    summary = ["KO[G]"]
                elif isinstance(result.error, BuildError):
                    summary = ["KO[B]"]
                elif isinstance(result.error, ValidError):
                    summary = ["KO[V]"]
                elif isinstance(result.error, DisabledTest):
                    summary = ["N/A"]
                else:
                    summary = ["KO[{}]".format(result.error)]
                test_map[name] = summary
        return TestSummary(test_map)

def split_str(s):
    """ split s around ',' and removed empty sub-string """
    return [sub for sub in s.split(",") if s!= ""]

if __name__ == "__main__":
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
    arg_parser.add_argument("--select", dest="select", action="store",
                            default=None, type=(lambda v: v.split(",")),
                            help="limit test to those whose tag matches one of string list")
    arg_parser.add_argument("--exclude", dest="exclude", action="store",
                            default=[], type=(lambda v: v.split(",")),
                            help="limit test to those whose tag does not match one of string list")
    arg_parser.add_argument("--reference", dest="reference", action="store",
                            default=None,
                            help="load a reference result and compare them")
    arg_parser.add_argument("--scalar-targets", dest="scalar_targets", action="store",
                            default=["generic", "x86", "x86_avx2"],
                            type=split_str,
                            help="list of targets")
    arg_parser.add_argument("--vector-targets", dest="vector_targets", action="store",
                            default=["vector", "x86_avx2"],
                            type=split_str,
                            help="list of vector_targets")
    arg_parser.add_argument("--gen-reference", dest="gen_reference", action="store",
                            default=None,
                            help="generate a new reference file")
    arg_parser.add_argument("--bench-test-number", dest="bench_test_number", action="store",
                            default=None, type=int,
                            help="set the number of loop to run during performance bench (0 disable performance benching alltogether)")
    arg_parser.add_argument("--error-eval", dest="error_eval", action="store_const",
                            default=False, const=True,
                            help="evaluate error without failing on innacurate functions")
    arg_parser.add_argument("--timestamp", dest="timestamp", action="store_const",
                            default=False, const=True,
                            help="enable filename timestamping")
    arg_parser.add_argument("--bench-loop-num", dest="bench_loop_num", action="store",
                            default=100,
                            help="set the number of bench's loops")
    arg_parser.add_argument("--libm", dest="custom_libm", action="store",
                            default=None,
                            help="select custom libm")
    arg_parser.add_argument(
        "--verbose", dest="verbose_enable", action=VerboseAction,
        const=True, default=False,
        help="enable Verbose log level")
    args = arg_parser.parse_args(sys.argv[1:])

    # settings custom libm
    if not args.custom_libm is None:
        SourceFile.libm = args.custom_libm

    # number of self-checking test to be generated
    if args.error_eval:
        NUM_AUTO_TEST =0
        ENANBLE_STD_TEST = False
        MAX_ERROR_EVAL = True
    else:
        NUM_AUTO_TEST = 1024 # to be divisible by standard vector length
        ENANBLE_STD_TEST = True
        MAX_ERROR_EVAL = False
    test_list = generate_test_list(NUM_AUTO_TEST,
                                   args.bench_test_number,
                                   args.scalar_targets,
                                   args.vector_targets,
                                   ENANBLE_STD_TEST=ENANBLE_STD_TEST,
                                   MAX_ERROR_EVAL=MAX_ERROR_EVAL,
                                   NUM_BENCH_LOOP=args.bench_loop_num)

    def match_select(tag):
        return any(list(map(lambda e: re.match(e, tag), args.select)))
    # generating global test list
    for function_test in [f for f in FUNCTION_LIST if ((not f.tag in args.exclude) and (not f.title in args.exclude) and (args.select is None or match_select(f.tag) or match_select(f.title)))]:
        function = function_test.ctor
        local_test_list = []
        # updating copy
        for test in test_list:
            for sub_test in function_test.arg_map_list:
                option = test.copy()
                if not function_test.predicate(test):
                    option["disabled"] = True
                opt_fname = option["function_name"]
                opt_oname = option["output_file"]
                # extra_passes requrie specific management, as we do not wish to
                # overwrite one list with the other, but rather to concatenate them
                extra_passes = []
                if "extra_passes" in option:
                    extra_passes += [ep for ep in option["extra_passes"] if not ep in extra_passes]
                if "extra_passes" in sub_test:
                    extra_passes += [ep for ep in sub_test["extra_passes"] if not ep in extra_passes]

                option.update(sub_test)
                fname = sub_test["function_name"] if "function_name" in sub_test else function_test.tag # function.function_name
                option["extra_passes"] = extra_passes
                option["function_name"] = fname + "_" + function_test.title + "_" + opt_fname
                option["output_file"] = fname + "_" +function_test.title + "_" + opt_oname
                # specialization
                function_test.specific_opts_builder(option)
                local_test_list.append(option)
        test_case = SubFunctionTest(
            function_test.tag,
            function, # class / constructor
            local_test_list
        )
        global_test_list.append(test_case)


    # forcing exception cause to be raised
    Log.exit_on_error = False

    test_result = execute_test_list(global_test_list)


    test_summary = test_result.summarize()


    evolution = {}
    if args.reference:
        reference_summary = TestSummary.import_from_file(args.reference)
        #reference_summary.dump(lambda s: print("REF " + s, end=""))
        evolution = reference_summary.compare(test_summary)

    # generate output filename (possibly with timestamp)
    output_filename = args.output
    if args.timestamp:
        # this value may also be used in gen_reference
        current_date = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        out_prefix, out_suffix = os.path.splitext(output_filename)
        output_filename = "{}.{}{}".format(out_prefix, current_date, out_suffix)

    generate_pretty_report(output_filename, test_list, test_result, evolution)
    if args.gen_reference:
        # generate reference data filename (possibly with timestamp)
        reference_filename = args.gen_reference
        if args.timestamp:
            ref_prefix, ref_suffix = os.path.splitext(reference_filename)
            reference_filename = "{}.{}{}".format(ref_prefix, current_date, ref_suffix)
        with open(reference_filename, "w") as new_ref:
            test_summary.dump(lambda s: new_ref.write(s))

    exit(0)
