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
import metalibm_functions.ml_acos

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

    @property
    def tag(self):
        return self.ctor.function_name

GEN_LOG_ARGS = {"basis": sollya.exp(1), "function_name": "ml_genlog", "extra_passes" : ["beforecodegen:fuse_fma"]}
GEN_LOG2_ARGS =  {"basis": 2, "function_name": "ml_genlog2", "extra_passes" : ["beforecodegen:fuse_fma"]}
GEN_LOG10_ARGS =  {"basis": 10, "function_name": "ml_genlog10", "extra_passes" : ["beforecodegen:fuse_fma"]}

FUNCTION_LIST = [
    FunctionTest(metalibm_functions.ml_tanh.ML_HyperbolicTangent, [{}], title="ml_tanh"),
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

    FunctionTest(metalibm_functions.ml_acos.ML_Acos, [{}]),
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
        "auto_test_std": lambda _: "--auto-test-std ",
        "target": lambda v: "--target {}".format(v),
        "precision": lambda v: "--precision {}".format(v),
        "vector_size": lambda v: "--vector-size {}".format(v),
        "function_name": lambda v: "--fname {}".format(v),
        "output_name": lambda v: "--output {}".format(v),
    }
    return " ".join(OPTION_MAP[option](option_value[option]) for option in option_list) 

class CompResultType:
    pass
class Downgraded(CompResultType):
    """ Test used to be OK, but is now KO """
    @staticmethod
    def html_msg(_):
        return  """<font color="red"> &#8600; </font>"""
class Upgraded(CompResultType):
    """ Test used to be KO, but is now OK """
    @staticmethod
    def html_msg(_):
        return """<font color="green"> &#8599; </font>"""
class NotFound(CompResultType):
    """ Test was not found in reference """
    @staticmethod
    def html_msg(_):
        return "NA"
class Improved(CompResultType):
    """ Test was and is OK, performance has improved """
    @staticmethod
    def html_msg(comp_result):
        return """<font color="green"> +{:.2f}% </font>""".format(comp_result.rel_delta)
class Decreased(CompResultType):
    """ Test was and is OK, performance has decreased """
    @staticmethod
    def html_msg(comp_result):
        return """<font color="red"> {:.2f}% </font>""".format(comp_result.rel_delta)

class CompResult:
    """ test comparison result """
    def __init__(self, comp_result):
        self.comp_result = comp_result

    @property
    def html_msg(self):
        return self.comp_result.html_msg(self)

class PerfCompResult(CompResult):
    def __init__(self, abs_delta, rel_delta):
        if abs_delta > 0:
            comp_result = Decreased
        else:
            comp_result = Improved
        CompResult.__init__(self, comp_result)
        self.abs_delta = abs_delta
        self.rel_delta = rel_delta

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
                if result.title in evolution:
                    local_evolution = evolution[result.title]
                    evolution_summary = local_evolution.html_msg
                if result.get_result():
                    cpe_measure = "-"
                    if result.return_value != None:
                        if "cpe_measure" in result.return_value:
                            cpe_measure = "{:.2f}".format(result.return_value["cpe_measure"])
                    msg += color_cell("  OK     ", submsg="<br />[%s]%s" % (cpe_measure, evolution_summary), color="green")
                elif isinstance(result.error, GenerationError):
                    msg += color_cell("  KO[G]  ", submsg=evolution_summary, color="red")
                elif isinstance(result.error, BuildError):
                    msg += color_cell(" KO[B] ", submsg=evolution_summary, color="red")
                elif isinstance(result.error, ValidError):
                    msg += color_cell(" KO[V] ", submsg=evolution_summary, color="orange")
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
        print_report("      Success: {:.1f}%".format(float(success_count) / NUM_TESTS * 100))

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


def generate_test_list(NUM_AUTO_TEST, NUM_BENCH_TEST):
    """ generate a list of test """
    # list of all possible test for a single function
    test_list = []

    # generating scalar tests and adding them to test_list
    for scalar_target in SCALAR_TARGET_LIST:
        for precision in SCALAR_PRECISION_LIST:
            options = {
                "precision": precision,
                "target": scalar_target,
                "auto_test": NUM_AUTO_TEST,
                "auto_test_std": True,
                "execute_trigger": True,
                "bench_test_number": NUM_BENCH_TEST,
                "output_name": "{}_{}.c".format(precision, scalar_target.target_name),
                "function_name": "{}_{}".format(precision, scalar_target.target_name),
            }
            options.update(TARGET_OPTIONS_MAP[scalar_target])
            test_list.append(options)
    # generating vector tests and adding them to test_list
    for vector_target in VECTOR_TARGET_LIST:
        for precision in VECTOR_PRECISION_LIST:
            for vector_size in [4, 8]:
                options = {
                    "precision": precision,
                    "target": vector_target,
                    "vector_size": vector_size,
                    "auto_test": NUM_AUTO_TEST,
                    "bench_test_number": NUM_BENCH_TEST,
                    "auto_test_std": True,
                    "execute_trigger": True,
                    "output_name": "v{}-{}_{}.c".format(vector_size, precision, vector_target.target_name),
                    "function_name": "v{}_{}_{}".format(vector_size, precision, vector_target.target_name),
                }
                options.update(TARGET_OPTIONS_MAP[vector_target])
                test_list.append(options)
    return test_list



class SubFunctionTest(NewSchemeTest):
    def sub_case_title(self, arg_tc):
        """ method to generate sub-case title """
        return arg_tc["function_name"]

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
                    if result.return_value != None:
                        if "cpe_measure" in result.return_value:
                            cpe_measure = "{:.2f}".format(result.return_value["cpe_measure"])
                    summary = ["OK", cpe_measure]
                elif isinstance(result.error, GenerationError):
                    summary = ["KO[G]"]
                elif isinstance(result.error, BuildError):
                    summary = ["KO[B]"]
                elif isinstance(result.error, ValidError):
                    summary = ["KO[V]"]
                else:
                    summary = ["KO[{}]".format(result.error)]
                test_map[name] = summary
        return TestSummary(test_map)

class TestSummary:
    # current version of the test summary format version
    format_version = "0"
    # list of format versions compatible with this implementation
    format_version_compatible_list = ["0"]
    def __init__(self, test_map):
        self.test_map = test_map

    def dump(self, write_callback):
        write_callback("# format_version={}\n".format(TestSummary.format_version))
        for name in self.test_map:
            write_callback(" ".join([name] + self.test_map[name]) + "\n")

    @staticmethod
    def import_from_file(ref_file):
        """ import a test summary from a file """
        with open(ref_file, "r") as stream:
            test_map = {}
            header_line = stream.readline().replace('\n', '')
            if header_line[0] != "#":
                Log.report(Log.Error, "failed to read starter char '#' in header \"{}\"", header_line)
                return None
            property_list = [tuple(v.split("=")) for v in header_line.split(" ") if "=" in v]
            properties = dict(property_list)
            ref_format_version = properties["format_version"]
            if not ref_format_version in TestSummary.format_version_compatible_list:
                Log.report(Log.Error, "reference format_version={} is not in compatibility list {}", ref_format_version, TestSummary.format_version_compatible_list)
            
            for line in stream.readlines():
                fields = line.replace('\n','').split(" ")
                name = fields[0]
                test_map[name] = fields[1:]
            return TestSummary(test_map)

    def compare(ref, res):
        """ compare to test summaries and record differences """
        # number of tests found in ref but not in res
        not_found = 0
        # number of tests successful in ref but fail in res
        downgraded = 0
        upgraded = 0
        perf_downgraded = 0
        perf_upgraded = 0
        compare_result = {}
        for label in ref.test_map:
            if not label in res.test_map:
                not_found += 1
                compare_result[label] = "not found"
            else:
                ref_status = ref.test_map[label][0]
                res_status = res.test_map[label][0]
                if ref_status == "OK" and res_status != "OK":
                    downgraded += 1
                    compare_result[label] = CompResult(Downgraded)
                elif ref_status != "OK" and res_status == "OK":
                    upgraded += 1
                    compare_result[label] = CompResult(Upgraded)
                elif ref_status == "OK" and res_status == "OK":
                    try:
                        ref_cpe = float(ref.test_map[label][1])
                        res_cpe = float(res.test_map[label][1])
                    except ValueError:
                        compare_result[label] = CompResult(NotFound)

                    else:
                        abs_delta = ref_cpe - res_cpe
                        rel_delta = ((1 - res_cpe / ref_cpe) * 100)
                        compare_result[label] = PerfCompResult(abs_delta, rel_delta)
                        if ref_cpe > res_cpe:
                            perf_downgraded += 1
                        elif ref_cpe < res_cpe:
                            perf_upgraded += 1
        return compare_result

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
    arg_parser.add_argument("--gen-reference", dest="gen_reference", action="store",
                            default=None,
                            help="generate a new reference file")
    arg_parser.add_argument("--bench-test-number", dest="bench_test_number", action="store",
                            default=None,
                            help="set the number of loop to run during performance bench (0 disable performance benching alltogether)")
    arg_parser.add_argument(
        "--verbose", dest="verbose_enable", action=VerboseAction,
        const=True, default=False,
        help="enable Verbose log level")
    args = arg_parser.parse_args(sys.argv[1:])

    # number of self-checking test to be generated
    NUM_AUTO_TEST = 1024 # to be divisible by standard vector length
    test_list = generate_test_list(NUM_AUTO_TEST, args.bench_test_number)

    # generating global test list
    for function_test in [f for f in FUNCTION_LIST if (not f.tag in args.exclude and (args.select is None or f.tag in args.select))]:
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
        test_case = SubFunctionTest(
            function_test.title,
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
    generate_pretty_report(args.output, test_list, test_result, evolution)
    if args.gen_reference:
        with open(args.gen_reference, "w") as new_ref:
            test_summary.dump(lambda s: new_ref.write(s))

    exit(0)
