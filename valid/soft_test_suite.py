# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/metalibm/metalibm)

# MIT License
#
# Copyright (c) 2021-2022 Nicolas Brunie
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
import functools
import argparse

from metalibm_core.code_generation.code_configuration import CodeConfiguration
from metalibm_core.core.ml_function import BuildError, ValidError
from metalibm_core.utility.log_report import Log
from valid.test_summary import TestSummary
from valid.test_utils import DisabledTest, GenerationError, NewSchemeTest


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

def get_cmdline_option(option_list, option_value):
    """ generate the command line equivalent to the list of options and their
        corresponding values """
    OPTION_MAP = {
        "passes": lambda vlist: ("--passes " + ",".join(vlist)),
        "extra_passes": lambda vlist: ("--extra-passes " + ",".join(vlist)),
        "execute_trigger": lambda v: "--execute",
        "auto_test": lambda v: "--auto-test {}".format(v),
        "bench_test_number": lambda v: "" if v is None else "--bench {}".format(v),
        "bench_loop_num": lambda v: "--bench-loop-num {}".format(v),
        "auto_test_std": lambda _: "--auto-test-std ",
        "target": lambda v: "--target {}".format(v),
        "precision": lambda v: "--precision {}".format(v),
        "vector_size": lambda v: "--vector-size {}".format(v),
        "function_name": lambda v: "--fname {}".format(v),
        "output_file": lambda v: "--output {}".format(v),
        "compute_max_error": lambda v: "--max-error {}".format(v),
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

    with open(filename, "w") as OUTPUT_FILE:
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
                            if max_error_ulps > ERROR_ULP_THRESHOLD or is_nan or max_error_ulps < 0:
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


def genGlobalTestList(fctList, testList):
    """ generate the global list of all tests to be executed """
    global_test_list = []
    for function_test in fctList:
        function = function_test.ctor
        local_test_list = []
        # updating copy
        for test in testList:
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
    return global_test_list


class SubFunctionTest(NewSchemeTest):
    def sub_case_title(self, arg_tc):
        """ method to generate sub-case title """
        return self.title + "_" + arg_tc["function_name"]


def execute_test_list(test_list, debug):
    """ execute all the tests listed in test_list """
    result_map = {}
    # forcing exception cause to be raised
    Log.exit_on_error = False
    for test_scheme in test_list:
        test_results = test_scheme.perform_all_test_no_reduce(debug=debug)
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
                            if abs(max_error_value) > ERROR_ULP_THRESHOLD or max_error_value != max_error_value or max_error_value < 0:
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


def populateTestSuiteArgParser(arg_parser: argparse.ArgumentParser):
    """ register standard argument for testsuite program """
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
                            default=None, type=int,
                            help="set the number of loop to run during performance bench (0 disable performance benching alltogether)")
    arg_parser.add_argument("--error-eval", dest="error_eval", action="store_const",
                            default=False, const=True,
                            help="evaluate error without failing on innacurate functions")
    arg_parser.add_argument("--timestamp", dest="timestamp", action="store_const",
                            default=False, const=True,
                            help="enable filename timestamping")
    arg_parser.add_argument("--bench-loop-num", dest="bench_loop_num", action="store",
                            default=100, type=int,
                            help="set the number of bench's loops")
    arg_parser.add_argument("--libm", dest="custom_libm", action="store",
                            default=None,
                            help="select custom libm")
    arg_parser.add_argument(
        "--verbose", dest="verbose_enable", action=VerboseAction,
        const=True, default=False,
        help="enable Verbose log level")
    return arg_parser


def split_str(s):
    """ split s around ',' and removed empty sub-string """
    return list(filter(lambda v: v != "", s.split(",")))