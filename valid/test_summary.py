# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/kalray/metalibm)
# MIT License
#
# Copyright (c) 2020 Kalray
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
#
""" module to manage (generate, load, compare) test summaries """
import argparse
import collections

from metalibm_core.code_generation.code_configuration import CodeConfiguration

from metalibm_core.utility.log_report import Log

class CompResultType:
    """ Parent for comparison result type """
    pass
class Downgraded(CompResultType):
    """ Test used to be OK, but is now KO """
    name = "Downgraded"
    @staticmethod
    def html_msg(_):
        return  """<font color="red"> &#8600; </font>"""
    @staticmethod
    def raw_msg(comp_result):
        return """ Down """
class Upgraded(CompResultType):
    """ Test used to be KO, but is now OK """
    name = "Upgraded"
    @staticmethod
    def html_msg(_):
        return """<font color="green"> &#8599; </font>"""
    @staticmethod
    def raw_msg(comp_result):
        return """ Up """
class NotFound(CompResultType):
    """ Test was not found in reference """
    name = "NotFound"
    @staticmethod
    def html_msg(_):
        return "NA"
    @staticmethod
    def raw_msg(comp_result):
        return """N/A"""
class Improved(CompResultType):
    """ Test was and is OK, performance has improved """
    name = "Improved"
    @staticmethod
    def html_msg(comp_result):
        return """<font color="green"> +{:.2f}% </font>""".format(comp_result.rel_delta)
    @staticmethod
    def raw_msg(comp_result):
        return """{:.2f}%""".format(comp_result.rel_delta)
class Decreased(CompResultType):
    """ Test was and is OK, performance has decreased """
    name = "Decreased"
    @staticmethod
    def html_msg(comp_result):
        return """<font color="red"> {:.2f}% </font>""".format(comp_result.rel_delta)
    @staticmethod
    def raw_msg(comp_result):
        return """{:.2f}%""".format(comp_result.rel_delta)


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

class TestSummary:
    """ test summary object """
    # current version of the test summary format version
    format_version = "1"
    # list of format versions compatible with this implementation
    format_version_compatible_list = ["0", "1"]
    def __init__(self, test_map):
        self.test_map = test_map

    def dump(self, write_callback):
        write_callback("# format_version={}\n".format(TestSummary.format_version))
        comment_lines = ["# {}\n".format(line) for line in CodeConfiguration.get_git_comment().split("\n")]
        write_callback("".join(comment_lines))
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
                if line[0] == '#':
                    # skip comment lines
                    continue
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
                compare_result[label] = CompResult(NotFound)
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
    @staticmethod
    def dump_compare_result(compare_result, improved_threshold=0.0, decreased_threshold=0.0):
        TEST_CATEGORY = [
            # predicate(label, test_result), title, message(label, test_result)
            ( lambda label, test_result: test_result.comp_result is Upgraded,
                "upgraded tests",
                lambda label, test_result: ("  {:40}: {} ".format(label, test_result.comp_result.raw_msg(test_result)))
            ),
            ( lambda label, test_result: test_result.comp_result is Improved and abs(test_result.rel_delta) > improved_threshold,
                "improved tests",
                lambda label, test_result: ("  {:40}: {} ".format(label, test_result.comp_result.raw_msg(test_result)))
            ),
            ( lambda label, test_result: test_result.comp_result is Downgraded,
                "downgraded tests",
                lambda label, test_result: ("  {:40}: {} ".format(label, test_result.comp_result.raw_msg(test_result)))
            ),
            ( lambda label, test_result: test_result.comp_result is Decreased and abs(test_result.rel_delta) > decreased_threshold,
                "decreased tests",
                lambda label, test_result: ("  {:40}: {} ".format(label, test_result.comp_result.raw_msg(test_result)))
            ),
        ]

        result_by_category = collections.defaultdict(list)

        uncategorized_tests = []

        for label in compare_result:
            result = compare_result[label]
            for index, test_tuple in enumerate(TEST_CATEGORY):
                predicate, _, _ = test_tuple
                if predicate(label, result):
                    result_by_category[index].append((label, result))
                    break
            else:
                uncategorized_tests.append((label, result))
        for index in range(len(TEST_CATEGORY)):
            _, title, msg_fct = TEST_CATEGORY[index]
            print(title)
            for label, result in result_by_category[index]:
                print(msg_fct(label, result))
        print("Summary")
        for index in range(len(TEST_CATEGORY)):
            _, title, msg_fct = TEST_CATEGORY[index]
            print("  {}: {}".format(title, len(result_by_category[index])))
        print("  Uncategorized test(s): {}".format(len(uncategorized_tests)))
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("test summary utilities")
    # enable debug mode
    arg_parser.add_argument("action",choices=["compare"],
                            help="select action")
    arg_parser.add_argument('input_files', metavar='N', type=str, nargs=2,
                            help='input files')
    arg_parser.add_argument('--decreased-threshold', type=float, default=10.0,
                            help='performance improvement threshold')
    arg_parser.add_argument('--improved-threshold', type=float, default=10.0,
                            help='performance improvement threshold')

    args = arg_parser.parse_args()

    test_summaries = []
    for filename in args.input_files: 
        test_summaries.append(TestSummary.import_from_file(filename))

    if args.action == "compare":
        comp_result = test_summaries[0].compare(test_summaries[1])
        TestSummary.dump_compare_result(comp_result,
                                        improved_threshold=args.improved_threshold,
                                        decreased_threshold=args.decreased_threshold)
    else:
        raise NotImplementedError("unsupported action {}".format(action))
