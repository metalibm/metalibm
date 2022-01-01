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
import argparse
import sys
import re
import os

from metalibm_core.targets.riscv.riscv import RISCV_RV64
from metalibm_core.targets.riscv.riscv_vector import RISCV_RVV64

from sollya import Interval

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.ml_template import target_parser

from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64
from metalibm_core.core.random_gen import UniformInterval


from metalibm_core.utility.build_utils import SourceFile

from metalibm_functions.vla_function import VLAFunction

from valid.test_summary import TestSummary
from valid.soft_test_suite import (
    FunctionTest, execute_test_list, genGlobalTestList, generate_pretty_report,
    populateTestSuiteArgParser, split_str)


# default directory to load AXF file from
AXF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "metalibm_functions", "axf")


# global bench test range
BENCH_TEST_RANGE = {
    "exp": [UniformInterval(0, 80)],
    "expm1": [Interval(-1, 1)],
    "log": [UniformInterval(0, 100)],
    "log1p": [Interval(-1, 1)],
    "trigo": [UniformInterval(-1e7, 1e7)],
    "tanh": [UniformInterval(-5,5)],
    "sinh": [UniformInterval(-5,5)],
    "cosh": [UniformInterval(-5,5)],
}


RV64_GV_EXTRA_PASSES = ["optimization:basic_legalization", "optimization:fuse_fma", "beforecodegen:rvv_legalization"]

def vlaTargetPredicate(opts):
    return opts["target"] is RV64_GV

VLA_FUNCTION_LIST = [
    # meta-functions
    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "exp"}], title="vla_exp", predicate=vlaTargetPredicate),
    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "exp2"}], title="vla_exp2", predicate=vlaTargetPredicate),
    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "exp10"}], title="vla_exp10", predicate=vlaTargetPredicate),

    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "log"}], title="vla_log", predicate=vlaTargetPredicate),
    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "log2"}], title="vla_log2", predicate=vlaTargetPredicate),
    FunctionTest(VLAFunction, [{"extra_passes": RV64_GV_EXTRA_PASSES, "function": "log10"}], title="vla_log10", predicate=vlaTargetPredicate),
]


# instantiating target objects
RV64_G  = RISCV_RV64 .get_target_instance()
RV64_GV = RISCV_RVV64.get_target_instance()

TARGET_OPTIONS_MAP = {
    RV64_G:  {},
    RV64_GV: {},
}

TARGET_BY_NAME_MAP = {target.target_name: target for target in TARGET_OPTIONS_MAP}

SCALAR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

VECTOR_PRECISION_LIST = [ML_Binary32, ML_Binary64]

def cleanify_name(name):
    return name.replace("-", "_")


def generate_test_list(NUM_AUTO_TEST, NUM_BENCH_TEST,
                       scalar_target_tag_list,
                       vector_target_tag_list,
                       ENANBLE_STD_TEST=True,
                       MAX_ERROR_EVAL=False,
                       NUM_BENCH_LOOP=1000):
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
            options = {
                "precision": precision,
                "target": vector_target,
                "auto_test": NUM_AUTO_TEST,
                "bench_test_number": NUM_BENCH_TEST,
                "bench_loop_num": NUM_BENCH_LOOP,
                "auto_test_std": ENANBLE_STD_TEST,
                "compute_max_error": MAX_ERROR_EVAL,
                "execute_trigger": True,
                "output_file": "vla-{}_{}.c".format(precision, cleanify_name(vector_target.target_name)),
                "function_name": "vla_{}_{}".format(precision, cleanify_name(vector_target.target_name)),
            }
            options.update(get_target_option(vector_target))
            test_list.append(options)
    return test_list


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(" Metalibm non-regression tests")
    # enable debug mode
    arg_parser = populateTestSuiteArgParser(arg_parser)
    arg_parser.add_argument("--scalar-targets", dest="scalar_targets", action="store",
                            default=["rv64g"],
                            type=split_str,
                            help="list of targets")
    arg_parser.add_argument("--vector-targets", dest="vector_targets", action="store",
                            default=["rv64gv"],
                            type=split_str,
                            help="list of vector_targets")
    args = arg_parser.parse_args(sys.argv[1:])

    # settings custom libm
    if not args.custom_libm is None:
        SourceFile.libm = args.custom_libm

    # number of self-checking test to be generated
    if args.error_eval:
        NUM_AUTO_TEST = 0
        ENANBLE_STD_TEST = False
        MAX_ERROR_EVAL = "true_ulp"
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
    def validTag(f):
        return (not f.tag in args.exclude) and \
               (not f.title in args.exclude) and \
                (args.select is None or match_select(f.tag) or match_select(f.title))
    # generating global test list
    global_test_list = genGlobalTestList(filter(validTag, VLA_FUNCTION_LIST), test_list)


    # forcing exception cause to be raised
    Log.exit_on_error = False
    test_result  = execute_test_list(global_test_list, args.debug)
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
