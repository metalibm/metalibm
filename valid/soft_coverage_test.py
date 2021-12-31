# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/metalibm/metalibm)

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
from metalibm_core.core.random_gen import UniformInterval

from metalibm_core.targets.common.vector_backend import VectorBackend

from metalibm_core.targets.intel.x86_processor import (
        X86_Processor, X86_AVX2_Processor)
from metalibm_core.code_generation.generic_processor import GenericProcessor


from metalibm_core.utility.build_utils import SourceFile

from valid.test_summary import TestSummary
from valid.soft_test_suite import FunctionTest, execute_test_list, genGlobalTestList, generate_pretty_report


# default directory to load AXF file from
AXF_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "metalibm_functions", "axf")

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

GEN_LOG_ARGS = {"basis": sollya.exp(1), "function_name": "ml_genlog", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}
GEN_LOG2_ARGS =  {"basis": 2, "function_name": "ml_genlog2", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}
GEN_LOG10_ARGS =  {"basis": 10, "function_name": "ml_genlog10", "extra_passes" : ["beforecodegen:fuse_fma"], "bench_test_range": BENCH_TEST_RANGE["log"]}

class LibmFunctionTest(FunctionTest):
    def __init__(self, fname, emulate, precision, bench_range, predicate):
        FunctionTest.__init__(self, metalibm_functions.external_bench.ML_ExternalBench,
                              [{
                                "bench_function_name": fname,
                                "emulate": emulate, "precision": precision,
                                "auto_test": 0,
                                "bench_test_range": bench_range,
                                "headers": ["math.h"]}],
                                title="libm", predicate=predicate)
    @property
    def tag(self):
        # NOTES/FIXME: 0-th element of self.arg_map_list is chosen
        # for tag determination without considering the others
        return self.title + "_" + self.arg_map_list[0]["bench_function_name"]

S2 = sollya.SollyaObject(2)
S10 = sollya.SollyaObject(10)
def emulate_exp2(v):
    """ sollya emulation for exp2 a.k.a 2^x """
    return S2**v
def emulate_exp10(v):
    """ sollya emulation for exp10 a.k.a 10^x """
    return S10**v

# predicate to limit libm test validity
BINARY32_FCT = lambda opts: (opts["precision"] == ML_Binary32)
BINARY64_FCT = lambda opts: (opts["precision"] == ML_Binary64)


# libm functions
LIBM_FUNCTION_LIST = [
    # single precision
    #LibmFunctionTest(metalibm_functions.external_bench.ML_ExternalBench, [{"bench_function_name": fname, "emulate": emulate, "precision": ML_Binary32, "auto_test": 0, "bench_test_range": bench_range, "headers": ["math.h"]}], title="libm", predicate=BINARY32_FCT)
    LibmFunctionTest(fname, emulate, ML_Binary32, bench_range, BINARY32_FCT)
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
        ("coshf", sollya.cosh, BENCH_TEST_RANGE["cosh"]),
        ("sinhf", sollya.sinh, BENCH_TEST_RANGE["sinh"]),
        ("tanhf", sollya.tanh, BENCH_TEST_RANGE["tanh"]),
    ]
] + [
    #LibmFunctionTest(metalibm_functions.external_bench.ML_ExternalBench, [{"bench_function_name": fname, "emulate": emulate, "input_formats": [ML_Binary64], "bench_test_range": bench_range, "precision": ML_Binary64, "auto_test": 0, "headers": ["math.h"]}], title="libm", predicate=BINARY64_FCT)
    LibmFunctionTest(fname, emulate, ML_Binary64, bench_range, BINARY64_FCT)
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
        ("cosh", sollya.cosh, BENCH_TEST_RANGE["cosh"]),
        ("sinh", sollya.sinh, BENCH_TEST_RANGE["sinh"]),
        ("tanh", sollya.tanh, BENCH_TEST_RANGE["tanh"]),
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

    FunctionTest(metalibm_functions.ml_atan.MetaAtan, [{"auto_test_range": [UniformInterval(-10, 10)]}], title="ml_atan"),

    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG_ARGS], title="ml_genlog"),
    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG2_ARGS], title="ml_genlog2"),
    FunctionTest(metalibm_functions.generic_log.ML_GenericLog,[GEN_LOG10_ARGS], title="ml_genlog10"),

    FunctionTest(metalibm_functions.ml_cosh.ML_HyperbolicCosine, [{}]),
    FunctionTest(metalibm_functions.ml_sinh.ML_HyperbolicSine, [{}]),
    FunctionTest(metalibm_functions.ml_exp.ML_Exponential, [{"bench_test_range": BENCH_TEST_RANGE["exp"]}]),
    FunctionTest(metalibm_functions.ml_log1p.ML_Log1p, [{"bench_test_range": BENCH_TEST_RANGE["log1p"]}]),

    FunctionTest(metalibm_functions.ml_div.ML_Division, [{}]),

    FunctionTest(metalibm_functions.ml_exp2.ML_Exp2, [{"bench_test_range": BENCH_TEST_RANGE["exp"]}]),
    FunctionTest(metalibm_functions.ml_cbrt.ML_Cbrt, [{"auto_test_range": [UniformInterval(-100, 100)]}]),
    FunctionTest(metalibm_functions.ml_sqrt.MetalibmSqrt, [{}]),
    FunctionTest(metalibm_functions.ml_isqrt.ML_Isqrt, [{}]),
    FunctionTest(metalibm_functions.ml_vectorizable_log.ML_Log, [{}], title="vectorizable_log"),

    FunctionTest(metalibm_functions.ml_sincos.ML_SinCos, [{"bench_test_range": BENCH_TEST_RANGE["trigo"]}]),

    FunctionTest(metalibm_functions.erf.ML_Erf, [{}]),

    FunctionTest(metalibm_functions.ml_acos.ML_Acos, [{"auto_test_range": [UniformInterval(-1, 1)]}]),

    FunctionTest(metalibm_functions.rootn.MetaRootN, [{}], specific_opts_builder=rootn_option_specialization),
]


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
                            default=100, type=int,
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
    global_test_list = genGlobalTestList(filter(validTag, FUNCTION_LIST), test_list)


    # forcing exception cause to be raised
    Log.exit_on_error = False

    test_result = execute_test_list(global_test_list, args.debug)


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
