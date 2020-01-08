# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
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
###############################################################################
# created:          Apr 23th, 2014
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

""" command-line argument templates """

import sys
import os
import argparse
import traceback

from sollya import Interval

from .arg_utils import extract_option_value, test_flag_option
from .log_report import Log

from ..core.ml_formats import *
from ..core.precisions import *

from ..code_generation.generic_processor import GenericProcessor
from ..core.target import TargetRegister
from ..targets import *
from ..code_generation.code_constant import *
from ..core.passes import Pass

from ..core.ml_hdl_format import (
    fixed_point, ML_StdLogicVectorFormat, RTL_FixedPointFormat,
    HdlVirtualFormat
)

from metalibm_core.code_generation.vhdl_backend import VHDLBackend

# import optimization passes
from metalibm_core.opt import *

import metalibm_core.utility.gappa_utils as gappa_utils

# populating target_map
target_map = {}
for target_name in TargetRegister.target_map:
    target_map[target_name] = TargetRegister.get_target_by_name(
        target_name
    )(None)


precision_map = {
    # floating-point formats
    "binary16": ML_Binary16,
    "binary32": ML_Binary32,
    "binary64": ML_Binary64,
    "bfloat16": BFloat16,
    # aliases
    "half": ML_Binary16,
    "float": ML_Binary32,
    "double": ML_Binary64,
    # multi-word formats
    "dd": ML_DoubleDouble,
    "ds": ML_SingleSingle,
    "td": ML_TripleDouble,
    "ts": ML_TripleSingle,
    # integer formats
    "int32": ML_Int32,
    "uint32": ML_UInt32,
    "int64":  ML_Int64,
    "uint64": ML_UInt64,
}


# Translation map for standard format, from str
# to their HDL compatible ML_Format counterpart
# (floating-point format are wrapper into virtual
#  formats with HDL support)
HDL_PRECISION_MAP = {
    # floating-formats
    "binary16": HdlVirtualFormat(ML_Binary16),
    "binary32": HdlVirtualFormat(ML_Binary32),
    "binary64": HdlVirtualFormat(ML_Binary64),
    "bfloat16": HdlVirtualFormat(BFloat16_Base),
    # aliases
    "half": HdlVirtualFormat(ML_Binary16),
    "float": HdlVirtualFormat(ML_Binary32),
    "double": HdlVirtualFormat(ML_Binary64),
    # integer formats
    "int32": fixed_point(32, 0, signed=True),
    "uint32": fixed_point(32, 0, signed=False),
    "int64": fixed_point(64, 0, signed=True),
    "uint64": fixed_point(64, 0, signed=False),
}

accuracy_map = {
    "faithful": ML_Faithful,
    "cr": ML_CorrectlyRounded,
}

language_map = {
    "c": C_Code,
    "opencl": OpenCL_Code,
    "gappa": Gappa_Code,
    "vhdl": VHDL_Code,

    "ll": LLVM_IR_Code,
}


# parse a string of character and convert it into
#  the corresponding ML_Format instance
#  @param precision_str string to convert
#  @return ML_Format intsance corresponding to the input string
def precision_parser(precision_str):
    """ string -> ML_Format return the precision associated
        to a string description """
    if precision_str in precision_map:
        return precision_map[precision_str]
    else:
        fixed_format_match = ML_Custom_FixedPoint_Format.match(precision_str)
        if fixed_format_match:
            fixed_format = ML_Custom_FixedPoint_Format.parse_from_match(fixed_format_match)
            return fixed_format
        else:
            try:
                eval_format = eval(precision_str)
                return eval_format
            except Exception as e:
                list_supported_formats = ", ".join(list(precision_map.keys()) + ["F[US]<integer>.<frac>"])
                Log.report(
                    Log.Error,
                    "unable to parse evaluated format {}.\nList of supported formats: {}",
                    precision_str, list_supported_formats, error=e)

def hdl_precision_parser(precision_str):
    """ translate a str to a ML_Format compatible with HDL backend
        @param precision_str (str)
        @return ML_Format object """
    if precision_str in HDL_PRECISION_MAP:
        return HDL_PRECISION_MAP[precision_str]
    else:
        fixed_format_match = RTL_FixedPointFormat.match(precision_str)
        if fixed_format_match:
            fixed_format = RTL_FixedPointFormat.parse_from_match(fixed_format_match)
            return fixed_format
        else:
            try:
                eval_format = eval(precision_str)
                return eval_format
            except Exception as e:
                Log.report(Log.Error, "unable to parse evaluated format {}", precision_str, error=e)


# Parse list of formats
#  @param format_str comma separated list of formats
#  @return the list of ML format objects
def format_list_parser(format_str):
    """ apply precision_parser to a comma-separated list of
        format string """
    return [precision_parser(prec_str) for prec_str in format_str.split(",")]

def hdl_format_list_parser(format_str):
    """ apply precision_parser to a comma-separated list of
        format string """
    return [hdl_precision_parser(prec_str) for prec_str in format_str.split(",")]

def hdl_format_map_parser(format_str):
    """ convert a comma separated list of tag:format into
        a dict of tag -> format object """
    tag_prec_list = format_str.split(",")
    tag_prec_split = [tuple(tag_prec.split(":")) for tag_prec in tag_prec_list]
    return dict((tag, hdl_precision_parser(prec)) for tag, prec in tag_prec_split)


def accuracy_parser(accuracy_str):
    """ string -> Accuracry, convert an accuracy description string
        to an accuracy object """
    if accuracy_str in accuracy_map:
        return accuracy_map[accuracy_str]
    else:
        return eval(accuracy_str)


def interval_parser(interval_str):
    """ string -> Interval conversion """
    return eval(interval_str)

## return the Target Constructor associated with
#  the string @p target_name
def target_parser(target_name):
    """ string -> target conversion """
    return target_map[target_name]
# Instanciate a target object from its string description


def target_instanciate(target_name):
    """ instanciate target object from target string
        Args:
            target_name (str): name of the target to instantiate
        Return:
            target object instance
    """
    target_class = target_parser(target_name)
    try:
        target_object = target_class()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.print_exc())
        Log.report(Log.Error, "failed to build target object")
        raise
    return target_object


def language_parser(language_str):
    """ string -> Language object conversion """
    if not language_str in language_map:
        Log.report(Log.Error, "unknown language {} (supported languages are: {})",
                   language_str, ", ".join(language_map.keys()))
    else:
        return language_map[language_str]


class DisplayExceptionAction(argparse.Action):
    """ Exception action for command-line argument """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DisplayExceptionAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        Log.exit_on_error = False

class ExitOnErrorAction(argparse.Action):
    """ Custom action for command-line command --exit-on-error """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ExitOnErrorAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        Log.exit_on_error = True

class DisablingGappa(argparse.Action):
    """ Custom action for command-line command --exit-on-error """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(DisablingGappa, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        gappa_utils.DISABLE_GAPPA = True

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

# list the available targets


def list_targets():
    for target_name in target_map:
        print("{color_prefix}{name}{color_suffix}:\n  {target_object}".format(
            name=target_name,
            target_object=target_map[target_name],
            color_prefix=BColors.OKGREEN,
            color_suffix=BColors.ENDC)
        )

class BColors:
    """ Terminal colors """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TargetInfoAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(TargetInfoAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        #print('TargetInfoAction %r %r %r' % (namespace, values, option_string))
        list_targets()
        exit(0)
        #setattr(namespace, "early_exit", True)

class PassListAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(PassListAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print("list of registered passes")
        for tag in Pass.get_pass_tag_list():
            print("  {}{}{}: {}".format(
                BColors.OKGREEN,
                tag,
                BColors.ENDC,
                Pass.get_pass_by_tag(tag).__doc__,
                )
            )
        exit(0)

# Command line action to set break on error in load module


class MLDebugAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(MLDebugAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print('MLDebugAction %r %r %r' % (namespace, values, option_string))
        ml_debug_bool = bool(values)
        setattr(namespace, "ml_debug", ml_debug_bool)
        Log.set_break_on_error(ml_debug_bool)


class ArgDefault(object):
    def __init__(self, default_value, level=0):
        self.default_value = default_value
        self.level = level

    def get_pair(self):
        return self.default_value, self.level

    def get_value(self):
        return self.default_value

    def get_level(self):
        return self.level

    @staticmethod
    def select(arg_list):
        arg_list = [ArgDefault(
            arg, -1
        ) if not isinstance(arg, ArgDefault) else arg for arg in arg_list]
        return min(arg_list, key=lambda v: v.get_level())

    @staticmethod
    def select_value(arg_list):
        return ArgDefault.select(arg_list).get_value()

# default argument template to be used when no specific value
#  are given for a specific parameter


class DefaultArgTemplate:
    base_name = "unknown_function"
    function_name = "undef_function"
    output_file = "undef.c"
    # metalim engine settings
    display_after_opt = False
    display_after_gen = False
    verbose_enable = False
    # enable dump of executed binary stdout
    display_stdout = True
    # 
    arity = 1
    # output/intermediate format Specification
    precision = ML_Binary32
    input_precisions = None
    # list of input precisions
    # None <=> [self.precision] * self.get_arity()
    abs_accuracy = None
    accuracy = ML_Faithful
    libm_compliant = False
    input_interval = Interval(0, 1)
    # Optimization parameters
    target = GenericProcessor()
    fuse_fma = False
    fast_path_extract = True
    dot_product_enabled = False
    # Debug verbosity
    debug = False
    # Vector related parameters
    vector_size = 1
    sub_vector_size = None
    language = C_Code
    # auto-test properties
    auto_test = False
    auto_test_range = Interval(0, 1)
    auto_test_std = False
    value_test = []
    # enable max error computation
    compute_max_error = False
    break_error = False
    # bench properties
    bench_test_number = 0
    bench_loop_num = 10000
    bench_test_range = Interval(0, 1)
    bench_function_name = "undefined"
    headers = []
    libraries = []
    # emulation numeric function
    emulate = lambda x: x

    # list of pre-code generation opt passe names (string tag)
    pre_gen_passes = []
    check_processor_support = True
    # source elaboration
    build_enable = False
    # when execution is required, export binary into python runtime
    # rather than executing it into a sub-process
    embedded_binary = True
    # cross-platform: build is done for and execution will be done on a remote machine (thus
    # using Target object's execute method is required)
    cross_platform = False
    # list of default optimization passes
    passes = []
    # list of extra optimization passes (to be added to default list)
    extra_passes = []
    # built binary execution
    execute_trigger = False

    def __init__(self, **kw):
        for key in kw:
            setattr(self, key, kw[key])



## default argument template to be used for entity
#  when no specific value are given for a specific parameter


class DefaultEntityArgTemplate(DefaultArgTemplate):
    base_name = "unknown_entity"
    entity_name = "unknown_entity"
    output_file = "entity.vhd"
    debug_file = None
    # Specification,
    precision = HdlVirtualFormat(ML_Binary32)
    io_precisions = None
    io_formats = None
    accuracy = ML_Faithful
    libm_compliant = False
    # Optimization parameters,
    backend = VHDLBackend()
    fuse_fma = None
    fast_path_extract = False
    # Debug verbosity,
    debug = False
    language = VHDL_Code
    # functional test related parameters
    auto_test = False
    auto_test_range = Interval(0, 1)
    auto_test_std = False
    embedded_test = True
    externalized_test_data = False
    # exit after test
    exit_after_test = True
    # RTL elaboration
    build_enable = False
    # RTL elaboration & simulation tool
    simulator = "vsim"
    # pipelined deisgn
    pipelined = False
    # pipeline register control (reset, synchronous)
    reset_pipeline = (False, True)
    negate_reset = False
    reset_name = "reset"
    recirculate_pipeline = False
    recirculate_signal_map = {}

    decorate_code = False



# Common ancestor for Argument Template class
class ML_CommonArgTemplate(object):
    # constructor for ML_CommonArgTemplate
    #  add generic arguments description (e.g. --debug)
    def __init__(self, parser, default_arg=DefaultArgTemplate):
        self.parser = parser
        self.parser.add_argument(
            "--debug", dest="debug",
            action="store_const", const=True,
            default=default_arg.debug,
            help="enable debug display in generated code")
        self.parser.add_argument(
            "--fuse-fma", dest="fuse_fma", action="store_const",
            const=True, default=default_arg.fuse_fma,
            help="disable FMA-like operation fusion")
        self.parser.add_argument(
            "--output", action="store", dest="output_file",
            default=default_arg.output_file,
            help="set output file")

        self.parser.add_argument(
            "--arity", dest="arity",
            action="store", 
            type=int,
            default=default_arg.arity,
            help="function arity (number of inputs)")

        self.parser.add_argument(
            "--accuracy", dest="accuracy", default=default_arg.accuracy,
            type=accuracy_parser, help="select accuracy")

        self.parser.add_argument(
            "--no-fpe", dest="fast_path_extract", action="store_const",
            const=False, default=default_arg.fast_path_extract,
            help="disable Fast Path Extraction")

        self.parser.add_argument(
            "--dot-product", dest="dot_product_enabled", action="store_const",
            const=True, default=default_arg.dot_product_enabled,
            help="enable Dot Product fusion")

        self.parser.add_argument(
            "--display-after-opt", dest="display_after_opt",
            action="store_const", const=True,
            default=default_arg.display_after_opt,
            help="display MDL IR after optimization")

        self.parser.add_argument(
            "--display-after-gen", dest="display_after_gen",
            action="store_const", const=True,
            default=default_arg.display_after_gen,
            help="display MDL IR after implementation generation")

        self.parser.add_argument(
            "--input-interval", dest="input_interval", type=interval_parser,
            default=default_arg.input_interval, help="select input range")

        self.parser.add_argument(
            "--vector-size", dest="vector_size", type=int,
            default=default_arg.vector_size,
            help="define size of vector (1: scalar implemenation)")
            
        self.parser.add_argument(
            "--sub-vector-size", dest="sub_vector_size", type=int,
            default=default_arg.sub_vector_size,
            help="define size of sub vector")
        # language selection
        self.parser.add_argument(
            "--language", dest="language", type=language_parser,
            default=default_arg.language,
            help="select language for generated source code")
        # auto-test related arguments
        self.parser.add_argument(
            "--auto-test", dest="auto_test", action="store", nargs='?',
            const=10, type=int, default=default_arg.auto_test,
            help="enable the generation of a self-testing numerical/functionnal\
      bench")

        self.parser.add_argument(
            "--auto-test-range", dest="auto_test_range", action="store",
            type=interval_parser, default=default_arg.auto_test_range,
            help="define the range of input values to be used during "
                 "functional testing")

        self.parser.add_argument(
            "--auto-test-std", dest="auto_test_std", action="store_const",
            const=True, default=default_arg.auto_test_std,
            help="enabling function test on standard test case list")

        self.parser.add_argument(
            "--value-test", dest="value_test", action="store",
            type=(lambda s: [tuple(sollya.parse(v) for v in t.split(",")) for t in s.split(":")]),
            default=default_arg.value_test,
            help="give input value for tests as ':'-separated list of tuples")

        # enable the computation of eval error (if self-testing enabled)
        self.parser.add_argument(
            "--max-error", dest="compute_max_error", action="store_const",
            const=True, default=default_arg.compute_max_error,
            help="enable the computation of the maximum error "
                 "(if auto-test is enabled)")
        self.parser.add_argument(
            "--break_error", dest="break_error", action="store_const",
            const=True, default=default_arg.break_error,
            help="forces tester to continue when encountering an error")
        # performance bench related arguments
        self.parser.add_argument(
            "--bench", dest="bench_test_number", action="store", nargs='?',
            const=1000, type=int, default=default_arg.bench_test_number,
            help="enable the generation of a performance bench")
        self.parser.add_argument(
            "--bench-loop-num", dest="bench_loop_num", action="store",
            type=int, default=default_arg.bench_loop_num,
            help="define the number of bench loop to run")
        self.parser.add_argument(
            "--bench-range", dest="bench_test_range", action="store",
            type=interval_parser, default=default_arg.bench_test_range,
            help="define the interval of input values to use during "
                  "performance bench")

        self.parser.add_argument(
            "--verbose", dest="verbose_enable", action=VerboseAction,
            const=True, default=default_arg.verbose_enable,
            help="enable Verbose log level")
        self.parser.add_argument(
            "--target-info", dest="target_info_flag", action=TargetInfoAction,
            const=True, default=False,
            help="display list of supported targets")

        self.parser.add_argument(
            "--display-exception-trace", dest="exception_on_error",
            action=DisplayExceptionAction, const=False,
            nargs=0,
            help="Display the full Exception trace when an error occurs")

        self.parser.add_argument(
            "--disable-gappa", dest="disable_gappa",
            action=DisablingGappa, const=True,
            default=False,
            nargs=0,
            help="disable gappa usage (even if installed)")

        self.parser.add_argument(
            "--exit-on-error", dest="exit_on_error",
            action=ExitOnErrorAction, const=True,
            default=False,
            nargs=0,
            help="convert Fatal error to sys exit rather than exception")

        self.parser.add_argument(
            "--ml-debug", dest="ml_debug", action=MLDebugAction, const=True,
            default=False, help="enable metalibm debug")
        self.parser.add_argument(
            "--pass-info", action=PassListAction,
            help="list available optmization passes")

        # list of
        self.parser.add_argument(
            "--passes", default=default_arg.passes, action="store", dest="passes",
            type=lambda s: s.split(","), help="comma separated list of slot:pass to be executed (replace default list) ")
        self.parser.add_argument(
            "--extra-passes", default=default_arg.extra_passes, action="store", dest="extra_passes",
            type=lambda s: s.split(","), help="comma separated list of slot:pass to be executed (extend list of default passes)")
        # disable check processor pass
        self.parser.add_argument(
            "--disable-check", default=True, action="store_const",
            const=False, dest="check_processor_support",
            help="disable check processor support pass run {default: enabled]")
        # enable generated code build
        self.parser.add_argument(
            "--build", dest="build_enable", action="store_const",
            const=True, default=default_arg.build_enable,
            help="enable RTL elaboration")

        # trigger generated code execution / simulation
        self.parser.add_argument(
          "--execute", dest = "execute_trigger", action = "store_const",
          const = True, default = default_arg.execute_trigger,
          help = "trigger post-build execution"
        )

        self.parser.add_argument(
            "--no-stdout", dest="display_stdout",
            action="store_const", default=True, const=False,
            help="disable display of binary stdout (test, bench execution)")

    # Extract argument from the command-line (sys.argv)
    def arg_extraction(self):
        self.args = self.parser.parse_args(sys.argv[1:])
        return self.args

    # Return @p self's parser object
    def get_parser(self):
        return self.parser

    # process argument to return overloadable arg_value
    #  @p arg_value argument value (bare or encapsulated within
    #     an ArgDefault object)
    #  @p processing function to pre-process argument value
    def process_arg(self, arg_value, processing=lambda v: v):
        if isinstance(arg_value, ArgDefault):
            value = arg_value.get_value()
            level = arg_value.get_level()
        else:
            value = arg_value
            level = -1
        return ArgDefault(processing(value), level)


# Argument template for entity object

class ML_EntityArgTemplate(ML_CommonArgTemplate):
    def __init__(self, default_entity_name,
                 default_output_file="ml_entity.vhd",
                 default_arg=DefaultEntityArgTemplate
        ):
        parser = argparse.ArgumentParser(
            " Metalibm %s entity generation script" % default_entity_name)
        self.default_output_file = default_output_file
        self.default_entity_name = default_entity_name

        ML_CommonArgTemplate.__init__(self, parser, default_arg=default_arg)

        self.parser.add_argument(
            "--entityname", dest="entity_name",
            default=self.default_entity_name,
            help="set entity name"
        )
        self.parser.add_argument("--backend", dest="backend", action="store",
            type=target_instanciate,
            default=default_arg.backend, help="select generation backend"
        )
        self.parser.add_argument(
            "--debug-file", dest="debug_file",
            action="store", help="help define output file for debug script"
        )
        self.parser.add_argument(
            "--pipelined", dest = "pipelined",
            action = "store", help = "define the number of pipeline stages"
        )
        self.parser.add_argument(
            "--reset-pipeline", dest="reset_pipeline",
            action="store_const", default=default_arg.reset_pipeline, const=(True, True),
        )
        self.parser.add_argument(
            "--async-reset-pipeline", dest="reset_pipeline",
            action="store_const", default=default_arg.reset_pipeline, const=(True, False),
        )
        self.parser.add_argument(
            "--negate-reset", dest="negate_reset",
            action="store_const", default=default_arg.negate_reset, const=True,
            help="set reset signal command to opposite (reset trigger when reset signal is 0)")
        self.parser.add_argument(
            "--reset-name", dest="reset_name",
            action="store", default=default_arg.reset_name,
            help="name of reset signal")
        self.parser.add_argument(
            "--recirculate-pipeline", dest="recirculate_pipeline",
            action="store_const", default=default_arg.recirculate_pipeline, const=True
        )
        def parse_recirculate_signal_map(rsm_s):
            """ translate a string description of recirculate_signal_map into
                a dict <stage-inde> : <signal name> """
            def parse_signal_index(s):
                index, name = s.split(":")
                return (int(index), name)
            return dict(parse_signal_index(v) for v in rsm_s.split(","))
        self.parser.add_argument(
            "--recirculate-signal-map", dest="recirculate_signal_map",
            action="store", default=default_arg.recirculate_signal_map, type=parse_recirculate_signal_map
        )
        self.parser.add_argument(
            "--no-exit",
            action="store_const",
            dest="exit_after_test",
            const=False,
            default=True,
            help="disable auto exit after functionnal test"
        )
        self.parser.add_argument(
            "--precision", dest="precision", type=hdl_precision_parser,
            default=default_arg.precision,
            help="select main precision")
        self.parser.add_argument(
            "--input-formats", dest="input_precisions",
            type=hdl_format_list_parser,
            default=default_arg.input_precisions,
            help="comma separated list of input formats")
        self.parser.add_argument(
            "--io-formats", dest="io_formats",
            type=hdl_format_map_parser,
            default=default_arg.io_formats,
            help="comma separated list of input formats")
        self.parser.add_argument(
            "--externalize-test", dest="externalized_test_data", action="store", nargs="?",
            const="test.input", default=False, help="externalize test inputs/expected in the specified data file")
        self.parser.add_argument(
            "--simulator", dest="simulator",
            choices = ["vsim", "ghdl"],
            default=default_arg.simulator,
            help="select RTL elaboration and simulation tool")
        self.parser.add_argument(
            "--decorate-code", dest="decorate_code",
            action="store_const", const=True,
            default=default_arg.decorate_code,
            help="enable code decoration with original operaiton graph")


# new argument template based on argparse module
class ML_NewArgTemplate(ML_CommonArgTemplate):
    def __init__(
            self,
            default_arg=DefaultArgTemplate
        ):

        parser = argparse.ArgumentParser(
            " Metalibm {} function generation script".format(
            default_arg.function_name))
        ML_CommonArgTemplate.__init__(self, parser, default_arg=default_arg)
        self.parser.add_argument(
            "--libm", dest="libm_compliant", action="store_const",
            const=True, default=False,
            help="generate libm compliante code"
        )
        self.parser.add_argument(
            "--no-embedded-bin", dest="embedded_binary", action="store_const",
            const=False, default=default_arg.embedded_binary,
            help="link test program as shared object to be embedded"
        )
        self.parser.add_argument(
            "--cross-platform", dest="cross_platform", action="store_const",
            const=True, default=default_arg.cross_platform,
            help="build for and execute on a remote platform"
        )
        self.parser.add_argument(
            "--fname", dest="function_name",
            default=default_arg.function_name, help="set function name"
        )
        self.parser.add_argument("--target", dest="target", action="store",
            type=target_instanciate, default=default_arg.target,
            help="select generation target"
        )
        self.parser.add_argument(
            "--precision", dest="precision", type=precision_parser,
            default=default_arg.precision,
            help="select main precision")
        self.parser.add_argument(
            "--input-formats", dest="input_precisions",
            type=format_list_parser,
            default=default_arg.input_precisions,
            help="comma separated list of input formats")




class ML_ArgTemplate(object):
    def __init__(self, default_output_file, default_function_name):
        self.default_output_file = default_output_file
        self.default_function_name = default_function_name
        self.help_map = {}
        self.parse_arg = [0]

    # standard argument extraction from command line and storing,
    #  Plus standard argument help and default value declaration
    def sys_arg_extraction(self, parse_arg=None, exit_on_info=True, check=True):
        # argument extraction
        parse_arg = self.parse_arg if parse_arg is None else parse_arg
        self.libm_compliant = test_flag_option(
            "--libm", True, False, parse_arg=parse_arg,
            help_map=self.help_map, help_str="enable libm compliance"
        )
        self.debug_flag = test_flag_option(
            "--debug", True, False, parse_arg=parse_arg,
            help_map=self.help_map,
            help_str="enable debug display in generated code"
        )
        target_name = extract_option_value(
            "--target", "none", parse_arg=parse_arg,
            help_map=self.help_map, help_str="select target"
        )
        self.fuse_fma = test_flag_option(
            "--disable-fma", False, True, parse_arg=parse_arg,
            help_map=self.help_map, help_str="disable FMA fusion"
        )
        self.output_file = extract_option_value(
            "--output", self.default_output_file,
            parse_arg=parse_arg, help_map=self.help_map,
            help_str="set output file"
        )
        self.function_name = extract_option_value(
            "--fname", self.default_function_name,
            parse_arg=parse_arg, help_map=self.help_map,
            help_str="set function name"
        )
        precision_name = extract_option_value(
            "--precision", "binary32", parse_arg=parse_arg,
            help_map=self.help_map, help_str="select main precision"
        )
        accuracy_value = extract_option_value(
            "--accuracy", "faithful", parse_arg=parse_arg,
            processing=accuracy_parser, help_map=self.help_map,
            help_str="select accuracy"
        )
        self.fast_path = test_flag_option(
            "--no-fpe", False, True, parse_arg=parse_arg,
            help_map=self.help_map, help_str="disable Fast Path Extraction"
        )
        self.dot_product_enabled = test_flag_option(
            "--dot-product", True, False, parse_arg=parse_arg,
            help_map=self.help_map, help_str="enable Dot Product fusion"
        )
        self.display_after_opt = test_flag_option(
            "--display-after-opt", True, False, parse_arg=parse_arg,
            help_map=self.help_map, help_str="display MDL IR after optimization"
        )
        self.display_after_gen = test_flag_option(
            "--display-after-gen", True, False, parse_arg=parse_arg,
            help_map=self.help_map,
            help_str="display MDL IR after implementation generation"
        )
        input_interval = extract_option_value(
            "--input-interval", "Interval(0,1)", parse_arg=parse_arg,
            processing=interval_parser, help_map=self.help_map,
            help_str="select input range"
        )
        self.vector_size = extract_option_value(
            "--vector-size", "1", parse_arg=parse_arg,
            processing=lambda v: int(v),
            help_map=self.help_map,
            help_str="define size of vector (1: scalar implemenation)"
        )
        self.language = extract_option_value(
            "--language", "c", parse_arg=parse_arg,
            processing=lambda v: language_map[v], help_map=self.help_map,
            help_str="select language for generated source code"
        )

        self.auto_test = test_flag_option(
            "--auto-test", True, False, parse_arg=parse_arg,
            help_map=self.help_map,
            help_str="enable the generation of a self-testing "\
                     "numerical/functionnal bench"
            )

        verbose_enable = test_flag_option(
            "--verbose", True, False, parse_arg=parse_arg,
            help_map=self.help_map, help_str="enable Verbose log level"
        )

        exception_on_error = test_flag_option(
            "--exception-error", True, False, parse_arg=parse_arg,
            help_map=self.help_map,
            help_str="convert Fatal error to python Exception rather "\
                     "than straight sys exit")

        if exception_on_error:
            Log.exit_on_error = False
        if verbose_enable:
            Log.enable_level(Log.Verbose)

        self.accuracy = accuracy_value
        self.target = target_map[target_name]()
        self.precision = precision_map[precision_name]
        self.input_interval = input_interval

        if check:
            self.check_args(parse_arg, exit_on_info)

        return parse_arg

    def test_flag_option(self, *args, **kwords):
        return test_flag_option(*args, help_map=self.help_map, **kwords)

    def extract_option_value(self, *args, **kwords):
        return extract_option_value(*args, help_map=self.help_map, **kwords)

    def display_help(self):
        spacew = max(len(o) for o in self.help_map)
        print("option list:")
        for option_name in self.help_map:
            print("  %s %s %s" % (
                option_name, " " * (spacew - len(option_name)),
                self.help_map[option_name]
            ))

    def check_args(self, parse_arg, exit_on_info=True):
        """ check that all options on command line have been parse
            and display info messages """
        help_flag = test_flag_option(
            "--help", True, False, parse_arg=parse_arg,
            help_map=self.help_map, help_str="display this message"
        )
        target_info_flag = test_flag_option(
            "--target-info", True, False, parse_arg=parse_arg,
            help_map=self.help_map,
            help_str="display the list of supported targets"
        )
        for i in range(1, len(sys.argv)):
            if not i in parse_arg:
                self.display_help()
                Log.report(
                    Log.Error,
                    "unknown command line argument: %s" % sys.argv[i]
                )
        if help_flag:
            self.display_help()
            if exit_on_info:
                sys.exit(0)
                return None
        if target_info_flag:
            spacew = max(len(v) for v in target_map)
            for target_name in target_map:
                print("%s: %s %s " % (
                    target_name, " " * (spacew - len(target_name)),
                    target_map[target_name]
                ))
            if exit_on_info:
                sys.exit(0)
                return None


if __name__ == "__main__":
    for target_name in target_map:
        print(target_name, ": ", target_map[target_name])
