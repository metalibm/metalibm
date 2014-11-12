# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014)
# All rights reserved
# created:          Apr 23th,  2014
# last-modified:    Apr 23th,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import sys

from pythonsollya import *

from .arg_utils import extract_option_value, test_flag_option
from .log_report import Log

from ..core.ml_formats import *

from ..code_generation.generic_processor import GenericProcessor
from ..core.target import TargetRegister
from ..targets import *

# populating target_map
target_map = {}
target_map["none"] = GenericProcessor
for target_name in TargetRegister.target_map:
    target_map[target_name] = TargetRegister.get_target_by_name(target_name)(None)


#target_map = {
    # "k1a": K1A_Processor, 
    # "k1b": K1B_Processor,
    # "sse": X86_SSE_Processor, 
    # "x86_fma": X86_FMA_Processor,
#    "none": GenericProcessor
#}

precision_map = {
    "binary32": ML_Binary32, 
    "binary64": ML_Binary64, 
    "int32": ML_Int32, 
    "uint32": ML_UInt32,
    "int64":  ML_Int64,
    "uint64": ML_UInt64,
}

accuracy_map = {
    "faithful": ML_Faithful,
    "cr"      : ML_CorrectlyRounded,
}

def accuracy_parser(accuracy_str):
    if accuracy_str in accuracy_map:
        return accuracy_map[accuracy_str]
    else:
        return eval(accuracy_str)




class ML_ArgTemplate:
    def __init__(self, default_output_file, default_function_name):
        self.default_output_file = default_output_file
        self.default_function_name = default_function_name

    def sys_arg_extraction(self):
        # argument extraction 
        parse_arg = [0]
        self.libm_compliant  = test_flag_option("--libm", True, False, parse_arg = parse_arg) 
        self.debug_flag      = test_flag_option("--debug", True, False, parse_arg = parse_arg)
        target_name     = extract_option_value("--target", "none", parse_arg = parse_arg)
        self.fuse_fma        = test_flag_option("--disable-fma", False, True, parse_arg = parse_arg)
        self.output_file     = extract_option_value("--output", self.default_output_file, parse_arg = parse_arg)
        self.function_name   = extract_option_value("--fname", self.default_function_name, parse_arg = parse_arg)
        precision_name  = extract_option_value("--precision", "binary32", parse_arg = parse_arg)
        accuracy_value  = extract_option_value("--accuracy", "faithful", parse_arg = parse_arg, processing = accuracy_parser)
        self.accuracy   = accuracy_value
        self.fast_path       = test_flag_option("--no-fpe", False, True, parse_arg = parse_arg)
        self.dot_product_enabled = test_flag_option("--dot-product", True, False, parse_arg = parse_arg)

        self.target          = target_map[target_name]()
        self.precision       = precision_map[precision_name]
        return parse_arg

    def check_args(self, parse_arg):
        for i in xrange(1, len(sys.argv)):
            if not i in parse_arg:
                Log.report(Log.Error, "unknown command line argument: %s" % sys.argv[i])


if __name__ == "__main__":
    for target_name in target_map:
        print target_name, ": ", target_map[target_name]
