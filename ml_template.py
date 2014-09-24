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

from utility.common import extract_option_value, test_flag_option

from core.ml_formats import ML_Binary64, ML_Binary32, ML_Int32, ML_Int64, ML_UInt32, ML_UInt64

from code_generation.generic_processor import GenericProcessor
from targets import *

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



class ML_ArgTemplate:
    def __init__(self, default_output_file, default_function_name):
        self.default_output_file = default_output_file
        self.default_function_name = default_function_name

    def sys_arg_extraction(self):
        # argument extraction 
        self.libm_compliant  = test_flag_option("--libm", True, False) 
        self.debug_flag      = test_flag_option("--debug", True, False)
        target_name     = extract_option_value("--target", "none")
        self.fuse_fma        = test_flag_option("--disable-fma", False, True)
        self.output_file     = extract_option_value("--output", self.default_output_file)
        self.function_name   = extract_option_value("--fname", self.default_function_name)
        precision_name  = extract_option_value("--precision", "binary32")
        accuracy_value  = extract_option_value("--accuracy", "2^-23")
        self.fast_path       = test_flag_option("--no-fpe", False, True)
        self.dot_product_enabled = test_flag_option("--dot-product", True, False)

        self.target          = target_map[target_name]()
        self.precision       = precision_map[precision_name]


if __name__ == "__main__":
    for target_name in target_map:
        print target_name, ": ", target_map[target_name]
