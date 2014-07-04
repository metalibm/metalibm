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

from core.ml_formats import ML_Binary64, ML_Binary32

from code_generation.generic_processor import GenericProcessor
from kalray_proprietary.k1a_processor import K1A_Processor
from kalray_proprietary.k1b_processor import K1B_Processor
from code_generation.x86_processor import X86_FMA_Processor, X86_SSE_Processor

target_map = {
    "k1a": K1A_Processor, 
    "k1b": K1B_Processor,
    "sse": X86_SSE_Processor, 
    "x86_fma": X86_FMA_Processor,
    "none": GenericProcessor
}
precision_map = {"binary32": ML_Binary32, "binary64": ML_Binary64}



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
