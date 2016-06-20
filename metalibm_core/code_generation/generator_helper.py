# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016-)
# All rights reserved
# created:          Apr 29th, 2016
# last-modified:    Apr 29th, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from ..core.ml_formats import *
from .generator_utility import FunctionOperator

def LibFunctionConstructor(require_header):
    def extend_kwords(kwords, ext_list):
        require_header_arg = [] if ((not "require_header" in kwords) or not kwords["require_header"]) else kwords["require_header"]
        require_header_arg += require_header
        kwords["require_header"] = require_header_arg
        return kwords
        
    return lambda *args, **kwords: FunctionOperator(*args, **extend_kwords(kwords, require_header))

Libm_Function       = LibFunctionConstructor(["math.h"])
Std_Function        = LibFunctionConstructor(["stdlib.h"])
Fenv_Function       = LibFunctionConstructor(["fenv.h"])
ML_Utils_Function   = LibFunctionConstructor(["support_lib/ml_utils.h"])
ML_Multi_Prec_Lib_Function   = LibFunctionConstructor(["support_lib/ml_multi_prec_lib.h"])


## wrapper for reinterpreting cast between 32-b float and integer
def Fp32AsInt32(value):
  return ML_Utils_Function("float_to_32b_encoding", arity = 1, output_precision = ML_Int32)(value)

## wrapper for reinterpreting-cast between 32-b integer and float
def Int32AsFp32(value):
  return ML_Utils_Function("float_from_32b_encoding", arity = 1, output_precision = ML_Int32)(value)
