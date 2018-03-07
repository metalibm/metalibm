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
# created:          Apr 29th, 2016
# last-modified:    Mar  7th, 2018
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
