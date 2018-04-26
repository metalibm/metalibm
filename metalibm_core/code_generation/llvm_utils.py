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
# created:          Apr  5th, 2018
# last-modified:    Apr  5th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Int64, ML_Binary32, ML_Binary64, v4float32, v4float64,
    v4int32, v4int64,
    v2int32, v2int64, v2float32, v2float64,
    v8int32, v8int64, v8float32, v8float64,
    ML_Bool, v2bool, v4bool, v8bool,
)
from metalibm_core.utility.log_report import Log
def llvm_ir_format(precision):
    """ Translate from Metalibm precision to string for LLVM-IR format """
    try:
        return {
            ML_Bool: "i1",
            v2bool: "<2 x i1>",
            v4bool: "<4 x i1>",
            v8bool: "<8 x i1>",
            ML_Int32: "i32",
            ML_Int64: "i64",
            ML_Binary32: "float",
            ML_Binary64: "double",
            v2int32: "<2 x i32>",
            v2int64: "<2 x i64>",
            v2float32: "<2 x float>",
            v2float64: "<2 x double>",
            v4float32: "<4 x float>",
            v4float64: "<4 x double>",
            v4int32: "<4 x i32>",
            v4int64: "<4 x i64>",
            v8int32: "<8 x i32>",
            v8float32: "<8 x float>",
            v8int64: "<8 x i64>",
            v8float64: "<8 x double>",
        }[precision]
    except KeyError:
        Log.report(Log.Error, "unknown precision {} in llvm_ir_format".format(precision), error=KeyError)
