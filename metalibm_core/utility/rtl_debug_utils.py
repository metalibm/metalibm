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
# last-modified:    Mar  7th, 2018
#
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
from metalibm_core.core.attributes import ML_Debug, ML_AdvancedDebug, ML_MultiDebug
from metalibm_core.core.ml_formats import *

from metalibm_core.utility.log_report import Log

## Helper for debug enabling with binary value display
debug_std          = ML_Debug(display_format = " -radix 2 ")
## Helper for debug enabling with decimal value display
debug_dec          = ML_Debug(display_format = " -radix 10 ")
## Helper for debug enabling with hexadecimal value display
debug_hex          = ML_Debug(display_format = " -radix 16 ")
## Helper for debug enabling with unsigned decimal value display
debug_dec_unsigned = ML_Debug(display_format = " -decimal -unsigned ")
## Helper for debug enabling with default value display
debug_cst_dec      = ML_Debug(display_format = " ")

## debug pre-process function for
#  fixed-point value
def fixed_debug_pre_process(value_name, optree):
  fixed_prec = optree.get_precision()
  signed_attr = "-signed" if fixed_prec.get_signed() else "-unsigned"
  # return "echo [get_fixed_value [examine -value {signed_attr} {value}] {weight}]".format(signed_attr = signed_attr, value = value_name, weight = -fixed_prec.get_frac_size())
  return "echo [get_fixed_value [examine -radix 10 {value}] {weight}]".format(signed_attr = signed_attr, value = value_name, weight = -fixed_prec.get_frac_size())


## Debug attributes specific for Fixed-Point values
debug_fixed = ML_AdvancedDebug(pre_process = fixed_debug_pre_process)
