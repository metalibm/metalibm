# -*- coding: utf-8 -*-

from metalibm_core.core.attributes import ML_Debug, ML_AdvancedDebug, ML_MultiDebug
from metalibm_core.core.ml_formats import *

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
  return "echo [get_fixed_value [examine -value {signed_attr} {value}] {weight}]".format(signed_attr = signed_attr, value = value_name, weight = -fixed_prec.get_frac_size())

## Debug attributes specific for Fixed-Point values
debug_fixed = ML_AdvancedDebug(pre_process = fixed_debug_pre_process)
