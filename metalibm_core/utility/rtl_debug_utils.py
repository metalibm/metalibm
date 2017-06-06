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
