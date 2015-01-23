# -*- coding: utf-8 -*-

from metalibm_core.core.attributes import ML_Debug

# debug utilities
# display single precision and double precision numbers
debugf        = ML_Debug(display_format = "%f")

debuglf       = ML_Debug(display_format = "%lf")

# display hexadecimal format for integer
debugx        = ML_Debug(display_format = "%x")

# display 64-bit hexadecimal format for integer
debuglx       = ML_Debug(display_format = "%\"PRIx64\"", )

# display long/int integer
debugd        = ML_Debug(display_format = "%d", pre_process = lambda v: "(int) %s" % v)

# display long long/ long int integer
debugld        = ML_Debug(display_format = "%ld")

# display hexadecimal of single precision fp number
debug_ftox  = ML_Debug(display_format = "%e, %\"PRIx32\"", pre_process = lambda v: "%s, float_to_32b_encoding(%s)" % (v, v), require_header = ["support_lib/ml_utils.h"])
debug_ftox_k1  = ML_Debug(display_format = "%\"PRIx32\" ev=%x", pre_process = lambda v: "float_to_32b_encoding(%s), __k1_fpu_get_exceptions()" % v, require_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double precision fp number
debug_lftolx  = ML_Debug(display_format = "%\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s)" % v, require_header = ["support_lib/ml_utils.h"])
debug_lftolx_k1  = ML_Debug(display_format = "%\"PRIx64\" ev=%x", pre_process = lambda v: "double_to_64b_encoding(%s), __k1_fpu_get_exceptions()" % v, require_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double double fp number
debug_ddtolx    = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v), require_header = ["support_lib/ml_utils.h"])

# display floating-point value of double double fp number
debug_dd      = ML_Debug(display_format = "{.hi=%lf, .lo=%lf}", pre_process = lambda v: "%s.hi, %s.lo" % (v, v))
