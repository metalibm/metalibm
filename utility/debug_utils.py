# -*- coding: utf-8 -*-

from metalibm_core.core.attributes import ML_Debug, ML_AdvancedDebug, ML_MultiDebug
from metalibm_core.core.ml_formats import *
from pythonsollya import *

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

debuglld        = ML_Debug(display_format = "%lld")


def fixed_point_pre_process(value, optree):
  scaling_factor = S2**-optree.get_precision().get_frac_size()
  return "(%e * (double)%s), %s" % (scaling_factor, value, value)

debug_fixed32 = ML_AdvancedDebug(display_format = "%e(%d)", pre_process = fixed_point_pre_process)
debug_fixed64 = ML_AdvancedDebug(display_format = "%e(%lld)", pre_process = fixed_point_pre_process)

# display hexadecimal of single precision fp number
debug_ftox  = ML_Debug(display_format = "%e, %\"PRIx32\"", pre_process = lambda v: "%s, float_to_32b_encoding(%s)" % (v, v), require_header = ["support_lib/ml_utils.h"])
debug_ftox_k1  = ML_Debug(display_format = "%\"PRIx32\" ev=%x", pre_process = lambda v: "float_to_32b_encoding(%s), __k1_fpu_get_exceptions()" % v, require_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double precision fp number
debug_lftolx  = ML_Debug(display_format = "%.20e, %\"PRIx64\"", pre_process = lambda v: "%s, double_to_64b_encoding(%s)" % (v, v), require_header = ["support_lib/ml_utils.h"])
debug_lftolx_k1  = ML_Debug(display_format = "%\"PRIx64\" ev=%x", pre_process = lambda v: "double_to_64b_encoding(%s), __k1_fpu_get_exceptions()" % v, require_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double double fp number
debug_ddtolx    = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v), require_header = ["support_lib/ml_utils.h"])

# display floating-point value of double double fp number
debug_dd      = ML_Debug(display_format = "{.hi=%lf, .lo=%lf}", pre_process = lambda v: "%s.hi, %s.lo" % (v, v))

debug_float2  = ML_Debug(display_format = "{%.3f, %.3f}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_float4  = ML_Debug(display_format = "{%.3f, %.3f, %.3f, %.3f}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_float8  = ML_Debug(display_format = "{%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_int2  = ML_Debug(display_format = "{%d, %d}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_int4  = ML_Debug(display_format = "{%d, %d, %d, %d}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_int8  = ML_Debug(display_format = "{%d, %d, %d, %d, %d, %d, %d, %d}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_multi = ML_MultiDebug({
  ML_Binary32: debug_ftox,
  ML_Binary64: debug_lftolx,
  ML_Float2: debug_float2,
  ML_Float4: debug_float4,
  ML_Float8: debug_float8,
  ML_Int32: debugd,
  ML_UInt32: debugd,
  ML_Int64: debuglld,
  ML_UInt64: debuglld,
  ML_Int2: debug_int2,
  ML_Int4: debug_int4,
  ML_Int8: debug_int8,
  ML_Bool2: debug_int2,
  ML_Bool4: debug_int4,
  ML_Bool8: debug_int8,
  ML_Bool:  debugd
})
