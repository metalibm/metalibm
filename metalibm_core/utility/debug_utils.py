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

from metalibm_core.core.attributes import ML_Debug, ML_AdvancedDebug, ML_MultiDebug
from metalibm_core.core.ml_formats import *

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

# display multi-precision floating-point value 
debug_dd      = ML_Debug(display_format="{.hi=%lf, .lo=%lf}", pre_process= lambda v: "%s.hi, %s.lo" % (v, v))
debug_ds      = ML_Debug(display_format="{.hi=%f, .lo=%f)", pre_process= lambda v : "%s.hi, %s.lo" % (v, v))
debug_td      = ML_Debug(display_format="{.hi=%lf, .me=%lf, .lo=%lf}", pre_process= lambda v: "%s.hi, %s.me, %s.lo" % (v, v, v))
debug_ts      = ML_Debug(display_format="{.hi=%f, .me=%f, .lo=%f)", pre_process= lambda v : "%s.hi, %s.me, %s.lo" % (v, v, v))

debug_float2  = ML_Debug(display_format = "{%a, %a}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_float4  = ML_Debug(display_format = "{%a, %a, %a, %a}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_float8  = ML_Debug(display_format = "{%a, %a, %a, %a, %a, %a, %a, %a}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_int2  = ML_Debug(display_format = "{%d, %d}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_int4  = ML_Debug(display_format = "{%d, %d, %d, %d}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_int8  = ML_Debug(display_format = "{%d, %d, %d, %d, %d, %d, %d, %d}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_uint2  = ML_Debug(display_format = "{%u, %u}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_uint4  = ML_Debug(display_format = "{%u, %u, %u, %u}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_uint8  = ML_Debug(display_format = "{%u, %u, %u, %u, %u, %u, %u, %u}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_long2  = ML_Debug(display_format = "{%ld, %ld}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_long4  = ML_Debug(display_format = "{%ld, %ld, %ld, %ld}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_long8  = ML_Debug(display_format = "{%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_ulong2  = ML_Debug(display_format = "{%lu, %lu}", pre_process = lambda v: "%s._[0], %s._[1]" % (v, v))
debug_ulong4  = ML_Debug(display_format = "{%lu, %lu, %lu, %lu}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
debug_ulong8  = ML_Debug(display_format = "{%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu}", pre_process = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

debug_multi = ML_MultiDebug({
  ML_Binary32: debug_ftox,
  ML_Binary64: debug_lftolx,

  v2float32: debug_float2,
  v4float32: debug_float4,
  v8float32: debug_float8,

  v2float64: debug_float2,
  v4float64: debug_float4,
  v8float64: debug_float8,

  ML_Int32: debugd,
  ML_UInt32: debugd,
  ML_Int64: debuglld,
  ML_UInt64: debuglld,

  v2int32: debug_int2,
  v4int32: debug_int4,
  v8int32: debug_int8,
  v2uint32: debug_uint2,
  v4uint32: debug_uint4,
  v8uint32: debug_uint8,

  v2int64: debug_long2,
  v4int64: debug_long4,
  v8int64: debug_long8,
  v2uint64: debug_ulong2,
  v4uint64: debug_ulong4,
  v8uint64: debug_ulong8,

  v2bool: debug_int2,
  v4bool: debug_int4,
  v8bool: debug_int8,
  ML_Bool:  debugd,

  ML_DoubleDouble: debug_dd,
  ML_SingleSingle: debug_ds,

  ML_TripleDouble: debug_td,
  ML_TripleSingle: debug_ts,
})
