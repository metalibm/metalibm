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
# This file is part of Metalibm tool
# created:
# last-modified:    Nov  7th, 2019
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.core.display_utils import fixed_point_beautify
from metalibm_core.core.ml_formats import *

## Debug attributes class to adapt the debug display message properties
#  @param display_format C string used when displaying debug message
#  @param color of the debug message
#  @param pre_process  pre_process function to be applied to the Node
#         before display
#  @param require_header list of headers required to generate the debug message
class ML_Debug(object):
    ## initialization of a new ML_Debug object
    def __init__(self, display_format = None, color = None, pre_process = lambda v: v, require_header = []):
        self.display_format = display_format
        self.color = color
        self.pre_process = pre_process
        self.require_header = require_header

    def get_display_format(self, default = "%f"):
        return self.display_format if self.display_format else default

    def get_pre_process(self, value_to_display, optree):
        return self.pre_process(value_to_display)

    def get_require_header(self):
        return self.require_header

    def select_object(self, optree):
        return self

class ML_MultiDebug(ML_Debug):
    """ Debug object which automatically select Debug message display
        according to node output precision """
    def __init__(self, debug_object_map, key_function = lambda optree: optree.get_precision()):
        self.debug_object_map = debug_object_map
        self.key_function = key_function

    def select_object(self, optree):
        """ Select debug_object corresponding to input optree
           in ML_MultiDebug debug_object_map dict """
        dbg_key = self.key_function(optree)
        try:
            return self.debug_object_map[dbg_key]
        except KeyError:
            Log.report(
                Log.Error,
                "unable to find key({}) in debug_object_map".format(dbg_key)
            )

    def add_mapping(self, debug_key, debug_object):
        """ Declare a new mapping between @p debug_key and @p debug_object """
        self.debug_object_map[debug_key] = debug_object

class ML_AdvancedDebug(ML_Debug):
  def get_pre_process(self, value_to_display, optree):
    return self.pre_process(value_to_display, optree)

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

debug_fixed = ML_AdvancedDebug(display_format = "%e(%d)", pre_process=fixed_point_pre_process)
debug_fixed32 = ML_AdvancedDebug(display_format = "%e(%d)", pre_process=fixed_point_pre_process)
debug_fixed64 = ML_AdvancedDebug(display_format = "%e(%lld)", pre_process=fixed_point_pre_process)

# display hexadecimal of single precision fp number
debug_ftox  = ML_Debug(display_format = "%a, %\"PRIx32\"", pre_process = lambda v: "%s, float_to_32b_encoding(%s)" % (v, v), require_header = ["ml_support_lib.h"])
debug_float  = ML_Debug(display_format = "%e, %\"PRIx32\"", pre_process = lambda v: "%s, float_to_32b_encoding(%s)" % (v, v), require_header = ["ml_support_lib.h"])

# display hexadecimal encoding of double precision fp number
debug_lftolx  = ML_Debug(display_format = "%.20e, %\"PRIx64\"", pre_process = lambda v: "%s, double_to_64b_encoding(%s)" % (v, v), require_header = ["ml_support_lib.h"])

# display hexadecimal encoding of double double fp number
debug_ddtolx    = ML_Debug(display_format = "%\"PRIx64\" %\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v), require_header = ["ml_support_lib.h"])

# display multi-precision floating-point value 
debug_dd      = ML_Debug(display_format="{.hi=%lf, .lo=%lf}", pre_process= lambda v: "%s.hi, %s.lo" % (v, v))
debug_ds      = ML_Debug(display_format="{.hi=%f, .lo=%f)", pre_process= lambda v : "%s.hi, %s.lo" % (v, v))
debug_td      = ML_Debug(display_format="{.hi=%lf, .me=%lf, .lo=%lf}", pre_process= lambda v: "%s.hi, %s.me, %s.lo" % (v, v, v))
debug_ts      = ML_Debug(display_format="{.hi=%f, .me=%f, .lo=%f)", pre_process= lambda v : "%s.hi, %s.me, %s.lo" % (v, v, v))

debug_float2  = ML_Debug(display_format = "{%a, %a}", pre_process = lambda v: "%s[0], %s[1]" % (v, v))
debug_float4  = ML_Debug(display_format = "{%a, %a, %a, %a}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3]" % (v, v, v, v))
debug_float8  = ML_Debug(display_format = "{%a, %a, %a, %a, %a, %a, %a, %a}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3], %s[4], %s[5], %s[6], %s[7]" % (v, v, v, v, v, v, v, v))

debug_int2  = ML_Debug(display_format = "{%d, %d}", pre_process = lambda v: "%s[0], %s[1]" % (v, v))
debug_int4  = ML_Debug(display_format = "{%d, %d, %d, %d}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3]" % (v, v, v, v))
debug_int8  = ML_Debug(display_format = "{%d, %d, %d, %d, %d, %d, %d, %d}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3], %s[4], %s[5], %s[6], %s[7]" % (v, v, v, v, v, v, v, v))

debug_uint2  = ML_Debug(display_format = "{%u, %u}", pre_process = lambda v: "%s[0], %s[1]" % (v, v))
debug_uint4  = ML_Debug(display_format = "{%u, %u, %u, %u}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3]" % (v, v, v, v))
debug_uint8  = ML_Debug(display_format = "{%u, %u, %u, %u, %u, %u, %u, %u}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3], %s[4], %s[5], %s[6], %s[7]" % (v, v, v, v, v, v, v, v))

debug_long2  = ML_Debug(display_format = "{%ld, %ld}", pre_process = lambda v: "%s[0], %s[1]" % (v, v))
debug_long4  = ML_Debug(display_format = "{%ld, %ld, %ld, %ld}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3]" % (v, v, v, v))
debug_long8  = ML_Debug(display_format = "{%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3], %s[4], %s[5], %s[6], %s[7]" % (v, v, v, v, v, v, v, v))

debug_ulong2  = ML_Debug(display_format = "{%lu, %lu}", pre_process = lambda v: "%s[0], %s[1]" % (v, v))
debug_ulong4  = ML_Debug(display_format = "{%lu, %lu, %lu, %lu}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3]" % (v, v, v, v))
debug_ulong8  = ML_Debug(display_format = "{%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu}", pre_process = lambda v: "%s[0], %s[1], %s[2], %s[3], %s[4], %s[5], %s[6], %s[7]" % (v, v, v, v, v, v, v, v))


debug_v4dualfloat64 = ML_Debug(display_format="[{%a, %a}, {%a, %a}, {%a, %a}, {%a, %a}]", pre_process=lambda v: (", ".join("{v}.hi[%d], {v}.lo[%d]" % (i, i) for i in range(4))).format(v=v)) 

class MultiDebugKey: pass
class Is32bFixedPoint: pass
class Is64bFixedPoint: pass
class IsGenericFixedPoint: pass

def advanced_debug_multi_key(optree):
    node_format = optree.get_precision()
    if isinstance(node_format, ML_Custom_FixedPoint_Format):
        if node_format.c_bit_size == 32:
            return Is32bFixedPoint
        elif node_format.c_bit_size == 64:
            return Is64bFixedPoint
        elif node_format.c_bit_size in [8, 16, 128]:
            return IsGenericFixedPoint
        else:
            Log.report(Log.Error, "format {} with c_bit_size={} is not supported in advanced_debug_multi_key", node_format, node_format.c_bit_size, error=NotImplementedError)
    else:
        return node_format

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

        v2lbool: debug_long2,
        v4lbool: debug_long4,
        v8lbool: debug_long8,

        ML_DoubleDouble: debug_dd,
        ML_SingleSingle: debug_ds,

        ML_TripleDouble: debug_td,
        ML_TripleSingle: debug_ts,

        v4dualfloat64: debug_v4dualfloat64,

        Is32bFixedPoint: debug_fixed32,
        Is64bFixedPoint: debug_fixed64,
        IsGenericFixedPoint: debug_fixed,
    },
    key_function=advanced_debug_multi_key
)
