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
###############################################################################
# created:          Nov 11th, 2018
# last-modified:    Nov 11th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


class DisplayFormat:
    """ Generic class to describe the display/print of a format value """
    def __init__(self, format_string, pre_process_fct=None, required_header=None, pre_process_arity=1):
        # string of format
        self.format_string = format_string
        # pre-processing function:
        # value_to_display, optree -> tuple to be instanciated
        # to display format_string properly
        self.pre_process_fct = (lambda x: x) if pre_process_fct is None else pre_process_fct
        # list of headers required to execute the pre-processing
        self.required_header = [] if required_header is None else required_header
        # pre-processing arity
        self.pre_process_arity = pre_process_arity

# Standard Display formats for Metalibm formats
DISPLAY_float32 = DisplayFormat("%f")
DISPLAY_float64 = DisplayFormat("%f")
DISPLAY_int32_hex = DisplayFormat("%x")
DISPLAY_int64_hex = DisplayFormat("%\"PRIx64\"", required_header=["inttypes.h"])


DISPLAY_ftox  = DisplayFormat(format_string = "%e, %\"PRIx32\"", pre_process_fct = lambda v: "%s, float_to_32b_encoding(%s)" % (v, v), required_header = ["support_lib/ml_utils.h"])
DISPLAY_ftox_k1  = DisplayFormat(format_string = "%\"PRIx32\" ev=%x", pre_process_fct = lambda v: "float_to_32b_encoding(%s), __k1_fpu_get_exceptions()" % v, required_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double precision fp number
DISPLAY_lftolx  = DisplayFormat(format_string = "%.20e, %\"PRIx64\"", pre_process_fct = lambda v: "%s, double_to_64b_encoding(%s)" % (v, v), required_header = ["support_lib/ml_utils.h"])
DISPLAY_lftolx_k1  = DisplayFormat(format_string = "%\"PRIx64\" ev=%x", pre_process_fct = lambda v: "double_to_64b_encoding(%s), __k1_fpu_get_exceptions()" % v, required_header = ["support_lib/ml_utils.h"])

# display hexadecimal encoding of double double fp number
DISPLAY_ddtolx    = DisplayFormat(format_string = "%\"PRIx64\" %\"PRIx64\"", pre_process_fct = lambda v: "double_to_64b_encoding(%s.hi), double_to_64b_encoding(%s.lo)" % (v, v), required_header = ["support_lib/ml_utils.h"])

# display multi-precision floating-point value 
DISPLAY_DD      = DisplayFormat(format_string="{.hi=%lf, .lo=%lf}", pre_process_fct= lambda v: "%s.hi, %s.lo" % (v, v))
DISPLAY_DS      = DisplayFormat(format_string="{.hi=%f, .lo=%f)", pre_process_fct= lambda v : "%s.hi, %s.lo" % (v, v))
DISPLAY_TD      = DisplayFormat(format_string="{.hi=%lf, .me=%lf, .lo=%lf}", pre_process_fct= lambda v: "%s.hi, %s.me, %s.lo" % (v, v, v))
DISPLAY_TS      = DisplayFormat(format_string="{.hi=%f, .me=%f, .lo=%f)", pre_process_fct= lambda v : "%s.hi, %s.me, %s.lo" % (v, v, v))

DISPLAY_float2  = DisplayFormat(format_string="{%a, %a}", pre_process_fct = lambda v: "%s._[0], %s._[1]" % (v, v))
DISPLAY_float4  = DisplayFormat(format_string="{%a, %a, %a, %a}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
DISPLAY_float8  = DisplayFormat(format_string="{%a, %a, %a, %a, %a, %a, %a, %a}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

DISPLAY_int2  = DisplayFormat(format_string="{%d, %d}", pre_process_fct = lambda v: "%s._[0], %s._[1]" % (v, v))
DISPLAY_int4  = DisplayFormat(format_string="{%d, %d, %d, %d}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
DISPLAY_int8  = DisplayFormat(format_string="{%d, %d, %d, %d, %d, %d, %d, %d}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

DISPLAY_uint2  = DisplayFormat(format_string="{%u, %u}", pre_process_fct = lambda v: "%s._[0], %s._[1]" % (v, v))
DISPLAY_uint4  = DisplayFormat(format_string="{%u, %u, %u, %u}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
DISPLAY_uint8  = DisplayFormat(format_string="{%u, %u, %u, %u, %u, %u, %u, %u}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

DISPLAY_long2  = DisplayFormat(format_string="{%ld, %ld}", pre_process_fct = lambda v: "%s._[0], %s._[1]" % (v, v))
DISPLAY_long4  = DisplayFormat(format_string="{%ld, %ld, %ld, %ld}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
DISPLAY_long8  = DisplayFormat(format_string="{%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))

DISPLAY_ulong2  = DisplayFormat(format_string="{%lu, %lu}", pre_process_fct = lambda v: "%s._[0], %s._[1]" % (v, v))
DISPLAY_ulong4  = DisplayFormat(format_string="{%lu, %lu, %lu, %lu}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3]" % (v, v, v, v))
DISPLAY_ulong8  = DisplayFormat(format_string="{%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu}", pre_process_fct = lambda v: "%s._[0], %s._[1], %s._[2], %s._[3], %s._[4], %s._[5], %s._[6], %s._[7]" % (v, v, v, v, v, v, v, v))
