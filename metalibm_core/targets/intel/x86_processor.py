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
# created:          Apr 11th, 2014
# last-modified:    Mar  7th, 2018
#
# Author(s):  Hugues de Lassus <hugues.de-lassus@univ-perp.fr>,
#             Nicolas Brunie <nicolas.brunie@kalray.eu>
###############################################################################

from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.complex_generator import ComplexOperator

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_operations import *
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_table import ML_TableFormat
from metalibm_core.utility.debug_utils import ML_Debug
from metalibm_core.core.special_values import (FP_PlusZero, FP_MinusZero)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_core.targets.common.vector_backend import VectorBackend

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.complex_generator import DynamicOperator

from .x86_processor_table import x86_sse_approx_table_map


def get_sse_scalar_cst(format_object, value, language = C_Code):
    base_format = format_object.get_base_format()
    if isinstance(value, FP_PlusZero):
        value_str = base_format.get_cst(0, language)
    elif isinstance(value, FP_MinusZero):
        value_str = "-" + base_format.get_cst(0, language)
    else:
        value_str = base_format.get_cst(value, language)
    return "{{{}}}/*sse*/".format(value_str)
def get_sse_vector_float_cst(format_object, value, language=C_Code):
    """ Generate vector constant value for SSE vector format """
    scalar_format = format_object.get_scalar_format()
    value_list = ["{}".format(scalar_format.get_cst(svalue, language)) for svalue in value]
    return "{{{}}}/* sse */".format(", ".join(value_list))

def signed2unsigned(value, width=32):
    """ convert a signed value to it's 2 complement unsigned
        encoding """
    if value >= 0:
        return int(value)
    else:
        return int(value + 2**(width) )
def unsigned2signed(value, width=32):
    """ convert an unsigned value representing the 2's complement
        encoding of a signed value to its numerical signed value """
    msb = value >> (width - 1)
    return int(value - msb * 2**width)


def get_sseavx_vector_bool_cst(format_object, value, language=C_Code):
    """ Convert the list of constant boolean values <value> into constant
        code for SSE/AVX vectors
    """
    # ML_<SSE/AVX>_v<i>bool is not a pratical format, it should not appear
    # during code generation (including constant generation)
    Log.report(Log.Error, "ML_(SSE/AVX)_v<i>bool format {} with value {} should not generate code", format_object, value)
    raise NotImplementedError

def get_sse_vector_int_cst(format_object, value, language=C_Code):
    """ integer constant must be packed as 64-bit signed values if built by gcc
    """
    scalar_format = format_object.get_scalar_format()
    scalar_w = scalar_format.get_bit_size()
    compound_cst = reduce((lambda acc, x: (acc * 2**scalar_w + signed2unsigned(x, scalar_w))), value[::-1], 0)
    component_w = 64
    value_list = []
    while compound_cst != 0:
        component_abs_value = compound_cst % 2**component_w
        compound_cst >>= component_w
        value_list.append(unsigned2signed(component_abs_value, component_w))

    value_enc_list = ["{}".format(ML_Int64.get_cst(value, language)) for value in value_list]
    return "{{{}}}/* sse */".format(", ".join(value_enc_list))

ML_SSE_m128  = ML_FormatConstructor(128, "__m128", None, lambda v: None)
ML_SSE_m128i = ML_FormatConstructor(128, "__m128i", None, lambda v: None)
ML_SSE_m128d = ML_FormatConstructor(128, "__m128d", None, lambda v: None)
ML_AVX_m256  = ML_FormatConstructor(256, "__m256", None, lambda v: None)
ML_AVX_m256i = ML_FormatConstructor(256, "__m256i", None, lambda v: None)
ML_AVX_m256d = ML_FormatConstructor(256, "__m256d", None, lambda v: None)

## format for a single fp32 stored in a XMM 128-bit register
ML_SSE_m128_v1float32 = VirtualFormatNoForward(ML_Binary32, ML_SSE_m128, get_sse_scalar_cst, True)
## format for single 1 fp64 in a XMM 128-bit register
ML_SSE_m128_v1float64 = VirtualFormatNoForward(ML_Binary64, ML_SSE_m128d, get_sse_scalar_cst, True)
## format for a single int32 stored in a XMM 128-bit register
ML_SSE_m128_v1int32  = VirtualFormatNoForward(ML_Int32, ML_SSE_m128i, get_sse_scalar_cst, True)
ML_SSE_m128_v1uint32  = VirtualFormatNoForward(ML_UInt32, ML_SSE_m128i, get_sse_scalar_cst, True)
## format for single 1 int64 in a XMM 128-bit register
ML_SSE_m128_v1int64  = VirtualFormatNoForward(ML_Int64, ML_SSE_m128i, get_sse_scalar_cst, True)

## format for packed 2 fp32 in a XMM 128-bit register
ML_SSE_m128_v2float32 = vector_format_builder("__m128", None, 2, ML_Binary32,
        cst_callback = get_sse_vector_float_cst)
## format for packed 4 fp32 in a XMM 128-bit register
ML_SSE_m128_v4float32 = vector_format_builder("__m128", None, 4, ML_Binary32,
        cst_callback = get_sse_vector_float_cst)
## format for packed 2 fp64 in a XMM 128-bit register
ML_SSE_m128_v2float64 = vector_format_builder("__m128d", None, 2, ML_Binary64)
## format for packed 2 int32 in a XMM 128-bit register
ML_SSE_m128_v2int32 = vector_format_builder("__m128i", None, 2, ML_Int32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 4 int32 in a XMM 128-bit register
ML_SSE_m128_v4int32   = vector_format_builder("__m128i", None, 4, ML_Int32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 2 int64 in a XMM 128-bit register
ML_SSE_m128_v2int64   = vector_format_builder("__m128i", None, 2, ML_Int64,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 2 uint32 in a XMM 128-bit register
ML_SSE_m128_v2uint32  = vector_format_builder("__m128i", None, 2, ML_UInt32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 4 uint32 in a XMM 128-bit register
ML_SSE_m128_v4uint32  = vector_format_builder("__m128i", None, 4, ML_UInt32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 2 uint64 in a XMM 128-bit register
ML_SSE_m128_v2uint64  = vector_format_builder("__m128i", None, 2, ML_UInt64,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)

# debug-format for SSE format
debug_sse_vfloat32  = ML_Debug(
    display_format="{%a, %a, %a, %a}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("float_from_32b_encoding(_mm_extract_ps({v}, {i}))".format(v=v,i=i) for i in range(4))
)
debug_sse_vint32  = ML_Debug(
    display_format="{%d, %d, %d, %d}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("_mm_extract_epi32({v}, {i})".format(v=v,i=i) for i in range(4))
)
# unsigned version
debug_sse_vuint32  = ML_Debug(
    display_format="{%u, %u, %u, %u}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("_mm_extract_epi32({v}, {i})".format(v=v,i=i) for i in range(4))
)
# virtual vector boolean format
ML_SSE_m128_v4bool = VirtualFormatNoForward(
    v4bool, ML_SSE_m128i, get_sseavx_vector_bool_cst, True)

ML_AVX_m256_v8bool  = VirtualFormatNoForward(
    v8bool, ML_AVX_m256i, get_sseavx_vector_bool_cst, True)

# registering ML_SSE_m128_v<i>float32 specific format
debug_multi.add_mapping(ML_SSE_m128_v4float32, debug_sse_vfloat32)
debug_multi.add_mapping(ML_SSE_m128_v2float32, debug_sse_vfloat32)
debug_multi.add_mapping(ML_SSE_m128_v1float32, debug_sse_vfloat32)
# registering ML_SSE_m128_v<i>int32 specific format
debug_multi.add_mapping(ML_SSE_m128_v4int32, debug_sse_vint32)
debug_multi.add_mapping(ML_SSE_m128_v2int32, debug_sse_vint32)
debug_multi.add_mapping(ML_SSE_m128_v1int32, debug_sse_vint32)
# registering ML_SSE_m128_v<i>uint32 specific format
debug_multi.add_mapping(ML_SSE_m128_v4uint32, debug_sse_vuint32)
debug_multi.add_mapping(ML_SSE_m128_v2uint32, debug_sse_vuint32)
debug_multi.add_mapping(ML_SSE_m128_v1uint32, debug_sse_vuint32)
# registering ML_SSE_m128_v<i>bool specific format
debug_multi.add_mapping(ML_SSE_m128_v4bool, debug_sse_vint32)

## format for packed 8 fp32 in a YMM 256-bit register
ML_AVX_m256_v8float32 = vector_format_builder("__m256", None, 8, ML_Binary32)
## format for packed 4 fp64 in a YMM 256-bit register
ML_AVX_m256_v4float64 = vector_format_builder("__m256d", None, 4, ML_Binary64)
## format for packed 4 int32 in a YMM 256-bit register
ML_AVX_m256_v4int32  = vector_format_builder("__m256i", None, 4, ML_Int32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 8 int32 in a YMM 256-bit register
ML_AVX_m256_v8int32   = vector_format_builder("__m256i", None, 8, ML_Int32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 4 int64 in a YMM 256-bit register
ML_AVX_m256_v4int64   = vector_format_builder("__m256i", None, 4, ML_Int64,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 4 uint32 in a YMM 256-bit register
ML_AVX_m256_v4uint32 = vector_format_builder("__m256i", None, 4, ML_UInt32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 8 uint32 in a YMM 256-bit register
ML_AVX_m256_v8uint32  = vector_format_builder("__m256i", None, 8, ML_UInt32,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)
## format for packed 4 uint64 in a YMM 256-bit register
ML_AVX_m256_v4uint64  = vector_format_builder("__m256i", None, 4, ML_UInt64,
        cst_callback = get_sse_vector_int_cst,
        compound_constructor = ML_IntegerVectorFormat)

# debug utilities for AVX format
# debug-format for SSE format
debug_avx_vfloat32  = ML_Debug(
    display_format="{%a, %a, %a, %a, %a, %an %a, %a}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("float_from_32b_encoding(_mm256_extract_epi32(_mm256_castps_si256({v}), {i}))".format(v=v,i=i) for i in range(8))
)
debug_avx_vint32  = ML_Debug(
    display_format="{%d, %d, %d, %d, %d, %d, %d, %d}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("_mm256_extract_epi32({v}, {i})".format(v=v,i=i) for i in range(8))
)
# unsigned version
debug_avx_vuint32  = ML_Debug(
    display_format="{%u, %u, %u, %u, %u, %u, %u, %u}",
    require_header=["ml_utils.h", "smmintrin.h"],
    pre_process=lambda v: ", ".join("_mm256_extract_epi32({v}, {i})".format(v=v,i=i) for i in range(8))
)

# registering ML_SSE_m128_v<i>float32 specific format
debug_multi.add_mapping(ML_AVX_m256_v8float32, debug_avx_vfloat32)
# registering ML_SSE_m128_v<i>int32 specific format
debug_multi.add_mapping(ML_AVX_m256_v8int32, debug_avx_vint32)
# registering ML_SSE_m128_v<i>uint32 specific format
debug_multi.add_mapping(ML_AVX_m256_v8uint32, debug_avx_vuint32)
# registering ML_SSE_m128_v<i>bool specific format
debug_multi.add_mapping(ML_AVX_m256_v8bool, debug_avx_vint32)

## Wrapper for intel x86_sse intrinsics
#  defined in <xmmintrin.h> header
def XmmIntrin(*args, **kw):
  kw.update({
    'require_header': ["xmmintrin.h"]
  })
  return FunctionOperator(*args, **kw)
## Wrapper for intel x86_sse2 intrinsics
#  defined in <emmintrin.h> header
def EmmIntrin(*args, **kw):
  kw.update({
    'require_header': ["emmintrin.h"]
  })
  return FunctionOperator(*args, **kw)
## Wrapper for intel x86_ssse3 intrinsics
#  defined in <tmmintrin.h> header
def TmmIntrin(*args, **kw):
  kw.update({
    'require_header': ["tmmintrin.h"]
  })
  return FunctionOperator(*args, **kw)
## Wrapper for intel x86 sse4.1 intrinsics
#  defined in <smmintrin.h> header
def SmmIntrin(*args, **kw):
  kw.update({
    'require_header': ["smmintrin.h"]
  })
  return FunctionOperator(*args, **kw)
## Wrapper for intel x86_avx2 intrinsics
#  defined in <immintrin.h> header
def ImmIntrin(*args, **kw):
  kw.update({
    'require_header': ["immintrin.h"]
  })
  return FunctionOperator(*args, **kw)


# Conversion function from any float to a float packed into a __m128 register
_mm_set_ss = XmmIntrin("_mm_set_ss", arity = 1, force_folding = True,
                       output_precision = ML_SSE_m128_v1float32)
_mm_set_sd = XmmIntrin("_mm_set_sd", arity = 1, force_folding = True,
                       output_precision = ML_SSE_m128_v1float64)
_mm_set1_epi32 = XmmIntrin("_mm_set1_epi32", arity = 1, force_folding = True,
                           output_precision = ML_SSE_m128_v1int32)

_mm_set1_epi64x = EmmIntrin("_mm_set1_epi64x", arity = 1, force_folding = True,
                            output_precision = ML_SSE_m128_v4int32)

# Conversion of a scalar float contained in a __m128 registers to a signed
# integer contained also in a __m128 register
_mm_cvt_ss2si = XmmIntrin("_mm_cvt_ss2si", arity = 1)
_mm_cvtss_si32 = _mm_cvt_ss2si # Both generate the same cvtss2si instruction
_mm_cvtsd_si64  = EmmIntrin("_mm_cvtsd_si64", arity = 1)
_mm_cvtsd_si32  = EmmIntrin("_mm_cvtsd_si32", arity = 1)
_mm_cvtss_f32 = XmmIntrin("_mm_cvtss_f32", arity = 1,
                          output_precision = ML_Binary32)
_mm_cvtsd_f64 = XmmIntrin("_mm_cvtsd_f64", arity = 1,
                          output_precision = ML_Binary64)

_mm_round_ss_rn = SmmIntrin("_mm_round_ss",
                            arg_map = {
                                0: FO_Arg(0),
                                1: FO_Arg(0),
                                2: "_MM_FROUND_TO_NEAREST_INT"
                            },
                            arity = 1,
                            output_precision = ML_SSE_m128_v1float32)

_mm_round_sd_rn = SmmIntrin("_mm_round_sd",
                            arg_map = {
                                0: FO_Arg(0),
                                1: FO_Arg(0),
                                2: "_MM_FROUND_TO_NEAREST_INT"
                            },
                            arity = 1,
                            output_precision = ML_SSE_m128_v1float64)



# 3-to-5-cycle latency / 1-to-2-cycle throughput approximate reciprocal, with a
# maximum relative error of 1.5 * 2^(-12).
_mm_rcp_ss = XmmIntrin("_mm_rcp_ss", arity = 1,
                       output_precision = ML_SSE_m128_v1float32)
_mm_rcp_ps = XmmIntrin("_mm_rcp_ps", arity = 1,
                       output_precision = ML_SSE_m128_v4float32)
_mm256_rcp_ps = ImmIntrin("_mm256_rcp_ps", arity = 1,
                          output_precision = ML_AVX_m256_v8float32)

_mm_add_ss = XmmIntrin("_mm_add_ss", arity = 2,
                       output_precision = ML_SSE_m128_v1float32)
_mm_mul_ss = XmmIntrin("_mm_mul_ss", arity = 2,
                       output_precision = ML_SSE_m128_v1float32)
_lzcnt_u32 = ImmIntrin("_lzcnt_u32", arity = 1,
                       output_precision = ML_UInt32)
_lzcnt_u64 = ImmIntrin("_lzcnt_u64", arity = 1,
                       output_precision = ML_UInt64)


# SSE2 instructions
_mm_unpackhi_pd       = EmmIntrin("_mm_unpackhi_pd", arity = 2,
                                  output_precision = ML_SSE_m128_v2float64)
_mm_unpacklo_pd       = EmmIntrin("_mm_unpacklo_pd", arity = 2,
                                  output_precision = ML_SSE_m128_v2float64)

# SSE4.1 instructions
_mm_mullo_epi32 = SmmIntrin("_mm_mullo_epi32", arity = 2,
                          output_precision = ML_SSE_m128_v4int32)

# AVX instructions
_mm256_cvtepi32_pd       = ImmIntrin("_mm256_cvtepi32_pd", arity = 1,
                                     output_precision = ML_AVX_m256_v4float64)
_mm256_extractf128_ps    = ImmIntrin("_mm256_extractf128_ps", arity = 2,
                                     output_precision = ML_SSE_m128_v4float32)
_mm256_extractf128_si256 = ImmIntrin("_mm256_extractf128_si256", arity = 2,
                                     output_precision = ML_SSE_m128_v4int32)
_mm256_insertf128_si256  = ImmIntrin("_mm256_insertf128_si256", arity = 3,
                                     output_precision = ML_SSE_m128_v4float32)
_mm256_permute_ps        = ImmIntrin("_mm256_permute_ps", arity = 2,
                                     output_precision = ML_AVX_m256_v8float32)
_mm256_unpackhi_pd       = ImmIntrin("_mm256_unpackhi_pd", arity = 2,
                                     output_precision = ML_AVX_m256_v4float64)
_mm256_unpacklo_pd       = ImmIntrin("_mm256_unpacklo_pd", arity = 2,
                                     output_precision = ML_AVX_m256_v4float64)

## AVX conversion metablock from 4 int64 to 4 packed double,
# with the condition that the 4 int64 fit in 4 int32 without overflow.
# @param optree is a Conversion.
# Details : input vector looks like D1 D0 | C1 C0 | B1 B0 | A1 A0
# and we want to convert D0 | C0 | B0 | A0 to double. We do this in 4 steps:
# 1: cast to 8 int32s
# 2: permute 32-bit words to get lower-significance words next to each other
# 3: extract the 2 lower words from high-256-bit part to form a vector of 4
#    int32s corresponding to the lower parts of the 4 int64s.
# 4: convert the 4 int32s to 4 float64s
def conversion_to_avx_mm256_cvtepi64_pd(optree):
    ymm0 = TypeCast(
        optree.get_input(0),
        precision=ML_AVX_m256_v8float32,
        tag="avx_conv_cast"
    )
    d1c1d0c0b1a1b0a0 = Permute(ymm0,
                               Constant(
                                   # Reorder [3, 2, 1, 0] -> [3, 1, 2, 0]
                                   int('3120', base = 4),
                                   precision = ML_Int32
                                   ),
                               precision = ymm0.get_precision()) # __m256
    __m256d_d1c1d0c0b1a1b0a0 = TypeCast(d1c1d0c0b1a1b0a0,
                                        precision = ML_AVX_m256_v4float64)
    __m128d_b1a1b0a0 = TypeCast(d1c1d0c0b1a1b0a0,
                                precision = ML_SSE_m128_v2float64)

    d1c1d0c0 = Extract(d1c1d0c0b1a1b0a0,
                       Constant(1, precision = ML_Int32),
                       precision = ML_SSE_m128_v4float32) # __m128
    d0c0b0a0 = VectorUnpack(
            __m128d_b1a1b0a0,
            TypeCast(d1c1d0c0, precision = ML_SSE_m128_v2float64),
            precision = ML_SSE_m128_v2float64
            ) # __m128d
    __m128i_d0c0b0a0 = TypeCast(d0c0b0a0, precision = ML_SSE_m128_v4int32)
    result = Conversion(__m128i_d0c0b0a0, precision = ML_AVX_m256_v4float64)
    return result

## AVX typecast metablock from 4 float32 to 2 float64
def _mm256_castps256_pd128(optree):
    ymm0 = optree.get_input(0)
    xmm0 = TypeCast(ymm0, precision=ML_SSE_m128_v4float32, tag = "castps256_lvl0")
    return TypeCast(xmm0, precision=ML_SSE_m128_v2float64, tag = "castps256_lvl1")

# AVX2 instructions
_mm256_max_epi32 = ImmIntrin("_mm256_max_epi32", arity = 2,
                             output_precision = ML_AVX_m256_v8int32)
# AVX2 bitwise AND of 256 bits representing integer data
_mm256_and_si256 = ImmIntrin("_mm256_and_si256", arity = 2,
                             output_precision = ML_AVX_m256_v8int32)



## check that list if made of only a single value replicated in each element
def uniform_list_check(value_list):
	return reduce((lambda acc, value: acc and value == value_list[0]), value_list, True)

## check whether @p optree is a uniform vector constant
def uniform_vector_constant_check(optree):
    if isinstance(optree, Constant) and not optree.get_precision() is None \
            and optree.get_precision().is_vector_format():
        return uniform_list_check(optree.get_value())
    else:
        return False

## check whether @p optree is a bit shift by a uniform vector constant
def uniform_shift_check(optree):
    if (isinstance(optree, BitLogicLeftShift)
            or isinstance(optree, BitLogicRightShift)
            or isinstance(optree, BitArithmeticRightShift)):
        return uniform_vector_constant_check(optree.get_input(1)) \
                or not optree.get_input(1).get_precision().is_vector_format()
    else:
        return False

## check whether @p optree is not a bit shift by a uniform vector constant
def variable_shift_check(optree):
    return not uniform_shift_check(optree)


## If optree is vector uniform constant modify it to be a
#  conversion between a scalar constant and a vector
def vector_constant_op(optree):
	assert isinstance(optree, Constant)
	cst_value_v = optree.get_value()
	op_format = optree.get_precision()
	if uniform_list_check(cst_value_v):
		scalar_format = op_format.get_scalar_format()
		scalar_cst = Constant(cst_value_v[0], precision = scalar_format)
		## TODO: Conversion class may be changed to VectorBroadCast
		return Conversion(scalar_cst, precision = op_format)
	else:
		raise NotImplementedError


def x86_fma_intrinsic_builder(intr_name):
    return _mm_cvtss_f32(
            FunctionOperator(
                intr_name, arity = 3,
                output_precision = ML_SSE_m128_v1float32,
                require_header = ["immintrin.h"])(
                    _mm_set_ss(FO_Arg(0)),
                    _mm_set_ss(FO_Arg(1)),
                    _mm_set_ss(FO_Arg(2))
                    )
            )
def x86_fmad_intrinsic_builder(intr_name):
    return _mm_cvtsd_f64(
            FunctionOperator(
                intr_name, arity = 3,
                output_precision = ML_SSE_m128_v1float64,
                require_header = ["immintrin.h"])(
                    _mm_set_sd(FO_Arg(0)),
                    _mm_set_sd(FO_Arg(1)),
                    _mm_set_sd(FO_Arg(2))
                    )
            )

## Builder for x86 FMA intrinsic within XMM register
# (native, no conversions)
#
def x86_fma_intr_builder_native(intr_name,
                                output_precision = ML_SSE_m128_v1float32):
    return FunctionOperator(intr_name, arity = 3,
                            output_precision = output_precision,
                            require_header = ["immintrin.h"]
                            )

def x86_fmad_intr_builder_native(intr_name,
                                 output_precision = ML_SSE_m128_v1float64):
    return FunctionOperator(intr_name, arity = 3,
                            output_precision = output_precision,
                            require_header = ["immintrin.h"]
                            )

## Convert a v4 to m128 conversion optree
def v4_to_m128_modifier(optree):
    conv_input = optree.get_input(0)
    elt_precision = conv_input.get_precision().get_scalar_format()

    elts = [VectorElementSelection(
        conv_input,
        Constant(i, precision = ML_Integer),
        precision = elt_precision
        ) for i in range(4)]
    return Conversion(elts[0], elts[1], elts[2], elts[3],
                      precision = optree.get_precision())

__m128ip_cast_operator = TemplateOperatorFormat(
        "(__m128i*){}", arity = 1,
        output_precision = ML_Pointer_Format(ML_SSE_m128_v4int32)
        )

_mm_fmadd_ss = x86_fma_intrinsic_builder("_mm_fmadd_ss")

def is_vector_cst_with_value(optree, value):
    if not isinstance(optree, Constant):
        return False
    else:
        return all(map(lambda v: v == value, optree.get_value()))

def build_format_constant(value, precision):
    """ Build a constant whose format is @p precision
        and set its value to @p value, possibly duplicating it if precision
        is a vector format """
    if precision.is_vector_format():
        return Constant([value] * precision.get_vector_size(), precision=precision)
    else:
        return Constant(value, precision=precision)

def pred_vector_select_mone_zero(optree):
    """ Predicate returns True if and only if
        optree is Select(cond, -1, 0) or Select(cond, 0, -1)
        False otherwise.
        Only returns True for integer-like format (-1 being equivalent
        to a mask fully set, and 0 to a mask fully cleared) """
    if not isinstance(optree, Select):
        return False
    elif not isinstance(optree.get_input(0), Comparison):
        return False
    elif not optree.get_precision().is_vector_format():
        return False
    else:
        lhs = optree.get_input(1)
        rhs = optree.get_input(2)
        cst_pred = (is_vector_cst_with_value(lhs, VIRTUAL_CST_MASK_M1) and is_vector_cst_with_value(rhs, VIRTUAL_CST_MASK_0)) or \
               (is_vector_cst_with_value(lhs, VIRTUAL_CST_MASK_0) and is_vector_cst_with_value(rhs, VIRTUAL_CST_MASK_M1))
        return cst_pred

def not_pred_vector_select_one_zero(optree):
    """ Negation of the predicate pred_vector_select_mone_zero """
    return not(pred_vector_select_mone_zero(optree))


def invert_comp_specifier(comp_specifier):
    """ return the opposite (logical negation) of @p comp_specifier """
    inverse_map = {
        Comparison.Equal: Comparison.NotEqual,
        Comparison.Less: Comparison.GreaterOrEqual,
        Comparison.LessOrEqual: Comparison.Greater,
        Comparison.NotEqual: Comparison.Equal,
        Comparison.Greater: Comparison.LessOrEqual,
        Comparison.GreaterOrEqual: Comparison.Less,
    }
    return inverse_map[comp_specifier]

def generate_sse_avx_select_boolean_value(cond, precision, negate=False):
    """ Generate a code generation operator for a comparison between two
        values stored in SSE/AVX registers and whose boolean result is casted
        to a value of same format as the operands (precision).
        Negate indicates that condition must be reversed.
        The Value 0 should be returned when cond is False
        and -1 when cond is True """
    assert isinstance(cond, Comparison)
    specifier_map = {
        Comparison.Equal: "eq",
        Comparison.GreaterOrEqual: "ge",
        Comparison.Greater: "gt",
        Comparison.NotEqual: "neq",
        Comparison.LessOrEqual: "le",
        Comparison.Less: "lt",
    }
    SIGNED_PREDICATE_LIST = [
        Comparison.GreaterOrEqual, Comparison.Greater,
        Comparison.Less, Comparison.LessOrEqual
    ]
    scalar_precision = precision.get_scalar_format()
    if is_std_unsigned_integer_format(scalar_precision) \
            and cond.specifier in SIGNED_PREDICATE_LIST:
        Log.report(Log.Warning,
                   "Generating code for unsigned comparison with signed " \
                           "specifier in generate_sse_avx_select_boolean_value")
    format_suffix = {
        ML_SSE_m128_v4int32: "epi32",
        ML_SSE_m128_v4uint32: "epi32",
        ML_SSE_m128_v4float32: "ps",

        ML_AVX_m256_v8int32: "epi32",
        ML_AVX_m256_v8uint32: "epi32",
        ML_AVX_m256_v8float32: "ps",
    }
    format_prefix = {
        ML_SSE_m128_v4int32: "mm",
        ML_SSE_m128_v4uint32: "mm",
        ML_SSE_m128_v4float32: "mm",

        ML_AVX_m256_v8int32: "mm256",
        ML_AVX_m256_v8uint32: "mm256",
        ML_AVX_m256_v8float32: "mm256",
    }
    intrinsic_builder = {
        ML_SSE_m128_v4int32: XmmIntrin,
        ML_SSE_m128_v4uint32: XmmIntrin,
        ML_SSE_m128_v4float32: XmmIntrin,

        ML_AVX_m256_v8int32: ImmIntrin,
        ML_AVX_m256_v8uint32: ImmIntrin,
        ML_AVX_m256_v8float32: ImmIntrin,
    }
    def opcode_builder(precision, cond_specifier):
        """ Build a function generator for comparison intrinsics
            which requires the comparison specifier to be encoded
            in the intrinsic mnemonic """
        return intrinsic_builder[precision](
                "_{}_cmp{}_{}".format(format_prefix[precision],
                                      specifier_map[cond_specifier],
                                      format_suffix[precision]),
                output_precision = precision, arity = 2
                )
    def specifier_op_builder(precision, cond_specifier):
        """ Build a function generator for comparison intrinsics
            which requires the comparison specifier to be encoded
            as an integer immediate value in the 3rd operand """
        # TODO/FIXME: cleaning with ordered/unordered predicate
        # default to ordered and non signaling when possible
        # https://www.felixcloutier.com/x86/CMPPD.html#tbl-3-1
        specifier_code = {
            Comparison.Equal: 16,
            Comparison.Less: 1,
            Comparison.LessOrEqual: 2,
            Comparison.NotEqual: 20,
            Comparison.Greater: 14,
            Comparison.GreaterOrEqual: 13,
        }[cond_specifier]
        return intrinsic_builder[precision](
                "_{}_cmp_{}".format(format_prefix[precision],
                                    format_suffix[precision]
                    ),
                arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: str(specifier_code)},
                output_precision = precision, arity = 2
                )
    mnemonic_builder = {
        ML_SSE_m128_v4int32: opcode_builder,
        ML_SSE_m128_v4uint32: opcode_builder,
        ML_SSE_m128_v4float32: opcode_builder,

        ML_AVX_m256_v8int32: opcode_builder,
        ML_AVX_m256_v8uint32: opcode_builder,
        ML_AVX_m256_v8float32: specifier_op_builder,

    }
    cond_specifier = cond.specifier if not negate \
            else invert_comp_specifier(cond_specifier)

    return mnemonic_builder[precision](precision, cond_specifier)


def expand_sse_avx_bool_comparison(optree):
    """ Expand a comparison between numeric values to a boolean output
        to a comparison with numeric format for result (supported by SSE/AVX)
        and a cast to boolean format to match operand prototype """
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    specifier = optree.specifier
    input_format = lhs.get_precision()
    return TypeCast(
        Comparison(lhs, rhs, specifier=specifier, precision=input_format),
        precision=optree.get_precision()
    )

def generate_sse_avx_comparison(optree):
    return generate_sse_avx_select_boolean_value(optree, optree.get_precision())

def squash_sse_avx_cst_select(optree):
    """ Convert Select(cond, 0, -1) into cond
        and Select(cond, -1, 0) into not(cond) """
    assert isinstance(optree, Select)
    cond = optree.get_input(0)
    lhs = optree.get_input(1)
    rhs = optree.get_input(2)
    op_prec = optree.get_precision()
    assert lhs.get_precision() == rhs.get_precision()
    # insert a TypeCast if select operands and optree formats do not match
    if op_prec != lhs.get_precision():
        result_prec = lhs.get_precision()
        wrapper = lambda op: TypeCast(op, precision=op_prec)
    else:
        result_prec = op_prec
        wrapper = lambda op: op

    cond_lhs = cond.get_input(0)
    cond_rhs = cond.get_input(1)
    new_cond = cond.copy(copy_map={cond_lhs: cond_lhs, cond_rhs: cond_rhs})
    cond_prec = cond_lhs.get_precision()
    new_cond.set_precision(cond_prec)
    if cond_prec != result_prec:
        # if condition and result formats do not match, we need to insert a cast
        new_cond = TypeCast(new_cond, precision=result_prec)

    if is_vector_cst_with_value(lhs, VIRTUAL_CST_MASK_M1) and is_vector_cst_with_value(rhs, VIRTUAL_CST_MASK_0):
        return wrapper(new_cond)
    elif is_vector_cst_with_value(lhs, VIRTUAL_CST_MASK_0) and is_vector_cst_with_value(rhs, VIRTUAL_CST_MASK_M1):
        new_cond.specifier = invert_comp_specifier(cond.specifier)
        return wrapper(new_cond)
    else:
        raise NotImplementedError


class VIRTUAL_CST_MASK:
    """ Virtual constant value for  mask,
        used to avoid confusion with constant with identical numerical value 
        but different physical values  when manipulating binary mask """
    
class VIRTUAL_CST_MASK_M1(VIRTUAL_CST_MASK):
    def __str__(self):
        return "VIRTUAL_CST_MASK_M1"
    def __repr__(self):
        return "VIRTUAL_CST_MASK_M1"

class VIRTUAL_CST_MASK_0(VIRTUAL_CST_MASK):
    def __str__(self):
        return "VIRTUAL_CST_MASK_0"
    def __repr__(self):
        return "VIRTUAL_CST_MASK_0"

def convert_select_to_logic(optree):
    """ Convert Select(cond, a, b) into
        LogicalOr(
            LogicalAnd(Select(cond, -1, 0), a),
            LogicalAnd(Select(cond, 0, -1, b))
        )
    """
    assert isinstance(optree, Select)
    cond = optree.get_input(0)
    lhs = optree.get_input(1)
    rhs = optree.get_input(2)
    prec = optree.get_precision()
    # determining integer precision
    C0 = build_format_constant(VIRTUAL_CST_MASK_0, prec)
    CM1 = build_format_constant(VIRTUAL_CST_MASK_M1, prec)
    result = BitLogicOr(
        BitLogicAnd(
            Select(
                cond,
                CM1,
                C0,
                precision=prec
            ),
            lhs,
            precision=prec
        ),
        BitLogicAnd(
            Select(
                cond,
                C0,
                CM1,
                precision=prec
            ),
            rhs,
            precision=prec
        ),
        precision=prec
    )
    Log.report(
        Log.Verbose,
        "legalizing Select(\n\t{},\n\t{}\n\t{}\n) into {}".format(
            cond, lhs, rhs,
            result
        )
    )
    return result


def linearize_2d_tableload(optree):
    """ convert TableLoad(table, index_d0, index_d1) into
        TableLoad(table, index_d0 * dim(table)[1] + index_d1 """
    assert isinstance(optree, TableLoad)
    table = optree.get_input(0)
    assert len(table.dimensions) >= 1
    index_0 = optree.get_input(1)
    index_1 = optree.get_input(2)
    index_prec = index_0.get_precision()
    prec = optree.get_precision()

    table_ptr = TypeCast(table, precision=ML_Pointer_Format(prec.get_scalar_format()))


    result = TableLoad(
        # TODO: TyoeCast to convert for multi-dim to linear table required
        table_ptr,
        Addition(
            Multiplication(
                index_0,
                build_format_constant(table.dimensions[1], precision=index_prec),
                precision=index_prec
            ),
            index_1,
            precision=index_prec
        ),
        precision=prec
    )
    Log.report(
        Log.Verbose,
        "legalizing {} into {}".format(
            optree.get_str(display_precision=True),
            result.get_str(display_precision=True)
        )
    )
    return result


def expand_sse_comparison(optree):
    """ SSE only supports eq/gt/lt predicate for integer comparison,
        thus all other must be expanded """
    assert isinstance(optree, Comparison)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    op_prec = optree.get_precision()
    if optree.specifier is Comparison.LessOrEqual:
        return BitLogicOr(
            Comparison(lhs, rhs, specifier=Comparison.Less, precision=op_prec),
            Comparison(lhs, rhs, specifier=Comparison.Equal, precision=op_prec),
            precision=op_prec
        )
    elif optree.specifier is Comparison.NotEqual:
        return BitLogicOr(
            Comparison(lhs, rhs, specifier=Comparison.Less, precision=op_prec),
            Comparison(lhs, rhs, specifier=Comparison.Greater, precision=op_prec),
            precision=op_prec
        )
    elif optree.specifier is Comparison.GreaterOrEqual:
        return BitLogicOr(
            Comparison(lhs, rhs, specifier=Comparison.Greater, precision=op_prec),
            Comparison(lhs, rhs, specifier=Comparison.Equal, precision=op_prec),
            precision=op_prec
        )
    else:
        raise NotImplementedError

# TODO refactor this asap with above function
def expand_avx2_comparison(optree):
    """ AVX2 only supports eq/gt predicates for integer comparison,
        thus all other must be legalized to use only those predicates  """
    assert isinstance(optree, Comparison)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    op_prec = optree.get_precision()
    if optree.specifier is Comparison.LessOrEqual:
        # x ≤ y <=> (x < y) || (x == y)
        return BitLogicOr(
            Comparison(lhs, rhs, specifier=Comparison.Less, precision=op_prec),
            Comparison(lhs, rhs, specifier=Comparison.Equal, precision=op_prec),
            precision=op_prec
        )
    elif optree.specifier is Comparison.Less:
        # cmplt x, y <=> cmpgt y, x
        return Comparison(rhs, lhs, specifier = Comparison.Greater,
                          precision = op_prec)
    elif optree.specifier is Comparison.NotEqual:
        # x ≠ y <=> !(x == y)
        return BitLogicNegate(
                Comparison(lhs, rhs, specifier = Comparison.Equal,
                           precision = op_prec),
                precision = op_prec
                )
    elif optree.specifier is Comparison.GreaterOrEqual:
        # x ≥ y <=> (x > y) || (x == y)
        return BitLogicOr(
                Comparison(lhs, rhs, specifier = Comparison.Greater,
                           precision = op_prec),
                Comparison(lhs, rhs, specifier = Comparison.Equal,
                           precision = op_prec),
                precision = op_prec
                )
    else:
        raise NotImplementedError


def expand_vec_mantissa_extraction(optree):
    """ Expand a vector MantissaExtraction operation into its
        And & Or counterparts """
    assert isinstance(optree, MantissaExtraction)
    op_in = optree.get_input(0)
    precision = optree.get_precision()
    bias = precision.get_scalar_format().get_bias()
    p = precision.get_scalar_format().get_precision()
    def build_vec_cst(cst_value, precision):
        vec_size = precision.get_vector_size()
        return Constant([cst_value] * vec_size, precision=precision)
    return BitLogicOr(
        BitLogicAnd(
            op_in,
            build_vec_cst(-(S2**(1 + bias) - S2**(1 + bias - p)), precision),
            precision=precision
        ),
        build_vec_cst(1.0, precision),
        precision=precision,
        tag="exp_mant_extraction"
    )


def error_raise_fct(optree):
    Log.report(
        Log.Error,
        "Generation of the following node is only supported by a dummy operator in x86 backend.: \n {}",
        optree
    )
    raise NotImplementedError

# Placeholder operator triggering an error when used for code generation
# This operator is generally used to declare a dummy implementation
# such that the m(128/256)_promotion passes assume that the node is supported
# and promotes it. The promoted node will later be optimized (for example select
# expansion) and this operator shall never be used
ERROR_OPERATOR = ComplexOperator(optree_modifier=error_raise_fct)


sse_c_code_generation_table = {
    Min: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    ImmIntrin("_mm_min_ps", arity = 2),
            },
        },
    },
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    ImmIntrin("_mm_max_ps", arity = 2),
            },
        },
    },
    Select: {
        None: {
            pred_vector_select_mone_zero: {
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
                    ComplexOperator(squash_sse_avx_cst_select),
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32):
                    ComplexOperator(squash_sse_avx_cst_select),
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    ComplexOperator(squash_sse_avx_cst_select),
            },
            not_pred_vector_select_one_zero: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    ComplexOperator(convert_select_to_logic),
            }
        },
    },
    MantissaExtraction: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    ComplexOperator(optree_modifier=expand_vec_mantissa_extraction),
            },
        }
    },
    Comparison: {
        Comparison.NotEqual: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
                    ComplexOperator(expand_sse_comparison),
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32):
                    ComplexOperator(expand_sse_comparison),
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
                    ComplexOperator(expand_sse_avx_bool_comparison),
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32):
                    ComplexOperator(expand_sse_avx_bool_comparison),
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
        Comparison.Equal: {
            lambda _: True: {
                type_strict_match_or_list([
                        (ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
        Comparison.LessOrEqual: {
            lambda _: True: {
                type_strict_match_or_list([
                        (ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32)
                    ]):
                    ComplexOperator(expand_sse_comparison),
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
        Comparison.Less: {
            lambda _: True: {
                type_strict_match_or_list([
                        (ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
        Comparison.Greater: {
            lambda _: True: {
                type_strict_match_or_list([
                        (ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
        Comparison.GreaterOrEqual: {
            lambda _: True: {
                type_strict_match_or_list([
                        (ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32)
                    ]):
                    ComplexOperator(expand_sse_comparison),
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32),
                        (ML_SSE_m128_v4bool, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32)]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            }
        },
    },
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(*(3*(ML_SSE_m128_v1float32,))):
                    _mm_add_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
                # vector addition
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_add_ps", arity = 2),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_and_ps", arity = 2,
                              output_precision = ML_SSE_m128_v4float32),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_or_ps", arity = 2,
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(*(3*(ML_SSE_m128_v2float64,))):
                    XmmIntrin("_mm_or_pd", arity = 2,
                              output_precision = ML_SSE_m128_v2float64),
            },
        },
    },
    Conversion: {
        None: {
            lambda _: True: {
                # not supported in SSE (else fallback on generic erroneous
                # implementation)
                type_strict_match(ML_Int32, ML_SSE_m128_v1int32):
                    ERROR_OPERATOR,
                type_strict_match(ML_UInt32, ML_SSE_m128_v1int32):
                    ERROR_OPERATOR,
                type_strict_match(ML_SSE_m128_v1float32, ML_Binary32):
                    _mm_set_ss,
                type_strict_match(ML_Binary32, ML_SSE_m128_v1float32):
                    _mm_cvtss_f32,
                type_strict_match(ML_SSE_m128_v1float64, ML_Binary64):
                    _mm_set_sd,
                type_strict_match(ML_Binary64, ML_SSE_m128_v1float64):
                    _mm_cvtsd_f64,
                # m128 float vector from ML's generic vector format
                type_strict_match(ML_SSE_m128_v4float32, v4float32):
                    XmmIntrin("_mm_load_ps", arity = 1,
                              output_precision = ML_SSE_m128_v4float32)(
                                  TemplateOperatorFormat(
                                      "GET_VEC_FIELD_ADDR({})", arity = 1,
                                      output_precision = ML_Pointer_Format(
                                          ML_Binary32
                                          )
                                      )
                                  ),
                # m128 float vector from ML's generic vector format
                type_strict_match(ML_SSE_m128_v4uint32, v4uint32):
                    XmmIntrin("_mm_load_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4uint32)(
                              __m128ip_cast_operator(
                                  TemplateOperatorFormat(
                                      "GET_VEC_FIELD_ADDR({})", arity = 1,
                                      output_precision = ML_Pointer_Format(
                                          ML_UInt32
                                          )
                                      )
                                  )),
                # m128 float vector to ML's generic vector format
                type_strict_match(v4float32, ML_SSE_m128_v4float32):
                    TemplateOperatorFormat(
                        "_mm_store_ps(GET_VEC_FIELD_ADDR({}), {})",
                        arity = 1,
                        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
                        require_header = ["xmmintrin.h"]
                        ),
                    #XmmIntrin("_mm_store_ps", arity = 2, arg_map = {0: FO_Result(0), 1: FO_Arg(0)})
                    #  (FunctionOperator("GET_VEC_FIELD_ADDR", arity = 1, output_precision = ML_Pointer_Format(ML_Binary32))(FO_Result(0)), FO_Arg(0)),
                type_strict_match(v4uint32, ML_SSE_m128_v4uint32):
                    TemplateOperatorFormat(
                        "_mm_store_si128((__m128i*)GET_VEC_FIELD_ADDR({}), {})",
                        arity = 1,
                        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
                        require_header = ["emmintrin.h"]
                        ),
                # signed integer format
                type_strict_match(ML_SSE_m128_v4int32, v4int32):
                    XmmIntrin("_mm_load_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4int32)(
                              __m128ip_cast_operator(
                                  TemplateOperatorFormat(
                                      "GET_VEC_FIELD_ADDR({})", arity = 1,
                                      output_precision = ML_Pointer_Format(
                                          ML_Int32
                                          )
                                      )
                                  )),
                type_strict_match(v4int32, ML_SSE_m128_v4int32):
                    TemplateOperatorFormat(
                        "_mm_store_si128((__m128i*)GET_VEC_FIELD_ADDR({}), {})",
                        arity = 1,
                        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
                        require_header = ["emmintrin.h"]
                        ),
                # identity operators
                lambda dst_type, src_type, **kwords: dst_type == src_type:
                    IdentityOperator(),
                # signed/unsigned conversions
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32):
                    TransparentOperator(),
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4int32):
                    TransparentOperator(),
            },
        },
    },
    ReciprocalSeed: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    _mm_rcp_ps(FO_Arg(0)),
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_rcp_ss(FO_Arg(0)),
                type_strict_match(ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(FO_Arg(0)))),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_mul_ps", arity = 2,
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(*(3*(ML_SSE_m128_v1float32,))):
                    _mm_mul_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
                # vector multiplication
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_mul_ps", arity = 2),
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    _mm_cvt_ss2si(_mm_set_ss(FO_Arg(0))),
            },
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                # Float negation
                type_strict_match(*(2*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_xor_ps", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm_set1_ps(-0.0f)", ML_SSE_m128_v4float32)
                    ),
            },
        },
    },
    Subtraction: {
        None: {
            lambda _: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4float32,))):
                    XmmIntrin("_mm_sub_ps", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v1float32,))):
                    XmmIntrin("_mm_sub_ss", arity = 2),
            },
        },
    },
    TableLoad: {
        None: {
            lambda optree: True: {
                # XMM version
                type_custom_match(FSM(ML_SSE_m128_v1float32),
                                  TCM(ML_TableFormat),
                                  FSM(ML_Int32)):
                    XmmIntrin("_mm_load_ss", arity = 1,
                        output_precision = ML_SSE_m128_v1float32)(
                            TemplateOperatorFormat(
                                "(float*)&{}[{}]", arity=2,
                                output_precision=ML_Pointer_Format(ML_Binary32)
                            )
                        ),
                type_custom_match(FSM(ML_SSE_m128_v1float32),
                                  TCM(ML_TableFormat),
                                  FSM(ML_SSE_m128_v1int32)):
                    XmmIntrin("_mm_load_ss", arity = 1,
                        output_precision = ML_SSE_m128_v1float32)(
                            TemplateOperatorFormat(
                                "(float*)&{}[_mm_cvtsi128_si32({})]", arity=2,
                                output_precision=ML_Pointer_Format(ML_Binary32)
                            )
                        ),
            },
        },
    },
    TypeCast: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4int32): IdentityOperator(),
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32): IdentityOperator(),
            },
        },
    },
}

sse2_c_code_generation_table = {
    Select: {
        None: {
            not_pred_vector_select_one_zero: {
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4bool, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
                    ComplexOperator(convert_select_to_logic),
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4bool, ML_SSE_m128_v4uint32, ML_SSE_m128_v4uint32):
                    ComplexOperator(convert_select_to_logic),
            },
        },
    },
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v1int32,))):
                    EmmIntrin("_mm_add_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_add_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    EmmIntrin("_mm_add_epi32", arity = 2),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_and_si128", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    EmmIntrin("_mm_and_si128", arity = 2),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_or_si128", arity = 2,
                              output_precision = ML_SSE_m128_v4int32),
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    EmmIntrin("_mm_or_si128", arity = 2,
                              output_precision = ML_SSE_m128_v4uint32),
                type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                    EmmIntrin("_mm_or_si128", arity = 2,
                              output_precision = ML_SSE_m128_v2int64),
                type_strict_match(*(3*(ML_SSE_m128_v2uint64,))):
                    EmmIntrin("_mm_or_si128", arity = 2,
                              output_precision = ML_SSE_m128_v2uint64),
            },
        },
    },
    BitLogicNegate: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
                    ImmIntrin("_mm_andnot_si128", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm_set1_epi32(-1)", ML_SSE_m128_v4int32)
                        ),
                    },
        },
    },
    LogicalNot: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4bool):
                    ImmIntrin("_mm_andnot_si128", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm_set1_epi32(-1)", ML_SSE_m128_v4bool)
                        ),
                    },
        },
    },
    LogicalOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4bool,))):
                    XmmIntrin("_mm_or_si128", arity = 2,
                              output_precision = ML_SSE_m128_v4bool),
            },
        },
    },
    LogicalAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4bool,))):
                    XmmIntrin("_mm_and_si128", arity = 2,
                              output_precision = ML_SSE_m128_v4bool),
            },
        },
    },
    BitLogicLeftShift: {
        None: {
            uniform_shift_check: {
                type_strict_match_list(
                    [ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32,],
                    [ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32,],
                    [ML_Int32, ML_UInt32,]):
                    EmmIntrin("_mm_slli_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)),
                # TODO the last argument is a scalar on 64 bits here, see
                # documentation on _mm_sll_epi32. We need to make sure that the
                # last vector is a constant that can be changed into either an
                # imm8 (above) or an ML_SSE_m128_v1[u]int64.
                type_strict_match_list(*(3*([ ML_SSE_m128_v4int32,
                    ML_SSE_m128_v4uint32, ],))):
                    EmmIntrin("_mm_sll_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)),
                type_strict_match_list(*(3*([ ML_SSE_m128_v2int64,
                    ML_SSE_m128_v2uint64, ],))):
                    EmmIntrin("_mm_sll_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)),
            },
        },
    },
    BitLogicRightShift: {
        None: {
            uniform_shift_check: {
                type_strict_match_list(
                    [ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32,],
                    [ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32,],
                    [ML_Int32, ML_UInt32,]):
                    EmmIntrin("_mm_srli_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)
                                  ),
                # TODO the last argument is a scalar here, see documentation on
                # _mm_srl_epi32. We need to make sure that the last vector is a
                # constant that can be changed into either an imm8 (above) or
                # an ML_SSE_m128_v1int32
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_srl_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)
                                  ),
                # TODO: using signed primitives for unsigned formats
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    EmmIntrin("_mm_srl_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)
                                  ),
            },
        },
    },
    BitArithmeticRightShift: {
        None: {
            uniform_shift_check: {
                type_strict_match_list(
                    [ ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32 ],
                    [ ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32 ],
                    [ ML_Int32, ML_UInt32 ]
                    ):
                    EmmIntrin("_mm_srai_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)
                                  ),
                # TODO the last argument is a scalar here, see documentation on
                # _mm_srl_epi32. We need to make sure that the last vector is a
                # constant that can be changed into either an imm8 (above) or
                # an ML_SSE_m128_v1int32
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_sra_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0), FO_Arg(1)
                                  ),
            },
        },
    },
    Conversion: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4int32):
                    EmmIntrin("_mm_cvtepi32_ps", arity = 1),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4float32):
                    EmmIntrin("_mm_cvtps_epi32", arity = 1),
                type_strict_match(ML_SSE_m128_v1int32, ML_Int32):
                    _mm_set1_epi32,
                type_strict_match(ML_Int32, ML_SSE_m128_v1int32):
                    EmmIntrin("_mm_cvtsi128_si32", arity = 1),
                type_strict_match(v4int32, ML_SSE_m128_v4int32):
                    TemplateOperatorFormat(
                        "_mm_store_si128((__m128i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True
                        ),
                type_strict_match(*((ML_SSE_m128_v4int32,) + 4*(ML_Int32,))):
                    XmmIntrin("_mm_set_epi32", arity = 4),
                #type_strict_match(ML_SSE_m128_v4int32, v4int32):
                #    ComplexOperator(optree_modifier = v4_to_m128_modifier),
                type_strict_match(ML_SSE_m128_v4int32, v4int32):
                    XmmIntrin(
                        "_mm_load_si128", arity = 1,
                        output_precision = ML_SSE_m128_v4int32
                        )(__m128ip_cast_operator(
                            TemplateOperatorFormat(
                                "GET_VEC_FIELD_ADDR({})", arity = 1,
                                output_precision = ML_Pointer_Format(ML_Int32)
                                )
                            )),
                # broadcast implemented as conversions
                type_strict_match(ML_SSE_m128_v4int32, ML_Int32):
                    XmmIntrin("_mm_set1_epi32", arity = 1),
                type_strict_match(ML_SSE_m128_v4float32, ML_Binary32):
                    XmmIntrin("_mm_set1_ps", arity = 1),
                # boolean vectors
                type_strict_match(v4bool, ML_SSE_m128_v4bool):
                    TemplateOperatorFormat(
                        "_mm_store_si128((__m128i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True
                        ),
                # dummy implementation
                type_strict_match(ML_SSE_m128_v4bool, v4bool):
                    XmmIntrin(
                        "_mm_load_si128", arity = 1,
                        output_precision = ML_SSE_m128_v4bool
                        )(__m128ip_cast_operator(
                            TemplateOperatorFormat(
                                "GET_VEC_FIELD_ADDR({})", arity = 1,
                                output_precision = ML_Pointer_Format(ML_Bool)
                                )
                            )),
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int64, ML_Binary64):
                    _mm_cvtsd_si64(_mm_set_sd(FO_Arg(0))),
                type_strict_match(ML_Int32, ML_Binary64):
                    _mm_cvtsd_si32(_mm_set_sd(FO_Arg(0))),
            },
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                # binary32 negation is in the X86_SSE_Processor
                type_strict_match(*(2*(ML_SSE_m128_v2float64,))):
                    EmmIntrin("_mm_xor_pd", arity = 2)(
                        FO_Value("_mm_set1_pd(-0.0f)", ML_SSE_m128_v2float64),
                        FO_Arg(0)
                    ),
                # Integer negation
                type_strict_match(*(2*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_sub_epi32", arity = 2)(
                        FO_Value("_mm_set1_epi32(0)", ML_SSE_m128_v4int32),
                        FO_Arg(0)
                    ),
                type_strict_match(*(2*(ML_SSE_m128_v2int64,))):
                    EmmIntrin("_mm_sub_epi64", arity = 2)(
                        FO_Value("_mm_set1_epi64x(0)", ML_SSE_m128_v2int64),
                        FO_Arg(0)
                    ),
            },
        },
    },
    Subtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin("_mm_sub_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                    EmmIntrin("_mm_sub_epi64", arity = 2),
            },
        },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                # 32-bit signed version
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4int32):
                    EmmIntrin("_mm_castsi128_ps", arity = 1,
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4float32):
                    EmmIntrin("_mm_castps_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4int32),
                type_strict_match(ML_SSE_m128_v4bool, ML_SSE_m128_v4float32):
                    EmmIntrin("_mm_castps_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4int32),
                # 32-bit unsigned version
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4uint32):
                    EmmIntrin("_mm_castsi128_ps", arity = 1,
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4float32):
                    EmmIntrin("_mm_castps_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4uint32),
                # 64-bit versions
                type_strict_match(ML_SSE_m128_v2float64, ML_SSE_m128_v2int64):
                    EmmIntrin("_mm_castsi128_pd", arity = 1,
                              output_precision = ML_SSE_m128_v2float64),
                type_strict_match(ML_SSE_m128_v2int64, ML_SSE_m128_v2float64):
                    EmmIntrin("_mm_castpd_si128", arity = 1,
                              output_precision = ML_SSE_m128_v2int64),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v2float64):
                    EmmIntrin("_mm_castpd_si128", arity = 1,
                              output_precision = ML_SSE_m128_v4int32),
                type_strict_match(ML_SSE_m128_v2float64, ML_SSE_m128_v4float32):
                    EmmIntrin("_mm_castps_pd", arity = 1,
                              output_precision = ML_SSE_m128_v2float64),
                # transparent cast
                type_strict_match(ML_SSE_m128_v4uint32, ML_SSE_m128_v4int32):
                    TransparentOperator(),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4uint32):
                    TransparentOperator(),
            },
        },
    },
    Constant: {
        None: {
            uniform_vector_constant_check: {
                type_strict_match(ML_SSE_m128_v4int32):
                    ComplexOperator(optree_modifier = vector_constant_op),
                type_strict_match(ML_SSE_m128_v4float32):
                    ComplexOperator(optree_modifier = vector_constant_op),
            },
        },
    },
    VectorUnpack: {
        VectorUnpack.Hi: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v2float64,))):
                    _mm_unpackhi_pd,
            },
        },
        VectorUnpack.Lo: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v2float64,))):
                    _mm_unpacklo_pd,
            },
        },
    },
}

## generates a check function which from a Constant vector node of vector_size
#  genrates a function which check that the constant value is uniform accross
#  every vector lane
def uniform_constant_check(optree):
	assert isinstance(optree, Constant)
	value_v = optree.get_value()
	init_value = value_v[0]
	return reduce(lambda acc, value: acc and (value == init_value), value_v, True)

sse3_c_code_generation_table = {}

ssse3_c_code_generation_table = {
    Negation: {
        None: {
            lambda optree: True: {
                # Float negation is handled by SSE2 instructions
                # 32-bit integer negation using SSSE3 sign_epi32 instruction
                type_strict_match(*(2*(ML_SSE_m128_v4int32,))):
                    TmmIntrin("_mm_sign_epi32", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm_set1_epi32(-1)", ML_SSE_m128_v4int32)
                    ),
            },
        },
    },
}

sse41_c_code_generation_table = {
    Test: {
        Test.IsMaskNotAnyZero: {
            lambda optree: True: {
                type_strict_match(ML_Bool, ML_SSE_m128_v4bool):
                    ImmIntrin("_mm_test_all_ones", arg_map={0: FO_Arg(0), 1: FO_Arg(0)}, arity=1), 
            },
        },
        Test.IsMaskAllZero: {
            lambda optree: True: {
                type_strict_match(ML_Bool, ML_SSE_m128_v4bool):
                    ImmIntrin("_mm_testz_si128", arg_map={0: FO_Arg(0), 1: FO_Arg(0)}, arity=1), 
            },
        },
    },
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    ImmIntrin("_mm_max_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    ImmIntrin("_mm_max_epu32", arity = 2),
            },
        },
    },
    Min: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    ImmIntrin("_mm_min_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4uint32,))):
                    ImmIntrin("_mm_min_epu32", arity = 2),
            },
        },
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    _mm_mullo_epi32,
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(*(2*(ML_SSE_m128_v1float32,))):
                    _mm_round_ss_rn,
                type_strict_match(*(2*(ML_SSE_m128_v1float64,))):
                    _mm_round_sd_rn,
                type_strict_match(ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_round_ss_rn(_mm_set_ss(FO_Arg(0)))),
                type_strict_match(ML_Binary64, ML_Binary64):
                    _mm_cvtsd_f64(_mm_round_sd_rn(_mm_set_sd(FO_Arg(0)))),
                type_strict_match(*(2*(ML_SSE_m128_v4float32,))):
                    SmmIntrin("_mm_round_ps", arity = 1,
                              arg_map = {0: FO_Arg(0),
                                         1: "_MM_FROUND_TO_NEAREST_INT"},
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4float32):
                  EmmIntrin("_mm_cvtps_epi32", arity = 1,
                            output_precision = ML_SSE_m128_v4int32)(
                                SmmIntrin(
                                    "_mm_round_ps", arity = 1,
                                    arg_map = {0: FO_Arg(0),
                                               1: "_MM_FROUND_TO_NEAREST_INT"},
                                    output_precision = ML_SSE_m128_v4float32)
                                ),
            },
        },
    },
}

sse42_c_code_generation_table = {}

def legalize_test_IsMaskNotAnyZero(optree):
    vmask = optree.get_input(0)
    legalized_node = Test(
        LogicalNot(vmask, precision=vmask.precision),
        specifier=Test.IsMaskAllZero,
        precision=optree.precision)
    forward_attributes(optree, legalized_node)
    return legalized_node

avx_c_code_generation_table = {
    Test: {
        Test.IsMaskNotAnyZero: {
            lambda optree: True: {
                type_strict_match(ML_Bool, ML_AVX_m256_v8bool):
                    ComplexOperator(optree_modifier=legalize_test_IsMaskNotAnyZero),
            },
        },
        Test.IsMaskAllZero: {
            lambda optree: True: {
                type_strict_match(ML_Bool, ML_AVX_m256_v8bool):
                    ImmIntrin("_mm256_testz_si256", arg_map={0: FO_Arg(0), 1: FO_Arg(0)}, arity=1), 
            },
        },
    },
    Min: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_min_ps", arity = 2),
            },
        },
    },
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_max_ps", arity = 2),
            },
        },
    },
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_add_ps", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_add_pd", arity = 2),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_and_ps", arity = 2,
                              output_precision = ML_AVX_m256_v8float32),
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_and_pd", arity = 2,
                              output_precision = ML_AVX_m256_v4float64),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_or_ps", arity = 2,
                              output_precision = ML_AVX_m256_v8float32),
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_or_pd", arity = 2,
                              output_precision = ML_AVX_m256_v4float64),
            },
        },
    },
    Constant: {
        None: {
            uniform_vector_constant_check: {
                type_strict_match_list([
                    ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32,
                    ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64,
                    ML_AVX_m256_v8float32, ML_AVX_m256_v4float64,
                    ]):
                    ComplexOperator(optree_modifier = vector_constant_op),
            },
        },
    },
    Conversion: {
        None: {
            lambda _: True: {
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8int32):
                    ImmIntrin("_mm256_cvtepi32_ps", arity = 1),
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_cvtps_epi32", arity = 1),
                type_strict_match(ML_AVX_m256_v8float32, v8float32):
                    ImmIntrin(
                        "_mm256_load_ps", arity = 1,
                        output_precision = ML_AVX_m256_v8float32)(
                            TemplateOperatorFormat(
                                "GET_VEC_FIELD_ADDR({})",
                                arity = 1,
                                output_precision = ML_Pointer_Format(
                                    ML_Binary32
                                    )
                                )
                            ),
                # __m256 float vector to ML's generic vector format
                type_strict_match(v8float32, ML_AVX_m256_v8float32):
                    TemplateOperatorFormat(
                        "_mm256_store_ps(GET_VEC_FIELD_ADDR({}), {})",
                        arity = 1,
                        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
                        require_header = ["immintrin.h"]
                        ),
                type_strict_match(ML_AVX_m256_v4float64, v4float64):
                    ImmIntrin(
                        "_mm256_load_pd", arity = 1,
                        output_precision = ML_AVX_m256_v4float64)(
                            TemplateOperatorFormat(
                                "GET_VEC_FIELD_ADDR({})",
                                arity = 1,
                                output_precision = ML_Pointer_Format(
                                    ML_Binary64
                                    )
                                )
                            ),
                type_strict_match(v4float64, ML_AVX_m256_v4float64):
                    TemplateOperatorFormat(
                        "_mm256_store_pd(GET_VEC_FIELD_ADDR({}), {})",
                        arity = 1,
                        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
                        require_header = ["immintrin.h"]
                        ),
                type_strict_match_list(
                    [v8int32, v8uint32],
                    [ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32]
                    ): TemplateOperatorFormat(
                        "_mm256_store_si256((__m256i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True,
                        require_header = ['immintrin.h']
                        ),
                type_strict_match(v4uint64, ML_AVX_m256_v4uint64): 
                    TemplateOperatorFormat(
                        "_mm256_store_si256((__m256i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True,
                        require_header = ['immintrin.h']
                        ),
                #type_strict_match(*((ML_SSE_m128_v4int32,) + 4*(ML_Int32,))):
                #    ImmIntrin("_mm256_set_epi32", arity = 4),
                type_strict_match_list([ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32], [v8int32, v8uint32]):
                    ImmIntrin(
                        "_mm256_load_si256", arity = 1,
                        output_precision = ML_AVX_m256_v8int32
                        )(TemplateOperatorFormat(
                            "(__m256i*){}", arity = 1,
                            output_precision = ML_Pointer_Format(
                                ML_AVX_m256_v8int32
                                )
                            )(
                                TemplateOperatorFormat(
                                    "GET_VEC_FIELD_ADDR({})", arity = 1,
                                    output_precision = ML_Pointer_Format(
                                        ML_Int32
                                        )
                                    )
                                )
                            ),
                type_strict_match(ML_AVX_m256_v4uint64, v4uint64):
                    ImmIntrin(
                        "_mm256_load_si256", arity = 1,
                        output_precision = ML_AVX_m256_v4uint64
                        )(TemplateOperatorFormat(
                            "(__m256i*){}", arity = 1,
                            output_precision = ML_Pointer_Format(
                                ML_AVX_m256_v4uint64
                                )
                            )(
                                TemplateOperatorFormat(
                                    "GET_VEC_FIELD_ADDR({})", arity = 1,
                                    output_precision = ML_Pointer_Format(
                                        ML_UInt64
                                        )
                                    )
                                )
                            ),
                type_strict_match(ML_AVX_m256_v8int32, ML_Int32):
                    ImmIntrin("_mm256_set1_epi32", arity = 1),
                type_strict_match(ML_AVX_m256_v8uint32, ML_UInt32):
                    ImmIntrin("_mm256_set1_epi32", arity = 1),
                type_strict_match(ML_AVX_m256_v8float32, ML_Binary32):
                    ImmIntrin("_mm256_set1_ps", arity = 1),
                type_strict_match(ML_AVX_m256_v4float64, ML_Binary64):
                    ImmIntrin("_mm256_set1_pd", arity = 1),
                type_strict_match(ML_AVX_m256_v4int64, ML_Int64):
                    ImmIntrin("_mm256_set1_epi64x", arity = 1),
                # TODO: CLEAN unsigned mix up
                type_strict_match(ML_AVX_m256_v4uint64, ML_UInt64):
                    ImmIntrin("_mm256_set1_epi64x", arity = 1),
                type_strict_match(ML_AVX_m256_v4int64, v4int64):
                    ImmIntrin(
                        "_mm256_load_si256", arity = 1,
                        output_precision = ML_AVX_m256_v4int64
                        )(TemplateOperatorFormat(
                            "(__m256i*){}", arity = 1,
                            output_precision = ML_Pointer_Format(
                                ML_AVX_m256_v4int64
                                )
                            )(
                                TemplateOperatorFormat(
                                    "GET_VEC_FIELD_ADDR({})", arity = 1,
                                    output_precision = ML_Pointer_Format(
                                        ML_Int64
                                        )
                                    )
                                )
                            ),
                type_strict_match(v4int64, ML_AVX_m256_v4int64):
                    TemplateOperatorFormat(
                        "_mm256_store_si256((__m256i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True,
                        require_header = ['immintrin.h']
                        ),
                type_strict_match(ML_AVX_m256_v4float64, ML_SSE_m128_v4int32):
                    _mm256_cvtepi32_pd,
                # signed/unsigned conversions
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32):
                    TransparentOperator(),
                type_strict_match(ML_AVX_m256_v8uint32, ML_AVX_m256_v8int32):
                    TransparentOperator(),
                # boolean conversion
                type_strict_match(v8bool, ML_AVX_m256_v8bool):
                    TemplateOperatorFormat(
                        "_mm256_store_si256((__m256i*){0}, {1})",
                        arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)},
                        void_function = True
                        ),
            },
            # AVX-based conversion of 4 int64 to 4 float64, valid if inputs fit
            # into 4 int32.
            lambda optree: optree.get_input(0).get_interval() is not None \
                    and optree.get_input(0).get_interval() \
                    in Interval(-2**31, 2**31 - 1): {
                type_strict_match(ML_AVX_m256_v4float64, ML_AVX_m256_v4int64):
                    ComplexOperator(
                        optree_modifier = conversion_to_avx_mm256_cvtepi64_pd
                    ),
            }
        },
    },
    Extract: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_AVX_m256_v8float32,
                                  ML_Int32):
                    _mm256_extractf128_ps,
            },
        },
    },
    ReciprocalSeed: {
        None: {
            lambda _: True: {
                type_strict_match(*(2*(ML_AVX_m256_v8float32,))):
                    _mm256_rcp_ps(FO_Arg(0)),
            },
        },
    },
    MantissaExtraction: {
        None: {
            lambda _: True: {
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    ComplexOperator(optree_modifier=expand_vec_mantissa_extraction),
                type_strict_match(ML_AVX_m256_v4float64, ML_AVX_m256_v4float64):
                    ComplexOperator(optree_modifier=expand_vec_mantissa_extraction),
            },
        }
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_mul_ps", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_mul_pd", arity = 2),
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_cvtps_epi32", arity = 1),
                type_strict_match(*(2*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_round_pd", arity=1,
                              arg_map={0: FO_Arg(0),
                                       1: "_MM_FROUND_TO_NEAREST_INT"},
                              output_precision=ML_AVX_m256_v4float64),
            },
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                # Float negation
                type_strict_match(*(2*(ML_AVX_m256_v8float32,))):
                    XmmIntrin("_mm256_xor_ps", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm256_set1_ps(-0.0f)",
                                 ML_AVX_m256_v8float32)
                        ),
                type_strict_match(*(2*(ML_AVX_m256_v4float64,))):
                    XmmIntrin("_mm256_xor_pd", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm256_set1_pd(-0.0)",
                                 ML_AVX_m256_v4float64)
                        ),
            },
        },
    },
    Permute: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8float32, ML_Int32):
                    _mm256_permute_ps,
            }
        },
    },
    Subtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8float32,))):
                    ImmIntrin("_mm256_sub_ps", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    ImmIntrin("_mm256_sub_pd", arity = 2),
            },
        },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                # binary32<->[u]int32
                type_strict_match_list(
                    [ML_AVX_m256_v8float32],
                    [ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32]
                    ): ImmIntrin("_mm256_castsi256_ps", arity = 1,
                                 output_precision = ML_AVX_m256_v8float32),
                # Signed output
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps_si256", arity = 1,
                              output_precision = ML_AVX_m256_v8int32),
                # Unsigned output
                type_strict_match(ML_AVX_m256_v8uint32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps_si256", arity = 1,
                              output_precision = ML_AVX_m256_v8uint32),
                # binary64<->[u]int64
                type_strict_match_list(
                    [ML_AVX_m256_v4float64],
                    [ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64]
                    ): ImmIntrin("_mm256_castsi256_pd", arity = 1,
                                 output_precision = ML_AVX_m256_v4float64),
                # Signed output
                type_strict_match(ML_AVX_m256_v4int64, ML_AVX_m256_v4float64):
                    ImmIntrin("_mm256_castpd_si256", arity = 1,
                              output_precision = ML_AVX_m256_v4int64),
                # Unsigned output
                type_strict_match(ML_AVX_m256_v4uint64, ML_AVX_m256_v4float64):
                    ImmIntrin("_mm256_castpd_si256", arity = 1,
                              output_precision = ML_AVX_m256_v4uint64),
                # YMM<->XMM (losing higher words)
                type_strict_match(ML_SSE_m128_v2float64, ML_AVX_m256_v4float64):
                    ImmIntrin("_mm256_castpd256_pd128", arity = 1,
                              output_precision = ML_SSE_m128_v2float64),
                type_strict_match(ML_SSE_m128_v4float32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps256_ps128", arity = 1,
                              output_precision = ML_SSE_m128_v4float32),
                # binary32 YMM-> binary64 XMM (losing 6 higher words)
                type_strict_match(ML_SSE_m128_v2float64, ML_AVX_m256_v8float32):
                    ComplexOperator(optree_modifier = _mm256_castps256_pd128),
                # transparent cast
                type_strict_match(ML_AVX_m256_v8uint32, ML_AVX_m256_v8int32):
                    TransparentOperator(),
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32):
                    TransparentOperator(),
                type_strict_match(ML_AVX_m256_v8bool, ML_AVX_m256_v8int32):
                    TransparentOperator(),
                type_strict_match(ML_AVX_m256_v8bool, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps_si256", arity = 1,
                              output_precision = ML_AVX_m256_v8bool),
            },
        },
    },
    VectorUnpack: {
        VectorUnpack.Hi: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    _mm256_unpackhi_pd,
            },
        },
        VectorUnpack.Lo: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v4float64,))):
                    _mm256_unpacklo_pd,
            },
        },
    },
}

## Generate a bit shift optree by an immediate value using @p optree inputs.
def generate_avx2_uniform_shift(optree):
    assert (isinstance(optree, BitArithmeticRightShift)
            or isinstance(optree, BitLogicLeftShift)
            or isinstance(optree, BitLogicRightShift)) \
                and isinstance(optree.get_input(1), Constant)

    shift_builder = optree.__class__
    return shift_builder(optree.get_input(0),
                         Constant(optree.get_input(1).get_value()[0],
                                  precision = ML_Int32),
                         precision = optree.get_precision())

avx2_c_code_generation_table = {
    Min: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_min_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_min_epu32", arity = 2),
            },
        },
    },
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_max_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_max_epu32", arity = 2),
            },
        },
    },
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match_or_list([ 3*(ML_AVX_m256_v8int32,),
                                            3*(ML_AVX_m256_v8uint32,) ]):
                    ImmIntrin("_mm256_add_epi32", arity = 2),
                type_strict_match_or_list([ 3*(ML_AVX_m256_v4int64,),
                                            3*(ML_AVX_m256_v4uint64,) ]):
                    ImmIntrin("_mm256_add_epi64", arity = 2),
            },
        },
    },
    BitLogicOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_or_si256", arity = 2,
                              output_precision = ML_AVX_m256_v8int32),
                type_strict_match(*(3*(ML_AVX_m256_v8uint32,))):
                    ImmIntrin("_mm256_or_si256", arity = 2,
                              output_precision = ML_AVX_m256_v8uint32),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    ImmIntrin("_mm256_or_si256", arity = 2,
                              output_precision = ML_AVX_m256_v4int64),
                type_strict_match(*(3*(ML_AVX_m256_v4uint64,))):
                    ImmIntrin("_mm256_or_si256", arity = 2,
                              output_precision = ML_AVX_m256_v4uint64),
            },
        },
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_mullo_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    ImmIntrin("_mm256_mullo_epi64", arity = 2),
            },
        },
    },
    BitArithmeticRightShift: {
        None: {
            uniform_shift_check: {
                # Shift by a constant less than the input bitsize
                # TODO check that constant value is in the valid range.
                type_strict_match_list(
                    *(2*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32, ],)
                        + ([ML_Int32, ML_UInt32 ],))
                    ):
                    ImmIntrin("_mm256_srai_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match_list(
                    *(2*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64, ],)
                        + ([ML_Int32, ML_UInt32 ],))
                    ):
                    ImmIntrin("_mm256_srai_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                # Modify optree to use an immediate value (better performance)
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
            },
            variable_shift_check: {
                # XMM version. Note: there's no YMM intrinsic available for
                # 64-bit integer arithmetic right shift. This comes with
                # AVX-512.
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    ImmIntrin("_mm_srav_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_srav_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            },
        },
    },
    BitLogicAnd: {
        None: {
            lambda optree: True: {
                type_strict_match_list(*(3*([
                    ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32,
                    ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64,
                    ML_AVX_m256_v4uint32, ML_AVX_m256_v4int32,
                    ],))):
                    _mm256_and_si256,
            },
        },
    },
    BitLogicLeftShift: {
        None: {
            uniform_shift_check: {
                # Shift by a constant less than the input bitsize
                # TODO check that constant value is in the valid range.
                type_strict_match_list(*(2*([ ML_AVX_m256_v8int32,
                    ML_AVX_m256_v8uint32, ],) + ([ML_Int32, ML_UInt32 ],))):
                    ImmIntrin("_mm256_slli_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match_list(*(2*([ ML_AVX_m256_v4int64,
                    ML_AVX_m256_v4uint64, ],) + ([ML_Int32, ML_UInt32 ],))):
                    ImmIntrin("_mm256_slli_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                # Shift by uniform vector constant (higher latency)
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
            },
            variable_shift_check: {
                # Variable bit shift is only available with AVX2
                # XMM version
                type_strict_match_list(*(3*([ ML_SSE_m128_v4int32,
                    ML_SSE_m128_v4uint32, ],))):
                    ImmIntrin("_mm_sllv_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match_list(*(3*([ ML_SSE_m128_v2int64,
                    ML_SSE_m128_v2uint64, ],))):
                    ImmIntrin("_mm_sllv_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                # YMM version
                type_strict_match_list(*(3*([ ML_AVX_m256_v8int32,
                    ML_AVX_m256_v8uint32, ],))):
                    ImmIntrin("_mm256_sllv_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match_list(*(3*([ ML_AVX_m256_v4int64,
                    ML_AVX_m256_v4uint64, ],))):
                    ImmIntrin("_mm256_sllv_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            },
        },
    },
    BitLogicNegate: {
        None: {
            lambda _: True: {
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8int32):
                    ImmIntrin("_mm256_andnot_si256", arity = 2)(
                              FO_Arg(0),
                              FO_Value("_mm256_set1_epi32(-1)",
                                       ML_AVX_m256_v8int32)
                              ),
                type_strict_match(ML_AVX_m256_v4int64, ML_AVX_m256_v4int64):
                    ImmIntrin("_mm256_andnot_si256", arity = 2)(
                              FO_Arg(0),
                              FO_Value("_mm256_set1_epi64x(-1)",
                                       ML_AVX_m256_v4int64)
                              ),
            },
        },
    },
    LogicalNot: {
        None: {
            lambda _: True: {
                type_strict_match(ML_AVX_m256_v8bool, ML_AVX_m256_v8bool):
                    ImmIntrin("_mm256_andnot_si256", arity = 2)(
                              FO_Arg(0),
                              FO_Value("_mm256_set1_epi32(-1)",
                                       ML_AVX_m256_v8bool)
                              ),
            },
        },
    },
    LogicalAnd: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8bool,))):
                    ImmIntrin("_mm256_and_si256", arity = 2,
                              output_precision = ML_AVX_m256_v8bool),
            },
        },
    },
    LogicalOr: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8bool,))):
                    ImmIntrin("_mm256_or_si256", arity = 2,
                              output_precision = ML_AVX_m256_v8bool),
            },
        },
    },
    BitLogicRightShift: {
        None: {
            uniform_shift_check: {
                # Shift by a constant less than the input bitsize
                # TODO check that constant value is in the valid range.
                type_strict_match_list(
                    *(2*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32, ],)
                        + ([ML_Int32, ML_UInt32 ],))
                    ):
                    ImmIntrin("_mm256_srli_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match_list(
                    *(2*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64, ],)
                        + ([ML_Int32, ML_UInt32 ],))):
                    ImmIntrin("_mm256_srli_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                # Shift by uniform vector constant (higher latency)
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64, ],))
                    ):
                    ComplexOperator(generate_avx2_uniform_shift),
            },
            variable_shift_check: {
                # XMM version
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    ImmIntrin("_mm_srlv_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                    ImmIntrin("_mm_srlv_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                # YMM version
                type_strict_match_list(
                    *(3*([ ML_AVX_m256_v8int32, ML_AVX_m256_v8uint32 ],))
                    ):
                    ImmIntrin("_mm256_srlv_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match(
                    *(3*([ ML_AVX_m256_v4int64, ML_AVX_m256_v4uint64 ],))
                    ):
                    ImmIntrin("_mm256_srlv_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            },
        },
    },
    Comparison: {
        Comparison.NotEqual: {
            lambda _: True: {
                type_strict_match_or_list([ 3*(ML_AVX_m256_v8int32,),
                                            3*(ML_AVX_m256_v8uint32,),
                                            3*(ML_AVX_m256_v4int64,),
                                            3*(ML_AVX_m256_v4uint64,) ]):
                    ComplexOperator(expand_avx2_comparison),
                # 3 Dummy operators used to allow m256_promotion to promote squashable comparison
                type_strict_match(ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32):
                    ERROR_OPERATOR,
                type_strict_match(ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32):
                    ERROR_OPERATOR,
            },
        },
        Comparison.Equal: {
            lambda _: True: {
                type_strict_match_or_list([ 3*(ML_AVX_m256_v8int32,),
                                            3*(ML_AVX_m256_v8uint32,),
                                            3*(ML_AVX_m256_v8float32,)]):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m256_promotion to promote
                # squashable comparison
                type_strict_match_or_list([
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32),
                    ]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            },
        },
        Comparison.Greater: {
            lambda _: True: {
                type_strict_match_or_list([ 3*(ML_AVX_m256_v8int32,),
                                            3*(ML_AVX_m256_v8uint32,),
                                            3*(ML_AVX_m256_v8float32,),]):
                    # TODO rename generate_sse_avx_comparison to reflect AVX
                    # support?
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m256_promotion to promote
                # squashable comparison
                type_strict_match_or_list([
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32),
                    ]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            },
        },
        Comparison.GreaterOrEqual: {
            lambda _: True: {
                type_strict_match_or_list([
                    (ML_AVX_m256_v8int32, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    ]):
                    ComplexOperator(expand_avx2_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32),
                    ]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    DynamicOperator(generate_sse_avx_comparison),
            },
        },
        Comparison.Less: {
            lambda _: True: {
                type_strict_match_or_list([ 3*(ML_AVX_m256_v8int32,),
                                            3*(ML_AVX_m256_v8uint32,) ]):
                    ComplexOperator(expand_avx2_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote
                # squashable comparison
                type_strict_match_or_list([
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32),
                    ]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    DynamicOperator(generate_sse_avx_comparison),
            },
        },
        Comparison.LessOrEqual: {
            lambda _: True: {
                type_strict_match_or_list([
                    (ML_AVX_m256_v8int32, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32)
                    ]):
                    ComplexOperator(expand_avx2_comparison),
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    DynamicOperator(generate_sse_avx_comparison),
                # 3 Dummy operators used to allow m128_promotion to promote squashable comparison
                type_strict_match_or_list([
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8int32, ML_AVX_m256_v8int32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32),
                    (ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32),
                    ]):
                    ComplexOperator(expand_sse_avx_bool_comparison),
            },
        },
    },
    CountLeadingZeros: {
        None: {
            lambda _: True: {
                type_strict_match(ML_UInt32, ML_UInt32):
                    _lzcnt_u32(FO_Arg(0)),
                type_strict_match(ML_UInt64, ML_UInt64):
                    _lzcnt_u64(FO_Arg(0)),
            },
        },
    },
    FusedMultiplyAdd: {
        FusedMultiplyAdd.Standard: {
            lambda optree: True: {
                # Scalar version
                type_strict_match(*(4*(ML_SSE_m128_v1float32,))):
                    x86_fma_intr_builder_native("_mm_fmadd_ss"),
                type_strict_match(*(4*(ML_SSE_m128_v1float64,))):
                    x86_fmad_intr_builder_native("_mm_fmadd_sd"),
                # XMM version
                type_strict_match(*(4*(ML_SSE_m128_v4float32,))):
                    x86_fma_intr_builder_native(
                        "_mm_fmadd_ps",
                        output_precision = ML_SSE_m128_v4float32
                        ),
                type_strict_match(*(4*(ML_SSE_m128_v2float64,))):
                    x86_fmad_intr_builder_native(
                        "_mm_fmadd_ps",
                        output_precision = ML_SSE_m128_v2float64
                        ),
                # YMM version
                type_strict_match(*(4*(ML_AVX_m256_v8float32,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fmadd_ps",
                        output_precision = ML_AVX_m256_v8float32
                        ),
                type_strict_match(*(4*(ML_AVX_m256_v4float64,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fmadd_pd",
                        output_precision = ML_AVX_m256_v4float64
                        ),
            },
        },
        FusedMultiplyAdd.Subtract: {
            lambda optree: True: {
                # Scalar version
                type_strict_match(*(4*(ML_SSE_m128_v1float32,))):
                    x86_fma_intr_builder_native("_mm_fmsub_ss"),
                type_strict_match(*(4*(ML_SSE_m128_v1float64,))):
                    x86_fmad_intr_builder_native("_mm_fmsub_sd"),
                # XMM version
                type_strict_match(*(4*(ML_SSE_m128_v4float32,))):
                    x86_fma_intr_builder_native(
                        "_mm_fmsub_ps",
                        output_precision = ML_SSE_m128_v4float32
                        ),
                type_strict_match(*(4*(ML_SSE_m128_v2float64,))):
                    x86_fmad_intr_builder_native(
                        "_mm_fmsub_ps",
                        output_precision = ML_SSE_m128_v2float64
                        ),
                # YMM version
                type_strict_match(*(4*(ML_AVX_m256_v8float32,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fmsub_ps",
                        output_precision = ML_AVX_m256_v8float32
                        ),
                type_strict_match(*(4*(ML_AVX_m256_v4float64,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fmsub_pd",
                        output_precision = ML_AVX_m256_v4float64
                        ),
                },
        },
        FusedMultiplyAdd.SubtractNegate: {
            lambda optree: True: {
            # Scalar version
            type_strict_match(*(4*(ML_SSE_m128_v1float32,))):
                x86_fma_intr_builder_native("_mm_fnmadd_ss"),
            type_strict_match(*(4*(ML_SSE_m128_v1float64,))):
                x86_fmad_intr_builder_native("_mm_fnmadd_sd"),
            # XMM version
            type_strict_match(*(4*(ML_SSE_m128_v4float32,))):
                x86_fma_intr_builder_native(
                    "_mm_fnmadd_ps",
                    output_precision = ML_SSE_m128_v4float32
                    ),
            type_strict_match(*(4*(ML_SSE_m128_v2float64,))):
                x86_fmad_intr_builder_native(
                    "_mm_fnmadd_ps",
                    output_precision = ML_SSE_m128_v2float64
                    ),
            # YMM version
            type_strict_match(*(4*(ML_AVX_m256_v8float32,))):
                x86_fma_intr_builder_native(
                    "_mm256_fnmadd_ps",
                    output_precision = ML_AVX_m256_v8float32
                    ),
            type_strict_match(*(4*(ML_AVX_m256_v4float64,))):
                x86_fma_intr_builder_native(
                    "_mm256_fnmadd_pd",
                    output_precision = ML_AVX_m256_v4float64
                    ),
            },
        },
        FusedMultiplyAdd.Negate: {
            lambda optree: True: {
                # Scalar version
                type_strict_match(*(4*(ML_SSE_m128_v1float32,))):
                    x86_fma_intr_builder_native("_mm_fnmsub_ss"),
                type_strict_match(*(4*(ML_SSE_m128_v1float64,))):
                    x86_fmad_intr_builder_native("_mm_fnmsub_sd"),
                # XMM version
                type_strict_match(*(4*(ML_SSE_m128_v4float32,))):
                    x86_fma_intr_builder_native(
                        "_mm_fnmsub_ps",
                        output_precision = ML_SSE_m128_v4float32
                        ),
                type_strict_match(*(4*(ML_SSE_m128_v2float64,))):
                    x86_fmad_intr_builder_native(
                        "_mm_fnmsub_ps",
                        output_precision = ML_SSE_m128_v2float64
                        ),
                # YMM version
                type_strict_match(*(4*(ML_AVX_m256_v8float32,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fnmsub_ps",
                        output_precision = ML_AVX_m256_v8float32
                        ),
                type_strict_match(*(4*(ML_AVX_m256_v4float64,))):
                    x86_fma_intr_builder_native(
                        "_mm256_fnmsub_pd",
                        output_precision = ML_AVX_m256_v4float64
                        ),
            },
        },
    },
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    _mm256_max_epi32,
            }
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                # Integer negation
                type_strict_match(*(2*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_sub_epi32", arity = 2)(
                        FO_Value("_mm256_set1_epi32(0)", ML_AVX_m256_v8int32),
                        FO_Arg(0)
                    ),
                type_strict_match(*(2*(ML_AVX_m256_v4int64,))):
                    EmmIntrin("_mm256_sub_epi64", arity = 2)(
                        FO_Value("_mm256_set1_epi64x(0)", ML_AVX_m256_v4int64),
                        FO_Arg(0)
                    ),
            },
        },
    },
    Select: {
        None: {
            pred_vector_select_mone_zero: {
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8bool,
                                  ML_AVX_m256_v8int32, ML_AVX_m256_v8int32):
                    # TODO update the naming to reflect AVX support
                    ComplexOperator(squash_sse_avx_cst_select),
                type_strict_match(ML_AVX_m256_v8uint32, ML_AVX_m256_v8bool,
                                  ML_AVX_m256_v8uint32, ML_AVX_m256_v8uint32):
                    # TODO update the naming to reflect AVX support
                    ComplexOperator(squash_sse_avx_cst_select),
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8bool,
                                  ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    # TODO update the naming to reflect AVX support
                    ComplexOperator(squash_sse_avx_cst_select),
            },
            not_pred_vector_select_one_zero: {
                type_strict_match(ML_AVX_m256_v8float32, ML_AVX_m256_v8bool, ML_AVX_m256_v8float32, ML_AVX_m256_v8float32):
                    ComplexOperator(convert_select_to_logic),
            }
        },
    },
    Subtraction: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_sub_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    ImmIntrin("_mm256_sub_epi64", arity = 2),
            },
        },
    },
    TableLoad: {
        None: {
            lambda optree: True: {
                # XMM version with 32-bit indices
                type_custom_match(FSM(ML_SSE_m128_v4float32),
                                  TCLM([ML_TableFormat, ML_Pointer_Format]),
                                  type_strict_match_list([
                                      ML_SSE_m128_v4uint32,
                                      ML_SSE_m128_v4int32,
                                      ])):
                    ImmIntrin("_mm_i32gather_ps", arity = 3,
                        output_precision = ML_SSE_m128_v4float32)(
                            FO_Arg(0),
                            FO_Arg(1),
                            FO_Value("4", ML_Int32)
                            ),
                type_custom_match(FSM(ML_SSE_m128_v2float64),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_SSE_m128_v2uint32,
                                      ML_SSE_m128_v2int32,
                                      ])):
                    ImmIntrin("_mm_i32gather_pd", arity = 3,
                        output_precision = ML_SSE_m128_v4float32)(
                            FO_Arg(0),
                            FO_Arg(1),
                            FO_Value("8", ML_Int32)
                            ),
                # XMM version with 32-bit indices and 2 index table
                type_custom_match(FSM(ML_SSE_m128_v4float32),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_SSE_m128_v4uint32,
                                      ML_SSE_m128_v4int32,
                                      ]),
                                  type_strict_match_list([
                                      ML_SSE_m128_v4uint32,
                                      ML_SSE_m128_v4int32,
                                      ])
                                      
                                ):
                    ComplexOperator(linearize_2d_tableload),
                # YMM version with 32-bit indices
                type_custom_match(FSM(ML_AVX_m256_v8float32),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_AVX_m256_v8uint32,
                                      ML_AVX_m256_v8int32,
                                      ])):
                    ImmIntrin("_mm256_i32gather_ps", arity = 3,
                              output_precision = ML_AVX_m256_v8float32)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("4", ML_Int32)
                                  ),
                type_custom_match(FSM(ML_AVX_m256_v4float64),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_AVX_m256_v4uint32,
                                      ML_AVX_m256_v4int32,
                                      ])):
                    ImmIntrin("_mm256_i32gather_pd", arity = 3,
                              output_precision = ML_AVX_m256_v4float64)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("8", ML_Int32)
                                  ),
                # XMM version with 64-bit indices
                type_custom_match(FSM(ML_SSE_m128_v2float32),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_SSE_m128_v2uint64,
                                      ML_SSE_m128_v2int64,
                                      ])):
                    ImmIntrin("_mm_i64gather_ps", arity = 3,
                              output_precision = ML_SSE_m128_v2float32)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("4", ML_Int32)
                                  ),
                type_custom_match(FSM(ML_SSE_m128_v2float64),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_SSE_m128_v2uint64,
                                      ML_SSE_m128_v2int64,
                                      ])):
                    ImmIntrin("_mm_i64gather_pd", arity = 3,
                              output_precision = ML_SSE_m128_v2float64)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("8", ML_Int32)
                                  ),
                # YMM version with 64-bit indices
                type_custom_match(FSM(ML_SSE_m128_v4float32),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_AVX_m256_v4uint64,
                                      ML_AVX_m256_v4int64,
                                      ])):
                    ImmIntrin("_mm256_i64gather_ps", arity = 3,
                              output_precision = ML_SSE_m128_v4float32)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("4", ML_Int32)
                                  ),
                type_custom_match(FSM(ML_AVX_m256_v4float64),
                                  TCM(ML_TableFormat),
                                  type_strict_match_list([
                                      ML_AVX_m256_v4uint64,
                                      ML_AVX_m256_v4int64,
                                      ])):
                    ImmIntrin("_mm256_i64gather_pd", arity = 3,
                              output_precision = ML_AVX_m256_v4float64)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("8", ML_Int32)
                                  ),
            },
        },
    },
}

avx512_c_code_generation_table = {
    BitArithmeticRightShift: {
        None: {
            lambda optree: True: {
                # YMM version. The XMM version is available since AVX2.
                type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                    ImmIntrin("_mm_srav_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    ImmIntrin("_mm256_srav_epi64", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            },
        },
    },
}

#rdtsc_operator = AsmInlineOperator(
#  "__asm volatile ( \"xor %%%%eax, %%%%eax\\n\"\n \"CPUID\\n\" \n\"rdtsc\\n\"\n : \"=A\"(%s));",
#  arg_map = {0: FO_Result(0)},
#  arity = 0
#)

rdtsc_operator = AsmInlineOperator(
"""{
    uint32_t cycles_hi = 0, cycles_lo = 0;
    asm volatile (
        "cpuid\\n\\t"
        "rdtsc\\n\\t"
        "mov %%%%edx, %%0\\n\\t"
        "mov %%%%eax, %%1\\n\\t"
        : "=r" (cycles_hi), "=r" (cycles_lo)
        :: "%%rax", "%%rbx", "%%rcx", "%%rdx");
    %s = ((uint64_t) cycles_hi << 32) | cycles_lo;
}""",
    arg_map = {0: FO_Result(0)},
    arity = 0
)

x86_c_code_generation_table = {
    SpecificOperation: {
        SpecificOperation.ReadTimeStamp: {
            lambda _: True: {
                type_strict_match(ML_Int64): rdtsc_operator
            }
        }
    },
}


class X86_Processor(VectorBackend):
    target_name = "x86"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_Processor)

    code_generation_table = {
        C_Code: x86_c_code_generation_table,
    }

    def __init__(self):
        VectorBackend.__init__(self)

    def get_current_timestamp(self):
        return SpecificOperation(
                specifier = SpecificOperation.ReadTimeStamp,
                precision = ML_Int64
                )


class X86_SSE_Processor(X86_Processor):
    target_name = "x86_sse"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSE_Processor)

    code_generation_table = {
        C_Code: sse_c_code_generation_table,
    }

    # approximation table map
    approx_table_map = x86_sse_approx_table_map

    TARGET_PREFERRED_VECTOR_SIZE_MAP = {
        ML_Binary32: 4,
        ML_Binary64: 2,
        ML_Int32: 4,
        ML_Int64: 2
    }


    def __init__(self):
        super(X86_SSE_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_SSE_Processor, self).get_compilation_options() \
                + ['-msse']

    def get_preferred_sub_vector_size(self, scalar_precision, vector_size):
        """ Returns the target preferred sub-vector size for a vector of size
            @p vector_size and of scalar precision @p scalar_precision """
        if scalar_precision in self.TARGET_PREFERRED_VECTOR_SIZE_MAP:
            return min(self.TARGET_PREFERRED_VECTOR_SIZE_MAP[scalar_precision], vector_size)
        else:
            return super(X86_AVX_Processor, self).get_preferred_sub_vector_size(
                scalar_precision, vector_size
            )


class X86_SSE2_Processor(X86_SSE_Processor):
    target_name = "x86_sse2"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSE2_Processor)

    code_generation_table = {
        C_Code: sse2_c_code_generation_table,
    }

    def __init__(self):
        super(X86_SSE2_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_SSE2_Processor, self).get_compilation_options() \
                + ['-msse2']


class X86_SSE3_Processor(X86_SSE2_Processor):
    target_name = "x86_sse3"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSE3_Processor)

    code_generation_table = {
        C_Code: sse3_c_code_generation_table,
    }

    def get_compilation_options(self):
        return super(X86_SSE3_Processor, self).get_compilation_options() \
                + ['-msse3']


class X86_SSSE3_Processor(X86_SSE3_Processor):
    target_name = "x86_ssse3"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSSE3_Processor)

    code_generation_table = {
        C_Code: ssse3_c_code_generation_table,
    }

    def __init__(self):
        super(X86_SSSE3_Processor, self).__init__()

    def get_compilation_options(self):
      return super(X86_SSSE3_Processor, self).get_compilation_options() \
              + ['-mssse3']


class X86_SSE41_Processor(X86_SSSE3_Processor):
    target_name = "x86_sse41"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSE41_Processor)

    code_generation_table = {
        C_Code: sse41_c_code_generation_table,
    }

    def __init__(self):
        super(X86_SSE41_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_SSE41_Processor, self).get_compilation_options() \
                + ['-msse4.1']


class X86_SSE42_Processor(X86_SSE41_Processor):
    target_name = "x86_sse42"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_SSE42_Processor)

    code_generation_table = {
        C_Code: sse42_c_code_generation_table,
    }

    def __init__(self):
        super(X86_SSE42_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_SSE42_Processor, self).get_compilation_options() \
                + ['-msse4.2']


class X86_AVX_Processor(X86_SSE42_Processor):
    target_name = "x86_avx"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_AVX_Processor)

    code_generation_table = {
        C_Code: avx_c_code_generation_table,
    }
    TARGET_PREFERRED_VECTOR_SIZE_MAP = {
        ML_Binary32: 8,
        ML_Binary64: 4,
        ML_Int32: 8,
        ML_Int64: 4
    }

    def __init__(self):
        super(X86_AVX_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_AVX_Processor, self).get_compilation_options() \
                + ['-mavx']



class X86_AVX2_Processor(X86_AVX_Processor):
    target_name = "x86_avx2"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_AVX2_Processor)

    code_generation_table = {
        C_Code: avx2_c_code_generation_table,
    }

    def __init__(self):
        super(X86_AVX2_Processor, self).__init__()

    def get_compilation_options(self):
      return super(X86_AVX2_Processor, self).get_compilation_options() \
              + ["-mfma", "-mavx2"]


# debug message
Log.report(LOG_BACKEND_INIT, "initializing INTEL targets")
