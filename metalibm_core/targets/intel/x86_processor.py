# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014)
# All rights reserved
# created:          Apr 11th,  2014
# last-modified:    Nov  6th,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.utility.log_report import Log
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.complex_generator import ComplexOperator
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_operations import *
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_table import ML_TableFormat
from metalibm_core.core.polynomials import is_cst_with_value

from metalibm_core.targets.common.vector_backend import VectorBackend

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.generic_processor import GenericProcessor

## TODO; change ML_SSE and ML_AVX format to be vector formats

def get_sse_scalar_cst(format_object, value, language = C_Code):
	base_format = format_object.get_base_format()
	return "{{{}}}/*sse*/".format(base_format.get_cst(value, language))
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

## format for a single fp32 stored in a XMM 128-bit register
ML_SSE_m128_v1float32 = VirtualFormatNoForward(ML_Binary32, ML_SSE_m128, get_sse_scalar_cst, True)
## format for single 1 fp64 in a XMM 128-bit register
ML_SSE_m128_v1float64 = VirtualFormatNoForward(ML_Binary64, ML_SSE_m128d, get_sse_scalar_cst, True)
## format for a single int32 stored in a XMM 128-bit register
ML_SSE_m128_v1int32  = VirtualFormatNoForward(ML_Int32, ML_SSE_m128i, get_sse_scalar_cst, True)
## format for single 1 int64 in a XMM 128-bit register
ML_SSE_m128_v1int64  = VirtualFormatNoForward(ML_Int64, ML_SSE_m128i, get_sse_scalar_cst, True)

# virtual boolean format
ML_SSE_m128_v4bool  = VirtualFormatNoForward(ML_Bool, ML_SSE_m128i, get_sse_scalar_cst, True)

## format for packed 4 fp32 in a XMM 128-bit register
ML_SSE_m128_v4float32 = vector_format_builder("__m128", None, 4, ML_Binary32, cst_callback = get_sse_vector_float_cst)
## format for packed 2 fp64 in a XMM 128-bit register
ML_SSE_m128_v2float64 = vector_format_builder("__m128d", None, 2, ML_Binary64)
## format for packed 4 int32 in a XMM 128-bit register
ML_SSE_m128_v4int32   = vector_format_builder("__m128i",  None, 4, ML_Int32, cst_callback = get_sse_vector_int_cst)
## format for packed 2 int64 in a XMM 128-bit register
ML_SSE_m128_v2int64   = vector_format_builder("__m128i",  None, 2, ML_Int64)
## format for packed 4 uint32 in a XMM 128-bit register
ML_SSE_m128_v4uint32  = vector_format_builder("__m128i",  None, 4, ML_UInt32, cst_callback = get_sse_vector_int_cst)
## format for packed 2 uint64 in a XMM 128-bit register
ML_SSE_m128_v2uint64  = vector_format_builder("__m128i",  None, 2, ML_UInt64)

## format for packed 8 fp32 in a YMM 256-bit register
ML_AVX_m256_v8float32 = vector_format_builder("__m256",  None, 8, ML_Binary32)
## format for packed 4 fp64 in a YMM 256-bit register
ML_AVX_m256_v4float64 = vector_format_builder("__m256d", None, 4, ML_Binary64)
## format for packed 8 int32 in a YMM 256-bit register
ML_AVX_m256_v8int32   = vector_format_builder("__m256i", None, 8, ML_Int32)
## format for packed 4 int64 in a YMM 256-bit register
ML_AVX_m256_v4int64   = vector_format_builder("__m256i", None, 4, ML_Int64)
## format for packed 8 uint32 in a YMM 256-bit register
ML_AVX_m256_v8uint32  = vector_format_builder("__m256i", None, 8, ML_UInt32)
## format for packed 4 uint64 in a YMM 256-bit register
ML_AVX_m256_v4uint64  = vector_format_builder("__m256i", None, 4, ML_UInt64)


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
_mm_max_epi32 = SmmIntrin("_mm_max_epi32", arity = 2,
                          output_precision = ML_SSE_m128_v4int32)
_mm_mul_epi32 = SmmIntrin("_mm_mul_epi32", arity = 2,
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



## check that list if made of only a single value replicated
#  in each element
def uniform_list_check(value_list):
	return reduce((lambda acc, value: acc and value == value_list[0]), value_list, True)

# check whether @p optree is a uniform vector constant
def uniform_vector_constant_check(optree):
	if isinstance(optree, Constant) and not optree.get_precision() is None and optree.get_precision().is_vector_format():
		return uniform_list_check(optree.get_value())
	else:
		return False

## If optree is vector uniform constant modify it to be a
#  conversion between a scalar constant and a vector
def vector_constant_op(optree):
	assert isinstance(optree, Constant)
	cst_value_v = optree.get_value()
	op_format = optree.get_precision()
	if uniform_list_check(cst_value_v):
		scalar_format = op_format.get_scalar_format()
		scalar_cst = Constant(cst_value_v[0], precision = scalar_format)
		## TODO: Conversion class may be changed to VectorBoardCast
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
        ) for i in xrange(4)]
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


def pred_vector_select_one_zero(optree):
    """ Predicate returns True if and only if
        optree is Select(cond, -1, 0) or Select(cond, 0, -1) 
        False otherwise """
    if not isinstance(optree, Select):
        return False
    elif not isinstance(optree.get_input(0), Comparison):
        return False
    elif not optree.get_precision().is_vector_format():
        return False
    else:
        lhs = optree.get_input(1)
        rhs = optree.get_input(2)
        cst_pred = (is_vector_cst_with_value(lhs, -1) and is_vector_cst_with_value(rhs, 0)) or \
               (is_vector_cst_with_value(lhs, 0) and is_vector_cst_with_value(rhs, -1))
        return cst_pred


sse_c_code_generation_table = {
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
            },
        },
    },
    FastReciprocal: {
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
}

sse2_c_code_generation_table = {
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v1int32,))):
                    EmmIntrin("_mm_add_epi32", arity = 2),
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
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
    BitLogicLeftShift: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_SSE_m128_v4int32,
                                  ML_SSE_m128_v4int32,
                                  ML_Int32):
                    EmmIntrin(
                        "_mm_slli_epi32", arity = 2,
                        arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}
                    )(FO_Arg(0), _mm_set1_epi64x(FO_Arg(1))),
                # TODO the last argument is a scalar here, see documentation on
                # _mm_sll_epi32. We need to make sure that the last vector is a
                # constant that can be changed into either an imm8 (above) or
                # an ML_SSE_m128_v1int32
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin(
                        "_mm_sll_epi32", arity = 2,
                        arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}
                    )(FO_Arg(0), FO_Arg(1)),
            },
        },
    },
    BitLogicRightShift: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_SSE_m128_v4int32,
                                  ML_SSE_m128_v4int32,
                                  ML_Int32):
                    EmmIntrin(
                        "_mm_srli_epi32", arity = 2,
                        arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}
                    )(FO_Arg(0), FO_Arg(1)),
                # TODO the last argument is a scalar here, see documentation on
                # _mm_srl_epi32. We need to make sure that the last vector is a
                # constant that can be changed into either an imm8 (above) or
                # an ML_SSE_m128_v1int32
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin(
                        "_mm_srl_epi32", arity = 2,
                        arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}
                    )(FO_Arg(0), FO_Arg(1)),
            },
        },
    },
    BitArithmeticRightShift: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_SSE_m128_v4int32,
                                  ML_SSE_m128_v4int32,
                                  ML_Int32):
                    EmmIntrin("_mm_srai_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0),
                                  _mm_set1_epi64x(FO_Arg(1))
                                  ),
                type_strict_match(ML_SSE_m128_v4uint32,
                                  ML_SSE_m128_v4uint32,
                                  ML_Int32):
                    EmmIntrin("_mm_srai_epi32", arity = 2,
                              arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(
                                  FO_Arg(0),
                                  _mm_set1_epi64x(FO_Arg(1))
                                  ),
                # TODO the last argument is a scalar here, see documentation on
                # _mm_srl_epi32. We need to make sure that the last vector is a
                # constant that can be changed into either an imm8 (above) or
                # an ML_SSE_m128_v1int32
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    EmmIntrin(
                        "_mm_sra_epi32", arity = 2,
                        arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}
                    )(FO_Arg(0), FO_Arg(1)),
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
                        FO_Value("_mm_set1_epi32(-1)", ML_SSE_m128_v4int32),
                        FO_Arg(0)
                    ),
            },
        },
    },
    Negation: {
        None: {
            lambda optree: True: {
                # Float negation
                type_strict_match(*(2*(ML_SSE_m128_v4float32,))):
                    EmmIntrin("_mm_xor_ps", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm_set1_ps(-0.0f)", ML_SSE_m128_v4float32)
                    ),
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
}

ssse3_c_code_generation_table = {
    Negation: {
        None: {
            lambda optree: True: {
                # Float negation is handled by SSE2 instructions
                # 32-bit integer negation using SSSE3 sign_epi32 instruction
                type_strict_match(*(2*(ML_SSE_m128_v4int32,))):
                    TmmIntrin("_mm_sign_epi32", arity = 2)(
                        FO_Value("_mm_set1_epi32(-1)", ML_SSE_m128_v4int32),
                        FO_Arg(0)
                    ),
            },
        },
    },
}

sse41_c_code_generation_table = {
    Max: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    _mm_max_epi32,
            }
        },
    },
    Multiplication: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                    _mm_mul_epi32,
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

avx_c_code_generation_table = {
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
                #type_strict_match(*((ML_SSE_m128_v4int32,) + 4*(ML_Int32,))):
                #    ImmIntrin("_mm256_set_epi32", arity = 4),
                type_strict_match(ML_AVX_m256_v8int32, v8int32):
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
                type_strict_match(ML_AVX_m256_v8int32, ML_Int32):
                    XmmIntrin("_mm256_set1_epi32", arity = 1),
                type_strict_match(ML_AVX_m256_v8float32, ML_Binary32):
                    XmmIntrin("_mm256_set1_ps", arity = 1),
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
            },
            # AVX-based conversion of 4 int64 to 4 float64, valid if inputs fit
            # into 4 int32.
            lambda optree: optree.get_input(0).get_interval() \
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
    FastReciprocal: {
        None: {
            lambda _: True: {
                type_strict_match(*(2*(ML_AVX_m256_v8float32,))):
                    _mm256_rcp_ps(FO_Arg(0)),
            },
        },
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
                    XmmIntrin("_mm256_xor_ps", arity = 2)(
                        FO_Arg(0),
                        FO_Value("_mm256_set1_ps(-0.0f)",
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
    Constant: {
        None: {
            uniform_vector_constant_check: {
                type_strict_match(ML_AVX_m256_v8int32):
                    ComplexOperator(optree_modifier = vector_constant_op),
                type_strict_match(ML_AVX_m256_v8float32):
                    ComplexOperator(optree_modifier = vector_constant_op),
                },
            },
        },
    TypeCast: {
        None: {
            lambda optree: True: {
                type_strict_match_list(
                    [ML_AVX_m256_v8float32],
                    [ML_AVX_m256_v8int32, ML_AVX_m256_v4int64]
                    ): ImmIntrin("_mm256_castsi256_ps", arity=1,
                                 output_precision=ML_AVX_m256_v8float32),
                # Signed
                type_strict_match(ML_AVX_m256_v8int32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps_si256", arity = 1,
                              output_precision = ML_AVX_m256_v8int32),
                # Unsigned
                type_strict_match(ML_AVX_m256_v8uint32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps_si256", arity = 1,
                              output_precision = ML_AVX_m256_v8uint32),
                type_strict_match_list(
                    [ML_AVX_m256_v4float64],
                    [ML_AVX_m256_v4int64, ML_AVX_m256_v8int32]
                    ): ImmIntrin("_mm256_castsi256_pd", arity=1,
                                 output_precision=ML_AVX_m256_v4float64),
                type_strict_match(ML_AVX_m256_v4int64, ML_AVX_m256_v4float64):
                    ImmIntrin("_mm256_castpd_si256", arity = 1,
                              output_precision = ML_AVX_m256_v4int64),
                type_strict_match(ML_SSE_m128_v2float64, ML_AVX_m256_v4float64):
                    ImmIntrin("_mm256_castpd256_pd128", arity = 1,
                              output_precision = ML_SSE_m128_v2float64),
                type_strict_match(ML_SSE_m128_v4float32, ML_AVX_m256_v8float32):
                    ImmIntrin("_mm256_castps256_ps128", arity = 1,
                              output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v2float64, ML_AVX_m256_v8float32):
                    ComplexOperator(optree_modifier = _mm256_castps256_pd128),
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

avx2_c_code_generation_table = {
    Addition: {
        None: {
            lambda optree: True: {
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    ImmIntrin("_mm256_add_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
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
                    ImmIntrin("_mm256_mul_epi32", arity = 2),
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    ImmIntrin("_mm256_mul_epi64", arity = 2),
            },
        },
    },
    BitArithmeticRightShift: {
        None: {
            lambda _: True: {
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
                type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                    _mm256_and_si256,
                type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                    _mm256_and_si256,
            },
        },
    },
    BitLogicLeftShift: {
        None: {
          lambda _: True: {
            # TODO implement fixed bit shift (sll, slli)
            # Variable bit shift is only available with AVX2
            # XMM version
            type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                ImmIntrin("_mm_sllv_epi32", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                ImmIntrin("_mm_sllv_epi64", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            # YMM version
            type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                ImmIntrin("_mm256_sllv_epi32", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
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
                                       ML_SSE_m128_v4int32)
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
    BitLogicRightShift: {
        None: {
          lambda optree: True: {
            # XMM version
            type_strict_match(*(3*(ML_SSE_m128_v4int32,))):
                ImmIntrin("_mm_srlv_epi32", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            type_strict_match(*(3*(ML_SSE_m128_v2int64,))):
                ImmIntrin("_mm_srlv_epi64", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            # YMM version
            type_strict_match(*(3*(ML_AVX_m256_v8int32,))):
                ImmIntrin("_mm256_srlv_epi32", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
            type_strict_match(*(3*(ML_AVX_m256_v4int64,))):
                ImmIntrin("_mm256_srlv_epi64", arity = 2,
                          arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
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
                # XMM version
                type_custom_match(FSM(ML_SSE_m128_v4float32),
                                  TCM(ML_TableFormat),
                                  FSM(ML_SSE_m128_v4int32)):
                    ImmIntrin("_mm_i32gather_ps", arity = 3,
                        output_precision = ML_SSE_m128_v4float32)(
                            FO_Arg(0),
                            FO_Arg(1),
                            FO_Value("4", ML_Int32)
                            ),
                type_custom_match(FSM(ML_SSE_m128_v2float64),
                                  TCM(ML_TableFormat),
                                  FSM(ML_SSE_m128_v2float64)):
                    ImmIntrin("_mm_i32gather_pd", arity = 3,
                        output_precision = ML_SSE_m128_v4float32)(
                            FO_Arg(0),
                            FO_Arg(1),
                            FO_Value("8", ML_Int32)
                            ),
                # YMM version with 32-bit indices
                type_custom_match(FSM(ML_AVX_m256_v8float32),
                                  TCM(ML_TableFormat),
                                  FSM(ML_AVX_m256_v8int32)):
                    ImmIntrin("_mm256_i32gather_ps", arity = 3,
                              output_precision = ML_AVX_m256_v8float32)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("4", ML_Int32)
                                  ),
                type_custom_match(FSM(ML_AVX_m256_v4float64),
                                  TCM(ML_TableFormat),
                                  FSM(ML_AVX_m256_v8int32)):
                    ImmIntrin("_mm256_i32gather_pd", arity = 3,
                              output_precision = ML_AVX_m256_v4float64)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("8", ML_Int32)
                                  ),
                # YMM version with 64-bit indices
                type_custom_match(FSM(ML_SSE_m128_v4float32),
                                  TCM(ML_TableFormat),
                                  FSM(ML_AVX_m256_v4int64)):
                    ImmIntrin("_mm256_i64gather_ps", arity = 3,
                              output_precision = ML_SSE_m128_v4float32)(
                                  FO_Arg(0),
                                  FO_Arg(1),
                                  FO_Value("4", ML_Int32)
                                  ),
                type_custom_match(FSM(ML_AVX_m256_v4float64),
                                  TCM(ML_TableFormat),
                                  FSM(ML_AVX_m256_v4int64)):
                    ImmIntrin("_mm256_i32gather_pd", arity = 3,
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
        GenericProcessor.__init__(self)

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

    def __init__(self):
        super(X86_SSE_Processor, self).__init__()

    def get_compilation_options(self):
        return super(X86_SSE_Processor, self).get_compilation_options() \
                + ['-msse']


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


class X86_SSSE3_Processor(X86_SSE2_Processor):
    target_name = "x86_ssse3"
    TargetRegister.register_new_target(target_name, lambda _: X86_SSSE3_Processor)

    code_generation_table = {
        C_Code: ssse3_c_code_generation_table,
    }

    def __init__(self):
        X86_SSE_Processor.__init__(self)

    def get_compilation_options(self):
      return super(X86_SSSE3_Processor, self).get_compilation_options() + ["-mssse3"]

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


class X86_AVX_Processor(X86_SSE41_Processor):
    target_name = "x86_avx"
    TargetRegister.register_new_target(target_name,
                                       lambda _: X86_AVX_Processor)

    code_generation_table = {
        C_Code: avx_c_code_generation_table,
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
