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

from ...utility.log_report import *
from ...code_generation.generator_utility import *
from ...core.ml_formats import *
from ...core.ml_operations import *
from ...code_generation.generic_processor import GenericProcessor
from ...core.target import TargetRegister

## format for a single fp32 stored in a XMM 128-bit register 
ML_SSE_m128_v1float32  = ML_FormatConstructor(128, "__m128",  None, lambda v: None)
## format for packed 4 fp32 in a XMM 128-bit register 
ML_SSE_m128_v4float32  = ML_FormatConstructor(128, "__m128",  None, lambda v: None)
## format for packed 1 fp64 in a XMM 128-bit register 
ML_SSE_m128_v1float64 = ML_FormatConstructor(128, "__m128d", None, lambda v: None)
## format for packed 2 fp64 in a XMM 128-bit register 
ML_SSE_m128_v2float64 = ML_FormatConstructor(128, "__m128d", None, lambda v: None)

## format for packed 8 fp32 in a YMM 256-bit register 
ML_AVX_m256  = ML_FormatConstructor(256, "__m256",  None, lambda v: None)
## format for packed 4 fp64 in a YMM 256-bit register 
ML_AVX_m256d = ML_FormatConstructor(256, "__m256d", None, lambda v: None)

# Conversion function from any float to a float packed into a __m128 register
_mm_set_ss = FunctionOperator("_mm_set_ss", arity = 1, force_folding = True, output_precision = ML_SSE_m128_v1float32, require_header = ["xmmintrin.h"])
# Conversion of a scalar float contained in a __m128 registers to a signed integer
# contained also in a __m128 register
_mm_cvt_ss2si = FunctionOperator("_mm_cvt_ss2si", arity = 1, require_header = ["xmmintrin.h"])

_mm_cvtsd2si  = FunctionOperator("_mm_cvtsd2si", arity = 1, require_header = ["emmintrin.h"])
_mm_cvtsd_si32  = FunctionOperator("_mm_cvtsd_si32", arity = 1, require_header = ["emmintrin.h"])

_mm_round_ss_rn = FunctionOperator("_mm_round_ss", arg_map = {0: FO_Arg(0), 1: FO_Arg(0), 2: "_MM_FROUND_TO_NEAREST_INT"}, arity = 1, output_precision = ML_SSE_m128_v1float32
, require_header = ["smmintrin.h"])
_mm_cvtss_f32 = FunctionOperator("_mm_cvtss_f32", arity = 1, output_precision = ML_Binary32, require_header = ["xmmintrin.h"])

_mm_set_sd = FunctionOperator("_mm_set_sd", arity = 1, force_folding = True,
                              output_precision = ML_SSE_m128_v1float64,
                              require_header = ["xmmintrin.h"])
_mm_round_sd_rn = FunctionOperator("_mm_round_sd", arg_map = {0: FO_Arg(0), 1: FO_Arg(0), 2: "_MM_FROUND_TO_NEAREST_INT"}, arity = 1, output_precision = ML_SSE_m128_v1float64, require_header = ["smmintrin.h"])
_mm_cvtsd_f64 = FunctionOperator("_mm_cvtsd_f64", arity = 1, output_precision = ML_Binary64, require_header = ["xmmintrin.h"])


# 3-to-5-cycle latency / 1-to-2-cycle throughput approximate reciprocal, with a
# maximum relative error of 1.5 * 2^(-12).
_mm_rcp_ss = FunctionOperator("_mm_rcp_ss", arity = 1,
                              output_precision = ML_SSE_m128_v1float32,
                              require_header = ["xmmintrin.h"])
_mm_rcp_ps = FunctionOperator("_mm_rcp_ps", arity = 1,
                              output_precision = ML_SSE_m128_v4float32,
                              require_header = ["xmmintrin.h"])
_mm256_rcp_ps = FunctionOperator("_mm256_rcp_ps", arity = 1,
                                 output_precision = ML_AVX_m256,
                                 require_header = ["immintrin.h"])

_mm_add_ss = FunctionOperator("_mm_add_ss", arity = 2,
                              output_precision = ML_SSE_m128_v1float32,
                              require_header = ["xmmintrin.h"])
_mm_mul_ss = FunctionOperator("_mm_mul_ss", arity = 2,
                              output_precision = ML_SSE_m128_v4float32,
                              require_header = ["xmmintrin.h"])
_lzcnt_u32 = FunctionOperator("_lzcnt_u32", arity = 1,
        output_precision = ML_UInt32,
        require_header = ["immintrin.h"])
_lzcnt_u64 = FunctionOperator("_lzcnt_u64", arity = 1,
        output_precision = ML_UInt64,
        require_header = ["immintrin.h"])

_mm_cvtss_f32 = FunctionOperator("_mm_cvtss_f32", arity = 1,
                                 output_precision = ML_Binary32,
                                 require_header = ["xmmintrin.h"])

def x86_fma_intrinsic_builder(intr_name):
    return _mm_cvtss_f32(
            FunctionOperator(intr_name, arity = 3,
                             output_precision = ML_SSE_m128_v1float32,
                             require_header = ["immintrin.h"]
                             )(_mm_set_ss(FO_Arg(0)),
                               _mm_set_ss(FO_Arg(1)),
                               _mm_set_ss(FO_Arg(2))))
def x86_fmad_intrinsic_builder(intr_name):
    return _mm_cvtsd_f64(FunctionOperator(intr_name, arity = 3, output_precision = ML_SSE_m128_v1float64, require_header = ["immintrin.h"])(_mm_set_sd(FO_Arg(0)), _mm_set_sd(FO_Arg(1)), _mm_set_sd(FO_Arg(2))))

## Builder for x86 FMA intrinsic within XMM register 
# (native, no conversions)
#  
def x86_fma_intr_builder_native(intr_name):
    return FunctionOperator(intr_name, arity = 3,
                             output_precision = ML_SSE_m128_v1float32,
                             require_header = ["immintrin.h"]
                             )
def x86_fmad_intr_builder_native(intr_name):
    return FunctionOperator(intr_name, arity = 3, output_precision = ML_SSE_m128_v1float64, require_header = ["immintrin.h"])



_mm_fmadd_ss = x86_fma_intrinsic_builder("_mm_fmadd_ss")

sse_c_code_generation_table = {
    # Arithmetic
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_add_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_mul_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
            },
        },
    },
    FastReciprocal: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_rcp_ss(FO_Arg(0)),
                type_strict_match(ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(FO_Arg(0)))),
            },
        },
    },
    NearestInteger: {
        None: {
            lambda optree: True: {
                # type_strict_match(ML_Binary32, ML_Binary32): _mm_cvtss_f32(_mm_set_ss(FO_Arg(0))),
                type_strict_match(ML_Int32, ML_Binary32):    _mm_cvt_ss2si(_mm_set_ss(FO_Arg(0))),
            },
        },
    },
}

sse2_c_code_generation_table = {
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int64, ML_Binary64):    _mm_cvtsd2si(_mm_set_sd(FO_Arg(0))),
                type_strict_match(ML_Int32, ML_Binary64):    _mm_cvtsd_si32(_mm_set_sd(FO_Arg(0))),
            },
        },
    },
}

sse41_c_code_generation_table = {
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): _mm_round_ss_rn,
                type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): _mm_round_sd_rn,

                type_strict_match(ML_Binary32, ML_Binary32): _mm_cvtss_f32(_mm_round_ss_rn(_mm_set_ss(FO_Arg(0)))),
                type_strict_match(ML_Binary64, ML_Binary64): _mm_cvtsd_f64(_mm_round_sd_rn(_mm_set_sd(FO_Arg(0)))),
            },
        },
    },
}


class X86_SSE_Processor(GenericProcessor):
    target_name = "x86_sse"
    TargetRegister.register_new_target(target_name, lambda _: X86_SSE_Processor)

    code_generation_table = {
        C_Code: sse_c_code_generation_table,
    }

    def __init__(self):
        GenericProcessor.__init__(self)

class X86_SSE2_Processor(X86_SSE_Processor):
    target_name = "x86_sse2"
    TargetRegister.register_new_target(target_name, lambda _: X86_SSE2_Processor)

    code_generation_table = {
        C_Code: sse2_c_code_generation_table,
    }

    def __init__(self):
        X86_SSE_Processor.__init__(self)

class X86_SSE41_Processor(X86_SSE2_Processor):
    target_name = "x86_sse41"
    TargetRegister.register_new_target(target_name, lambda _: X86_SSE41_Processor)

    code_generation_table = {
        C_Code: sse41_c_code_generation_table,
    }

    def __init__(self):
        X86_SSE2_Processor.__init__(self)

class X86_AVX2_Processor(X86_SSE41_Processor):
    target_name = "x86_avx2"
    TargetRegister.register_new_target(target_name, lambda _: X86_AVX2_Processor)

    code_generation_table = {
        C_Code: {
            FusedMultiplyAdd: {
                FusedMultiplyAdd.Standard: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fmadd_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fmadd_sd"),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmadd_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fmadd_sd"),
                    },
                },
                FusedMultiplyAdd.Subtract: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fmsub_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fmsub_sd"),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmsub_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fmsub_sd"),
                    },
                },
                FusedMultiplyAdd.SubtractNegate: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fnmadd_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fnmadd_sd"),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fnmadd_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fnmadd_sd"),
                    },
                },
                FusedMultiplyAdd.Negate: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fnmsub_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fnmsub_sd"),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fnmsub_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fnmsub_sd"),
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
        },
    }

    def __init__(self):
        X86_SSE41_Processor.__init__(self)


# debug message
print "initializing INTEL targets"
