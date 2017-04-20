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

from metalibm_core.utility.log_report import *
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.complex_generator import ComplexOperator
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_operations import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_table import ML_TableFormat

from metalibm_core.targets.common.vector_backend import VectorBackend

## format for a single fp32 stored in a XMM 128-bit register 
ML_SSE_m128_v1float32  = ML_FormatConstructor(128, "__m128",  None, lambda v: None)
## format for packed 4 fp32 in a XMM 128-bit register 
ML_SSE_m128_v4float32  = ML_FormatConstructor(128, "__m128",  None, lambda v: None)
## format for packed 1 fp64 in a XMM 128-bit register 
ML_SSE_m128_v1float64 = ML_FormatConstructor(128, "__m128d", None, lambda v: None)
## format for packed 2 fp64 in a XMM 128-bit register 
ML_SSE_m128_v2float64 = ML_FormatConstructor(128, "__m128d", None, lambda v: None)

## format for a single int32 stored in a XMM 128-bit register 
ML_SSE_m128_v1int32  = ML_FormatConstructor(128, "__m128i",  None, lambda v: None)
## format for packed 4 int32 in a XMM 128-bit register 
ML_SSE_m128_v4int32  = ML_FormatConstructor(128, "__m128i",  None, lambda v: None)

## format for packed 8 fp32 in a YMM 256-bit register 
ML_AVX_m256  = ML_FormatConstructor(256, "__m256",  None, lambda v: None)
## format for packed 4 fp64 in a YMM 256-bit register 
ML_AVX_m256d = ML_FormatConstructor(256, "__m256d", None, lambda v: None)

# Conversion function from any float to a float packed into a __m128 register
_mm_set_ss = FunctionOperator("_mm_set_ss", arity = 1, force_folding = True, output_precision = ML_SSE_m128_v1float32, require_header = ["xmmintrin.h"])

_mm_set1_epi32 = FunctionOperator("_mm_set1_epi32", arity = 1, force_folding = True, output_precision = ML_SSE_m128_v1int32, require_header = ["xmmintrin.h"])

_mm_set1_epi64x = FunctionOperator("_mm_set1_epi64x", arity = 1, force_folding = True, output_precision = ML_SSE_m128_v4int32, require_header = ["emmintrin.h"])

# Conversion of a scalar float contained in a __m128 registers to a signed integer
# contained also in a __m128 register
_mm_cvt_ss2si = FunctionOperator("_mm_cvt_ss2si", arity = 1, require_header = ["xmmintrin.h"])

_mm_cvtsd_si64  = FunctionOperator("_mm_cvtsd_si64", arity = 1, require_header = ["emmintrin.h"])
_mm_cvtsd_si32  = FunctionOperator("_mm_cvtsd_si32", arity = 1, require_header = ["emmintrin.h"])

_mm_round_ss_rn = FunctionOperator("_mm_round_ss", arg_map = {0: FO_Arg(0), 1: FO_Arg(0), 2: "_MM_FROUND_TO_NEAREST_INT"}, arity = 1, output_precision = ML_SSE_m128_v1float32
, require_header = ["smmintrin.h"])
_mm_cvtss_f32 = FunctionOperator("_mm_cvtss_f32", arity = 1, output_precision = ML_Binary32, require_header = ["xmmintrin.h"])

_mm_set_sd = FunctionOperator("_mm_set_sd", arity = 1, force_folding = True,
                              output_precision = ML_SSE_m128_v1float64,
                              require_header = ["xmmintrin.h"])
_mm_round_sd_rn = FunctionOperator("_mm_round_sd", arg_map = {0: FO_Arg(0), 1: FO_Arg(0), 2: "_MM_FROUND_TO_NEAREST_INT"}, arity = 1, output_precision = ML_SSE_m128_v1float64, require_header = ["smmintrin.h"])
_mm_cvtsd_f64 = FunctionOperator("_mm_cvtsd_f64", arity = 1, output_precision = ML_Binary64, require_header = ["xmmintrin.h"])


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
def x86_fma_intr_builder_native(intr_name, output_precision = ML_SSE_m128_v1float32):
    return FunctionOperator(intr_name, arity = 3,
                             output_precision = output_precision,
                             require_header = ["immintrin.h"]
                             )
def x86_fmad_intr_builder_native(intr_name):
    return FunctionOperator(intr_name, arity = 3, output_precision = ML_SSE_m128_v1float64, require_header = ["immintrin.h"])

## Convert a v4 to m128 conversion optree
def v4_to_m128_modifier(optree):
  conv_input = optree.get_input(0)
  elt_precision = conv_input.get_precision().get_scalar_format()
  
  elts = [VectorElementSelection(
    conv_input, 
    Constant(i, precision = ML_Integer), 
    precision = elt_precision
  ) for i in xrange(4)]
  return Conversion(elts[0], elts[1], elts[2], elts[3], precision = optree.get_precision())


_mm_fmadd_ss = x86_fma_intrinsic_builder("_mm_fmadd_ss")

sse_c_code_generation_table = {
    Conversion: {
      None: {
        lambda _: True: {
          type_strict_match(ML_SSE_m128_v1int32, ML_Int32): _mm_set1_epi32,

          type_strict_match(ML_SSE_m128_v1float32, ML_Binary32): _mm_set_ss,
          type_strict_match(ML_Binary32, ML_SSE_m128_v1float32): _mm_cvtss_f32,

          type_strict_match(ML_SSE_m128_v1float64, ML_Binary64): _mm_set_sd,
          type_strict_match(ML_Binary64, ML_SSE_m128_v1float64): _mm_cvtsd_f64,

          type_strict_match(ML_SSE_m128_v4float32, v4float32):
            XmmIntrin("_mm_load_ps", arity = 1, output_precision = ML_SSE_m128_v4float32)
              #(TemplateOperatorFormat("(__m128*){}", arity = 1, output_precision = ML_Pointer_Format(ML_SSE_m128_v4float32))
                (TemplateOperatorFormat("GET_VEC_FIELD_ADDR({})", arity = 1, output_precision = ML_Pointer_Format(ML_Binary32))),#),
          # m128 float vector to ML's generic vector format
          type_strict_match(v4float32, ML_SSE_m128_v4float32): 
            TemplateOperatorFormat("_mm_store_ps(GET_VEC_FIELD_ADDR({}), {})", 
              arity = 1, 
              arg_map = {0: FO_Result(0), 1: FO_Arg(0)}, 
              require_header = ["xmmintrin.h"]
            ),
            #XmmIntrin("_mm_store_ps", arity = 2, arg_map = {0: FO_Result(0), 1: FO_Arg(0)})
            #  (FunctionOperator("GET_VEC_FIELD_ADDR", arity = 1, output_precision = ML_Pointer_Format(ML_Binary32))(FO_Result(0)), FO_Arg(0)),

          type_strict_match(v4int32, ML_SSE_m128_v4int32): XmmIntrin("_mm_store_si128", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1),

          type_strict_match(ML_SSE_m128_v4int32, ML_Int32, ML_Int32, ML_Int32, ML_Int32): XmmIntrin("_mm_set_epi32", arity = 4),
          #type_strict_match(ML_SSE_m128_v4int32, v4int32): ComplexOperator(optree_modifier = v4_to_m128_modifier),
          type_strict_match(ML_SSE_m128_v4int32, v4int32): 
            XmmIntrin("_mm_load_si128", arity = 1, output_precision = ML_SSE_m128_v4int32)
              (TemplateOperatorFormat("(__m128i*){}", arity = 1, output_precision = ML_Pointer_Format(ML_SSE_m128_v4int32))
                (TemplateOperatorFormat("GET_VEC_FIELD_ADDR({})", arity = 1, output_precision = ML_Pointer_Format(ML_Int32)))),
        }
      },
    },
    BitLogicAnd: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): XmmIntrin("_mm_and_ps", arity = 2, output_precision = ML_SSE_m128_v4float32), 
        },
      },
    },
    BitLogicOr: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): XmmIntrin("_mm_or_ps", arity = 2, output_precision = ML_SSE_m128_v4float32), 
        },
      },
    },
    # Arithmetic
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_add_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
                # vector addition
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): XmmIntrin("_mm_add_ps", arity = 2),
            },
        },
    },
    Subtraction: {
        None: {
            lambda _: True: {
                # vector addition
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): XmmIntrin("_mm_sub_ps", arity = 2),
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): XmmIntrin("_mm_sub_ss", arity = 2),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    XmmIntrin("_mm_mul_ps", arity = 2, output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32):
                    _mm_mul_ss(FO_Arg(0), FO_Arg(1)),
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    _mm_cvtss_f32(_mm_mul_ss(_mm_set_ss(FO_Arg(0)),
                                             _mm_set_ss(FO_Arg(1)))),
                # vector multiplication
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): XmmIntrin("_mm_mul_ps", arity = 2),
            },
        },
    },
    FastReciprocal: {
        None: {
            lambda _: True: {
                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32):
                    XmmIntrin("_mm_rcp_ps", arity = 1),
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
                type_strict_match(ML_Int64, ML_Binary64):    _mm_cvtsd_si64(_mm_set_sd(FO_Arg(0))),
                type_strict_match(ML_Int32, ML_Binary64):    _mm_cvtsd_si32(_mm_set_sd(FO_Arg(0))),
            },
        },
    },
    TypeCast: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4int32): EmmIntrin("_mm_castsi128_ps", arity = 1, output_precision = ML_SSE_m128_v4float32), 
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4float32): EmmIntrin("_mm_castps_si128", arity = 1, output_precision = ML_SSE_m128_v4int32), 
        },
      },
    },
    Addition: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32): EmmIntrin("_mm_add_epi32", arity = 2),
        },
      },
    },
    Subtraction: { 
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32): EmmIntrin("_mm_sub_epi32", arity = 2),
        },
      },
    },
    BitLogicAnd: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32): EmmIntrin("_mm_and_si128", arity = 2),
        },
      },
    },
    BitLogicOr: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32): EmmIntrin("_mm_or_si128", arity = 2),
        },
      },
    },
    BitLogicLeftShift: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_Int32):
            EmmIntrin("_mm_sll_epi32", arity = 2, arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(FO_Arg(0), _mm_set1_epi64x(FO_Arg(1))),
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
            ImmIntrin("_mm_sllv_epi32", arity = 2, arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
        },
      },
    },
    BitLogicRightShift: {
      None: {
        lambda optree: True: {
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_Int32):
            EmmIntrin("_mm_sra_epi32", arity = 2, arg_map = {0: FO_Arg(0), 1: FO_Arg(1)})(FO_Arg(0), _mm_set1_epi64x(FO_Arg(1))),
          type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4int32, ML_SSE_m128_v4int32):
            ImmIntrin("_mm_srav_epi32", arity = 2, arg_map = {0: FO_Arg(0), 1: FO_Arg(1)}),
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

                type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): 
                  SmmIntrin("_mm_round_ps", arity = 1, arg_map = {0: FO_Arg(0), 1: "_MM_FROUND_TO_NEAREST_INT"}, output_precision = ML_SSE_m128_v4float32),
                type_strict_match(ML_SSE_m128_v4int32, ML_SSE_m128_v4float32): 
                  EmmIntrin("_mm_cvtps_epi32", arity = 1, output_precision = ML_SSE_m128_v4int32)
                    (SmmIntrin("_mm_round_ps", arity = 1, arg_map = {0: FO_Arg(0), 1: "_MM_FROUND_TO_NEAREST_INT"}, output_precision = ML_SSE_m128_v4float32)),
            },
        },
    },
}


class X86_SSE_Processor(VectorBackend):
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
            TableLoad: {
              None: {
                lambda optree: True: {
                  type_custom_match(FSM(ML_SSE_m128_v4float32), TCM(ML_TableFormat), FSM(ML_SSE_m128_v4int32)): ImmIntrin("_mm_i32gather_ps", arity = 3, output_precision = ML_SSE_m128_v4float32)(FO_Arg(0), FO_Arg(1), FO_Value("4", ML_Int32)),
                },
              },
            },
            FusedMultiplyAdd: {
                FusedMultiplyAdd.Standard: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native("_mm_fmadd_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native("_mm_fmadd_sd"),
                        # vectorial fma
                        type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): x86_fma_intr_builder_native("_mm_fmadd_ps", output_precision = ML_SSE_m128_v4float32),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmadd_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fmadd_sd"),
                    },
                },
                FusedMultiplyAdd.Subtract: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fmsub_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fmsub_sd"),
                        # vectorial fma
                        type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): x86_fma_intr_builder_native("_mm_fmsub_ps", output_precision = ML_SSE_m128_v4float32),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmsub_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fmsub_sd"),
                    },
                },
                FusedMultiplyAdd.SubtractNegate: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fnmadd_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fnmadd_sd"),

                        # vectorial fma
                        type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): x86_fma_intr_builder_native("_mm_fnmadd_ps", output_precision = ML_SSE_m128_v4float32),

                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fnmadd_ss"),
                        type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):x86_fmad_intrinsic_builder(" _mm_fnmadd_sd"),
                    },
                },
                FusedMultiplyAdd.Negate: {
                    lambda optree: True: {
                        type_strict_match(ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32, ML_SSE_m128_v1float32): x86_fma_intr_builder_native(" _mm_fnmsub_ss"),
                        type_strict_match(ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64, ML_SSE_m128_v1float64): x86_fmad_intr_builder_native(" _mm_fnmsub_sd"),
                        # vectorial fma
                        type_strict_match(ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32, ML_SSE_m128_v4float32): x86_fma_intr_builder_native("_mm_fnmsub_ps", output_precision = ML_SSE_m128_v4float32),

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

    def get_compilation_options(self):
      return super(X86_AVX2_Processor, self).get_compilation_options() + ["-mfma", "-mavx2"]


# debug message
print "initializing INTEL targets"
