# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014)
# All rights reserved
# created:          Apr 11th,  2014
# last-modified:    Apr 11th,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from utility.log_report import *
from code_generation.generator_utility import *
from core.ml_formats import *
from core.ml_operations import *
from utility.common import Callable
from code_generation.generic_processor import GenericProcessor

ML_SSE_m128 = ML_FormatConstructor(128, "__m128", None, lambda v: None)

_mm_set_ss = FunctionOperator("_mm_set_ss", arity = 1, force_folding = True, output_precision = ML_SSE_m128, require_header = ["xmmintrin.h"])
_mm_cvt_ss2si = FunctionOperator("_mm_cvt_ss2si", arity = 1, require_header = ["xmmintrin.h"])

_mm_round_ss_rn = FunctionOperator("_mm_round_ss", arg_map = {0: FO_Arg(0), 1: FO_Arg(0), 2: "_MM_FROUND_TO_NEAREST_INT"}, arity = 1, output_precision = ML_SSE_m128, require_header = ["smmintrin.h"])
_mm_cvtss_f32 = FunctionOperator("_mm_cvtss_f32", arity = 1, output_precision = ML_Binary32, require_header = ["xmmintrin.h"])

def x86_fma_intrinsic_builder(intr_name):
    return _mm_cvtss_f32(FunctionOperator(intr_name, arity = 3, output_precision = ML_SSE_m128, require_header = ["immintrin.h"])(_mm_set_ss(FO_Arg(0)), _mm_set_ss(FO_Arg(1)), _mm_set_ss(FO_Arg(2))))

_mm_fmadd_ss = x86_fma_intrinsic_builder("_mm_fmadd_ss")

sse_c_code_generation_table = {
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Binary32, ML_Binary32): _mm_cvtss_f32(_mm_round_ss_rn(_mm_set_ss(FO_Arg(0)))), 
                type_strict_match(ML_Int32, ML_Binary32): _mm_cvt_ss2si(_mm_set_ss(FO_Arg(0))),
            },
        },
    },
}


class X86_SSE_Processor(GenericProcessor):
    code_generation_table = {
        C_Code: sse_c_code_generation_table,
    }

    def __init__(self):
        GenericProcessor.__init__(self)


class X86_FMA_Processor(X86_SSE_Processor):
    code_generation_table = {
        C_Code: {
            FusedMultiplyAdd: {
                FusedMultiplyAdd.Standard: {
                    lambda optree: True: {
                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmadd_ss"),
                    },
                },
                FusedMultiplyAdd.Subtract: {
                    lambda optree: True: {
                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fmsub"),
                    },
                },
                FusedMultiplyAdd.SubtractNegate: {
                    lambda optree: True: {
                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fnmadd_ss"),
                    },
                },
                FusedMultiplyAdd.Negate: {
                    lambda optree: True: {
                        type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32):x86_fma_intrinsic_builder(" _mm_fnmsub_ss"),
                    },
                },
            },
        },
    }

    def __init__(self):
        X86_SSE_Processor.__init__(self)
