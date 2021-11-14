# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:              Oct  3rd, 2021
# last-modified:        Oct  3rd, 2021
#
# author(s):    Nicolas Brunie (metalibm POINT dev AT gmail com)
# desciprition: Backend for RISC-V Vector Extension (V)
###############################################################################



from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.target import UniqueTargetDecorator
from metalibm_core.core.ml_operations import (
    Addition, Conversion, Multiplication, NearestInteger, TableLoad, TableStore)
from metalibm_core.core.ml_formats import (
    ML_FormatConstructor, ML_Int64, ML_Int32,
    ML_Binary64, ML_Binary32, ML_Void)
from metalibm_core.core.vla_common import VLAGetLength, VLAOperation, VLAOp

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import (
    FSM, TCM, FunctionOperator, SymbolOperator, type_custom_match, type_strict_match)

from metalibm_core.utility.log_report import Log

from metalibm_core.targets.riscv.riscv import RISCV_RV64


def RVVIntrinsic(*args, **kw):
  """ Wrapper for Risc-V Vector intrinsics """
  return FunctionOperator(*args, require_header=["riscv_vector.h"], **kw)

def buildRVVType(lmul, eltType):
    """ generic build for RVV vector types """
    eltTypeTag = {
        ML_Int32:    "int32",
        ML_Int64:    "int64",
        ML_Binary32: "float32",
        ML_Binary64: "float64",
    }[eltType]
    typeName = "v{}m{}_t".format(eltTypeTag, lmul)
    return ML_FormatConstructor(None, typeName, None, lambda v: None, header="riscv_vector.h")

# build complete map of RVV vector types
RVV_vectorTypeMap = {(lmul, eltType): buildRVVType(lmul, eltType) for lmul in [1, 2, 4, 8] for eltType in [ML_Binary32, ML_Binary64, ML_Int32, ML_Int64]}

RVV_vectorFloatTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_Binary32, ML_Binary64]}
RVV_vectorIntTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_Int32, ML_Int64]}


# specific RVV type aliases
# RVV vector types with LMUL=1 (m1)
RVV_vBinary32_m1  = RVV_vectorTypeMap[(1, ML_Binary32)]

RVV_VectorSize_T = ML_FormatConstructor(None, "size_t", None, lambda v: None, header="stddef.h")

RVVIntrSuffix = {
    ML_Int32: "i32",
    ML_Int64: "i64",
    ML_Binary32: "f32",
    ML_Binary64: "f64",
}

def getF2IResultType(eltType):
    """ return the element result type of a float to integer conversion """
    return {
        ML_Binary32: ML_Int32, 
        ML_Binary64: ML_Int64,
    }[eltType]

rvv64_CCodeGenTable = {
    Conversion: {
        None: {
            lambda node: True: {
                type_strict_match(ML_Int32, RVV_VectorSize_T):
                    SymbolOperator("(int32_t)", arity=1),
            },
        },
    },
    VLAGetLength: {
        None: {
            lambda optree: True: {
                type_strict_match(RVV_VectorSize_T, RVV_VectorSize_T):
                    RVVIntrinsic("vsetvl_e32m1", arity=1, output_precision=RVV_VectorSize_T),
                type_strict_match(RVV_VectorSize_T, ML_Int32):
                    RVVIntrinsic("vsetvl_e32m1", arity=1, output_precision=RVV_VectorSize_T),
            }
        }
    },
    VLAOperation: {
        Addition: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfadd_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfadd_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        Multiplication: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfmul_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfmul_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
        NearestInteger: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfcvt_x_f_v_%sm%d" % (RVVIntrSuffix[getF2IResultType(eltType)], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        TableLoad: {
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_Pointer_Format), FSM(RVV_VectorSize_T), debug=True):
                    RVVIntrinsic("vle32_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[((lmul, eltType))]) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
        TableStore: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Void), TCM(ML_Pointer_Format), FSM(RVV_vectorTypeMap[((lmul, eltType))]), FSM(RVV_VectorSize_T)):
                    RVVIntrinsic("vse32_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=ML_Void, void_function=True) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
    },
    VLAOp: {
        Addition: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfadd_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
        TableLoad: {
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_Pointer_Format), FSM(RVV_VectorSize_T)):
                    RVVIntrinsic("vle32_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[((lmul, eltType))]) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
        TableStore: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Void), TCM(ML_Pointer_Format), FSM(RVV_vectorTypeMap[((lmul, eltType))]), FSM(RVV_VectorSize_T)):
                    RVVIntrinsic("vse32_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=ML_Void, void_function=True) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
    },
}

@UniqueTargetDecorator
class RISCV_RVV64(RISCV_RV64):
    target_name = "rv64gv"

    code_generation_table = {
        C_Code: rvv64_CCodeGenTable,
    }

    def __init__(self):
        super().__init__()

    def get_compilation_options(self, ML_SRC_DIR):
        return super(RISCV_RVV64, self).get_compilation_options(ML_SRC_DIR) + ["-march=rv64gcv"]

# debug message
Log.report(LOG_BACKEND_INIT, "initializing RISC-V Vector target")