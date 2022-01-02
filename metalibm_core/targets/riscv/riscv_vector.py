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


import os

from metalibm_core.code_generation.complex_generator import ComplexOperator, DynamicOperator
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format, ML_TableFormat
from metalibm_core.core.target import UniqueTargetDecorator
from metalibm_core.core.ml_operations import (
    Addition, BitArithmeticRightShift, BitLogicAnd, BitLogicLeftShift, BitLogicRightShift,
    Conversion, FusedMultiplyAdd, Modulo, Multiplication, NearestInteger, Negation, Splat, Subtraction, TableLoad, TableStore,
    TypeCast)
from metalibm_core.core.ml_formats import (
    ML_Binary16, ML_Bool16, ML_Bool32, ML_Bool64, ML_FP_Format, ML_Format, ML_FormatConstructor, ML_Int16, ML_Int64, ML_Int32,
    ML_Binary64, ML_Binary32, ML_Integer, ML_UInt16, ML_UInt32, ML_UInt64, ML_Void)
from metalibm_core.core.vla_common import VLAGetLength, VLAOperation

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import (
    FSM, TCM, FO_Arg, FunctionOperator, SymbolOperator, type_custom_match, type_strict_match, type_strict_match_list)
from metalibm_core.utility.debug_utils import debug_multi, ML_Debug

from metalibm_core.utility.log_report import Log

from metalibm_core.targets.riscv.riscv import RISCV_RV64


def RVVIntrinsic(*args, **kw):
  """ Wrapper for Risc-V Vector intrinsics """
  return FunctionOperator(*args, require_header=["riscv_vector.h"], **kw)

class RVV_VectorType(ML_FormatConstructor):
    def __init__(self, lmul, eltType):
        eltTypeTag = {
            ML_UInt32:    "uint32",
            ML_UInt64:    "uint64",
            ML_Int32:    "int32",
            ML_Int64:    "int64",
            ML_Binary32: "float32",
            ML_Binary64: "float64",
        }[eltType]
        typeName = "v{}m{}_t".format(eltTypeTag, lmul)
        ML_FormatConstructor.__init__(self, None, typeName, None, lambda v: None, header="riscv_vector.h")
        self.lmul = lmul
        self.eltType = eltType

class RVV_VectorMaskType(RVV_VectorType):
    """ Constructor for vector-mask type for RISC-V Vector extension"""
    def __init__(self, lmul, eltType):
        # ratio SEW / LMUL
        n = eltType.get_bit_size() / lmul
        typeName = "vbool{}_t".format(n)
        ML_FormatConstructor.__init__(self, None, typeName, None, lambda v: None, header="riscv_vector.h")
        self.lmul = lmul
        self.eltType = eltType

# build complete map of RVV vector types
RVV_vectorTypeMap = {(lmul, eltType): RVV_VectorType(lmul, eltType) for lmul in [1, 2, 4, 8] for eltType in [ML_Binary32, ML_Binary64, ML_Int32, ML_Int64, ML_UInt32, ML_UInt64]}

RVV_vectorFloatTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_Binary32, ML_Binary64]}
RVV_vectorIntTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_Int32, ML_Int64, ML_UInt32, ML_UInt64]}
RVV_vectorSIntTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_Int32, ML_Int64]}
RVV_vectorUIntTypeMap = {(lmul, eltType): RVV_vectorTypeMap[(lmul, eltType)] for lmul in [1, 2, 4, 8] for eltType in [ML_UInt32, ML_UInt64]}
RVV_vectorBoolTypeMap = {(lmul, eltType): RVV_VectorMaskType(lmul, eltType) for lmul in [1, 2, 4, 8] for eltType in [ML_Bool16, ML_Bool32, ML_Bool64]}

# correspondence mapping for cast 
RVV_castEltTypeMapping = {
    # from signed integers
    ML_Int64: [ML_Binary64, ML_UInt64],
    ML_Int32: [ML_Binary32, ML_UInt32],
    ML_Int16: [ML_Binary16, ML_UInt16],
    # from unsigned integers
    ML_UInt64: [ML_Binary64, ML_Int64],
    ML_UInt32: [ML_Binary32, ML_Int32],
    ML_UInt16: [ML_Binary16, ML_Int16],
    # from floating-point
    ML_Binary32: [ML_UInt32, ML_Int32],
    ML_Binary64: [ML_UInt64, ML_Int64],
    ML_Binary16: [ML_UInt16, ML_Int16],
}


# specific RVV type aliases
# RVV vector types with LMUL=1 (m1)
RVV_vBinary32_m1  = RVV_vectorTypeMap[(1, ML_Binary32)]

# TODO/FIXME: should extract VLEN*LMUL/SEW values
def getElt(eltType, index, v):
    return "v{f}mv_{xf}_s_{suffix}m1_{suffix}(vslidedown_vx_{suffix}m1({v}, {v}, {index}, {index}+1))".format(
                    xf="f" if isinstance(eltType, ML_FP_Format) else "x",
                    f="f" if isinstance(eltType, ML_FP_Format) else "",
                    v=v,
                    index=index,
                    suffix=RVVIntrSuffix[eltType])
#debug_vfloat32_m1  = ML_Debug(display_format = "{%a}", pre_process = lambda v: "vfmv_f_s_f32m1_f32(%s)" % (v))
#debug_vint32_m1    = ML_Debug(display_format = "{%d}", pre_process = lambda v: "vmv_x_s_i32m1_i32(%s)" % (v))
#debug_vfloat64_m1  = ML_Debug(display_format = "{%a}", pre_process = lambda v: "vfmv_f_s_f64m1_f64(%s)" % (v))
#debug_vint64_m1    = ML_Debug(display_format = "{%d}", pre_process = lambda v: "vmv_x_s_i64m1_i64(%s)" % (v))

# number of element per vector to be displayed during debug
DEBUG_LEN = 4
def replicateFmt(fmt, n):
    """ replicate debug type format string """
    return "{" + ", ".join([fmt] * n) + "}"

def generateDbg(eltType, n):
    eltFmt = "%a" if isinstance(eltType, ML_FP_Format) else "%d"
    pre_process = lambda v: (", ".join(getElt(eltType, index, "{0}") for index in range(DEBUG_LEN))).format(v)
    return ML_Debug(display_format=replicateFmt(eltFmt, DEBUG_LEN), pre_process=pre_process)

debug_vfloat32_m1  = generateDbg(ML_Binary32, DEBUG_LEN)
debug_vint32_m1    = generateDbg(ML_Int32, DEBUG_LEN)
debug_vfloat64_m1  = generateDbg(ML_Binary64, DEBUG_LEN)
debug_vint64_m1    = generateDbg(ML_Int64, DEBUG_LEN)

debug_multi.add_mapping(RVV_vBinary32_m1, debug_vfloat32_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Int32)], debug_vint32_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Binary64)], debug_vfloat64_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Int64)], debug_vint64_m1)

RVV_VectorSize_T = ML_FormatConstructor(None, "size_t", None, lambda v: None, header="stddef.h")

RVVIntrSuffix = {
    ML_UInt32: "u32",
    ML_UInt64: "u64",
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

def getI2FResultType(eltType):
    """ return the element result type of an integer to float conversion """
    return {
        ML_Int32: ML_Binary32, 
        ML_Int64: ML_Binary64,
        ML_UInt32: ML_Binary32, 
        ML_UInt64: ML_Binary64,
    }[eltType]

def RVV_legalizeFloatNearestInteger(node):
    """ Convert a NearestInteger float -> float into a sequence
        of (Nearest Integer float -> int) followed by
        a (Conversion int -> float) """
    convIn = node.get_input(0)
    convVL = node.get_input(1)
    outType =  node.get_precision()
    lmul = outType.lmul 
    eltType = getF2IResultType(outType.eltType)
    intVectorType = RVV_vectorIntTypeMap[(lmul, eltType)]
    return VLAOperation(
        VLAOperation(convIn, convVL, specifier=NearestInteger, precision=intVectorType),
        convVL,
        specifier=Conversion,
        precision=outType
    )

def swapOperand(arity, swapMap):
    def helper(node):
        """ Assuming node is a 2-operand operation, swap left hand side
            and right hand side operands """
        preOps = [node.get_input(i) for i in range(arity)]
        # direct mapping
        indexMap = {i: i for i in range(arity)}
        # patching mapping with each swap (both ways)
        for k in swapMap:
            indexMap[k] = swapMap[k]
            indexMap[swapMap[k]] = k
        ops = tuple(preOps[indexMap[i]] for i in range(arity))
        vl = node.get_input(arity)
        return VLAOperation(*ops, vl, precision=node.get_precision(), specifier=node.specifier)
    return helper


def expandOpSplat(splatType, opIndex, vlIndex=0):
    """ generate an helper function which modify <opIndex>-th operand of its
        node argument to be changed from a scalar to a vector of type <splatType> 
        which is filled with scalar replicated. vector length is the <vlIndex>-th
        operand of the node argument """
    def helper(node):
        splatInput = node.get_input(opIndex)
        vl = node.get_input(vlIndex)
        newInput = VLAOperation(splatInput, vl, specifier=Splat, precision=splatType)
        node.set_input(opIndex, newInput)
        return node
    return helper

def typeCastInput(opIndex, vlIndex, castType):
    """ generate an helper function which modify <opIndex>-th operand of its
        node argument to be changed from a scalar to a vector of type <splatType> 
        which is filled with scalar replicated. vector length is the <vlIndex>-th
        operand of the node argument """
    def helper(node):
        opToBeModified = node.get_input(opIndex)
        vl = node.get_input(vlIndex)
        newInput = VLAOperation(opToBeModified, vl, specifier=TypeCast, precision=castType)
        node.set_input(opIndex, newInput)
        return node
    return helper

SUPPORTED_GROUP_SIZES = [1, 2, 4, 8]
SUPPORTED_ELT_SIZES = [8, 16, 32, 64]

def vlaGetLengthOpGen(node):
    """ Operator generator for VLAGetLength """
    eltSize = node.get_input(1).value
    groupSize = node.get_input(2).value
    if not groupSize in SUPPORTED_GROUP_SIZES:
        Log.report(Log.Error, "groupSize {} is not within the list of supported values {}", groupSize, SUPPORTED_GROUP_SIZES)
    if not eltSize in SUPPORTED_ELT_SIZES:
        Log.report(Log.Error, "eltSize {} is not within the list of supported values {}", eltSize, SUPPORTED_ELT_SIZES)
    return RVVIntrinsic("vsetvl_e%dm%d" % (eltSize, groupSize), arity=1, output_precision=RVV_VectorSize_T)

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
                type_strict_match_list([RVV_VectorSize_T], [RVV_VectorSize_T, ML_Int32, ML_Int64], [ML_Integer], [ML_Integer]):
                    DynamicOperator(vlaGetLengthOpGen),
            }
        }
    },
    VLAOperation: {
        Addition: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfadd_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfadd_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vadd_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all fv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType,  RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    ComplexOperator(optree_modifier=swapOperand(2, {0: 1}))
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        Modulo: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vrem_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorSIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vremu_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorUIntTypeMap
            },
        },
        Subtraction: {
            lambda optree: True: {
                # generating mapping for all vv version of vfsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfsub_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vfsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfsub_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        BitLogicAnd: {
            lambda optree: True: {
                # generating mapping for all vv version of vand
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vand_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vand
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vand_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },

        },
        Multiplication: {
             lambda optree: True: {
                 # generating mapping for all vv version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfmul_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorTypeMap
             },
             lambda optree: True: {
                # generating mapping for all vf version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfmul_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorTypeMap
             },
             lambda optree: True: {
                # generating mapping for fv (swapped vf) version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    ComplexOperator(optree_modifier=swapOperand(2, {0: 1}))
                     for (lmul, eltType) in RVV_vectorTypeMap
            }
         },
        (FusedMultiplyAdd, FusedMultiplyAdd.Standard): {
            lambda optree: True: {
                # generating mapping for all vv version of vfmadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfmadd_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfmadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfmadd_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all fv version of vfmadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    ComplexOperator(swapOperand(3, {0: 1}))
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vff version of vfmadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, eltType, RVV_VectorSize_T): 
                    ComplexOperator(optree_modifier=expandOpSplat(RVV_vectorTypeMap[(lmul, eltType)], 2, 3))
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vff version of vfmadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    ComplexOperator(optree_modifier=expandOpSplat(RVV_vectorTypeMap[(lmul, eltType)], 2, 3))
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        # (FusedMultiplyAdd, FusedMultiplyAdd.Subtract): {
        #     lambda optree: True: {
        #         # generating mapping for all vv version of vfadd
        #         type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
        #             RVVIntrinsic("vfmsub_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
        #             for (lmul, eltType) in RVV_vectorTypeMap
        #     },
        #     lambda optree: True: {
        #         # generating mapping for all vv version of vfadd
        #         type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
        #             RVVIntrinsic("vfmsub_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
        #             for (lmul, eltType) in RVV_vectorTypeMap
        #     },
        # },
        (FusedMultiplyAdd, FusedMultiplyAdd.SubtractNegate): {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfnmsub_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfnmsub_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        Negation: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfneg_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        Splat: {
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vmv_v_x_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vfmv_v_f_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        BitLogicLeftShift: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vsll_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            }
        },
        BitLogicRightShift: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vsrl_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            }
        },
        BitArithmeticRightShift: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T): 
                    RVVIntrinsic("vsra_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            }
        },
        TypeCast: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                # TODO/FIXME: TypeCast should not have length operand
                type_strict_match(RVV_vectorTypeMap[(lmul, dstEltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vreinterpret_v_%sm%d_%sm%d" % (RVVIntrSuffix[eltType], lmul, RVVIntrSuffix[dstEltType], lmul), arity=1, output_precision=RVV_vectorTypeMap[(lmul, dstEltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap for dstEltType in RVV_castEltTypeMapping[eltType]
            }
        },
        Conversion: {
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, getI2FResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfcvt_f_x_v_%sm%d" % (RVVIntrSuffix[getI2FResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getI2FResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
        },
        NearestInteger: {
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    RVVIntrinsic("vfcvt_x_f_v_%sm%d" % (RVVIntrSuffix[getF2IResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T): 
                    ComplexOperator(optree_modifier=RVV_legalizeFloatNearestInteger) 
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        TableLoad: {
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_Pointer_Format), FSM(RVV_VectorSize_T), debug=True):
                    RVVIntrinsic("vle%d_v_%sm%d" % (eltType.get_bit_size(), RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, eltType)]) for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_TableFormat), FSM(RVV_vectorTypeMap[(lmul, ML_UInt32)]), FSM(RVV_VectorSize_T), debug=True):
                    RVVIntrinsic("vluxei%d_v_%sm%d" % (eltType.get_bit_size(), RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[((lmul, eltType))]) for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_TableFormat), FSM(RVV_vectorTypeMap[(lmul, ML_Int32)]), FSM(RVV_VectorSize_T), debug=True):
                    ComplexOperator(optree_modifier=typeCastInput(opIndex=1, vlIndex=2, castType=RVV_vectorTypeMap[(lmul, ML_UInt32)] )) for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        TableStore: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Void), TCM(ML_Pointer_Format), FSM(RVV_vectorTypeMap[((lmul, eltType))]), FSM(RVV_VectorSize_T)):
                    RVVIntrinsic("vse%d_v_%sm%d" % (eltType.get_bit_size(), RVVIntrSuffix[eltType], lmul), arity=3, output_precision=ML_Void, void_function=True) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
    },
}

@UniqueTargetDecorator
class RISCV_RVV64(RISCV_RV64):
    target_name = "rv64gv"
    vectorSizeType = RVV_VectorSize_T

    code_generation_table = {
        C_Code: rvv64_CCodeGenTable,
    }

    def __init__(self):
        super().__init__()

    def get_compilation_options(self, ML_SRC_DIR):
        try:
            RISCV_ENV = os.environ["RISCV"]
        except KeyError:
            Log.report(Log.Warning, "RISCV env variable must be set such than $RISCV/riscv64-unknown-elf/lib/ is accessible")
            RISCV_ENV = "<RISCV undef>"
        return super(RISCV_RVV64, self).get_compilation_options(ML_SRC_DIR) + [f"-L{RISCV_ENV}/riscv64-unknown-elf/lib/", f"--gcc-toolchain={RISCV_ENV}/ ", "-menable-experimental-extensions", "-march=rv64gcv0p10", "-target riscv64"]


    def get_execution_command(self, test_file):
        try:
            pk_bin = os.environ["PK_BIN"]
        except KeyError:
            Log.report(Log.Warning, "PK_BIN env var must point to proxy-kernel image")
            pk_bin = "<PK_BIN undef>"

        try:
            spike_bin = os.environ["SPIKE_BIN"]
        except KeyError:
            Log.report(Log.Warning, "SPIKE_BIN env var must point to spike simulator binary")
            spike_bin = "<SPIKE_BIN undef>"
        cmd = "{} --isa=RV64gcv {}  {}".format(spike_bin, pk_bin, test_file)
        return cmd

# debug message
Log.report(LOG_BACKEND_INIT, "initializing RISC-V Vector target")