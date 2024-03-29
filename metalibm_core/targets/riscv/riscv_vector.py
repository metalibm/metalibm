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
    Abs, Addition, BitArithmeticRightShift, BitLogicAnd, BitLogicLeftShift,
    BitLogicOr, BitLogicRightShift, Comparison, Conversion, Floor, FusedMultiplyAdd,
    Max, Min, Modulo, Multiplication, NearestInteger, Negation, ReciprocalSeed,
    Select, Splat, Subtraction, TableLoad, TableStore, Trunc, TypeCast)
from metalibm_core.core.ml_formats import (
    SUPPORT_FORMAT_MAP,
    ML_Binary16,
    ML_Bool8, ML_Bool16, ML_Bool32, ML_Bool64, ML_FP_Format, ML_FormatConstructor,
    ML_Int16, ML_Int64, ML_Int32,
    ML_Binary64, ML_Binary32, ML_Integer, ML_UInt16, ML_UInt32, ML_UInt64, ML_Void)
from metalibm_core.core.vla_common import VLAGetLength, VLAOperation

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import (
    FSM, TCM, FO_Arg, FunctionOperator, SymbolOperator, type_custom_match,
    type_strict_match, type_strict_match_list)
from metalibm_core.utility.debug_utils import debug_multi, ML_Debug

from metalibm_core.utility.log_report import Log

from metalibm_core.targets.riscv.riscv import RISCV_RV64_CLANG
from metalibm_core.targets.riscv.rvv_table import rvv_approx_table_map


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
        n = eltType.get_bit_size() // lmul
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

# associating previously built mask types to each pair (lmul, eltType)
RVV_vectorBoolTypeMap = {(lmul, eltType.get_bit_size()): RVV_VectorMaskType(lmul, eltType) for lmul in [1, 2, 4, 8] for eltType in [ML_Bool8, ML_Bool16, ML_Bool32, ML_Bool64]}


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
debug_vint32_m1    = generateDbg(ML_Int32,    DEBUG_LEN)
debug_vuint32_m1   = generateDbg(ML_UInt32,   DEBUG_LEN)
debug_vfloat64_m1  = generateDbg(ML_Binary64, DEBUG_LEN)
debug_vint64_m1    = generateDbg(ML_Int64,    DEBUG_LEN)
debug_vuint64_m1   = generateDbg(ML_UInt64,   DEBUG_LEN)

debug_pp = lambda v, index, isize: f"""vmv_x_s_i{isize}m1_i{isize}(vslidedown_vx_i{isize}m1(vmv_v_x_i{isize}m1(0, {DEBUG_LEN}), vmerge_vxm_i{isize}m1({v}, vmv_v_x_i{isize}m1(0, {DEBUG_LEN}), 1, {DEBUG_LEN}), {index}, {index+1}))"""

debug_vbool_i32 = ML_Debug(display_format=replicateFmt("%d", DEBUG_LEN), pre_process=lambda v: ", ".join(debug_pp(v, i, 32) for i in range(DEBUG_LEN)))
debug_vbool_i64 = ML_Debug(display_format=replicateFmt("%ld", DEBUG_LEN), pre_process=lambda v: ", ".join(debug_pp(v, i, 64) for i in range(DEBUG_LEN)))

debug_multi.add_mapping(RVV_vBinary32_m1, debug_vfloat32_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Int32)], debug_vint32_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_UInt32)], debug_vuint32_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Binary64)], debug_vfloat64_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_Int64)], debug_vint64_m1)
debug_multi.add_mapping(RVV_vectorTypeMap[(1, ML_UInt64)], debug_vuint64_m1)
debug_multi.add_mapping(RVV_vectorBoolTypeMap[(1, 32)], debug_vbool_i32)
debug_multi.add_mapping(RVV_vectorBoolTypeMap[(1, 64)], debug_vbool_i64)

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

def RVV_legalizeIntConversion(convSpecifier):
    """ Convert a Conversion to integer with float -> float prototype
        into a sequence of (specified Conversion float -> int) followed by
        a (Conversion int -> float) """
    def helper(node):
        convIn = node.get_input(0)
        convVL = node.get_input(1)
        outType =  node.get_precision()
        lmul = outType.lmul
        eltType = getF2IResultType(outType.eltType)
        intVectorType = RVV_vectorIntTypeMap[(lmul, eltType)]
        return VLAOperation(
            VLAOperation(convIn, convVL, specifier=convSpecifier, precision=intVectorType),
            convVL,
            specifier=Conversion,
            precision=outType
        )
    return helper

RVV_legalizeFloatNearestInteger = RVV_legalizeIntConversion(NearestInteger)

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
        nodeCopy = node.copy(copy_map={opToBeModified: newInput, vl: vl})
        return nodeCopy
    return helper


def typeCastOutput(castType):
    """ generate an helper function to modify a node output's type to <castType>
        and generate a wrapper around the modified node to cast the result back to its
        original result tye """
    def helper(node):
        outType = node.get_precision()
        node.set_precision(castType)
        return TypeCast(node, precision=outType)
    return helper


def composeModifiers(*funcs):
    """ compose a list of modifier functions """
    def helper(node):
        r = node
        for f in funcs[::-1]:
            r = f(r)
        return r
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
                # generating mapping for all vx version of vadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vadd_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vadd
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vadd_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
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
                    RVVIntrinsic("v%ssub_vv_%sm%d" % ("f" if isinstance(eltType, ML_FP_Format) else "",RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("v%ssub_v%s_%sm%d" % ("f" if isinstance(eltType, ML_FP_Format) else "", "f" if isinstance(eltType, ML_FP_Format) else "x", RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=expandOpSplat(RVV_vectorTypeMap[(lmul, eltType)], 0, 2))
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
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
        BitLogicOr: {
            lambda optree: True: {
                # generating mapping for all vv version of vor
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vor_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vor
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vor_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
        },
        Multiplication: {
             lambda optree: True: {
                 # generating mapping for all vv version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfmul_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for all vf version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vfmul_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for fv (swapped vf) version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=swapOperand(2, {0: 1}))
                     for (lmul, eltType) in RVV_vectorTypeMap
            },
             lambda optree: True: {
                # generating mapping for all vx version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmul_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorIntTypeMap
             },
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
        (FusedMultiplyAdd, FusedMultiplyAdd.Subtract): {
           lambda optree: True: {
               # generating mapping for all vv version of vfmsub
               type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                   RVVIntrinsic("vfmsub_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                   for (lmul, eltType) in RVV_vectorTypeMap
           },
           lambda optree: True: {
               # generating mapping for all vv version of vfmsub
               type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                   RVVIntrinsic("vfmsub_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                   for (lmul, eltType) in RVV_vectorTypeMap
           },
        },
        (FusedMultiplyAdd, FusedMultiplyAdd.SubtractNegate): {
            lambda optree: True: {
                # generating mapping for all vv version of vfnmsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfnmsub_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vf version of vfnmsub
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfnmsub_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        Negation: {
            lambda optree: True: {
                # generating mapping for all vv version of vneg
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("v%sneg_v_%sm%d" % ("f" if isinstance(eltType, ML_FP_Format) else "", RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vneg with unsigned input and signed output
                # FIXME: typecast of input may discard MSB value
                type_strict_match(RVV_vectorTypeMap[(lmul, SUPPORT_FORMAT_MAP[True][eltType.get_bit_size()])], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=typeCastInput(opIndex=0, vlIndex=1, castType=RVV_vectorSIntTypeMap[(lmul, SUPPORT_FORMAT_MAP[True][eltType.get_bit_size()])]))
                    for (lmul, eltType) in RVV_vectorUIntTypeMap
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
                # generating mapping for all vx version of vsrl
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vsrl_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorUIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vsrl
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], SUPPORT_FORMAT_MAP[True][eltType.get_bit_size()], RVV_VectorSize_T):
                    RVVIntrinsic("vsrl_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorUIntTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vx version of vsrl on signed types
                # signed input is casted to unsigned vector type before operation
                type_strict_match(RVV_vectorIntTypeMap[(lmul, eltType)], RVV_vectorIntTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=composeModifiers(typeCastOutput(RVV_vectorUIntTypeMap[(lmul, SUPPORT_FORMAT_MAP[False][eltType.get_bit_size()])]),
                                                                     typeCastInput(opIndex=0, vlIndex=2, castType=RVV_vectorUIntTypeMap[(lmul, SUPPORT_FORMAT_MAP[False][eltType.get_bit_size()])])))
                    for (lmul, eltType) in RVV_vectorSIntTypeMap
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
            },
        },
        Conversion: {
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, getI2FResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfcvt_f_x_v_%sm%d" % (RVVIntrSuffix[getI2FResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getI2FResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfcvt_x_f_v_%sm%d" % (RVVIntrSuffix[getF2IResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
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
        Floor: {
            # Floor is implemented by first converting to integer while rounding down before converting back to floating-point
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=RVV_legalizeIntConversion(Floor))
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                # TODO/FIXME: the mapping is wrong as it maps a floor (round-down) to a trunc (rtz)
                #              it only works for positive inputs
                type_strict_match(RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfcvt_rtz_x_f_v_%sm%d" % (RVVIntrSuffix[getF2IResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        Trunc: {
            # Floor is implemented by first converting to integer while rounding down before converting back to floating-point
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=RVV_legalizeIntConversion(Trunc))
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfcvt_rtz_x_f_v_%sm%d" % (RVVIntrSuffix[getF2IResultType(eltType)], lmul), arity=2, output_precision=RVV_vectorTypeMap[(lmul, getF2IResultType(eltType))])
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
                    RVVIntrinsic("vluxei32_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[((lmul, eltType))]) for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_TableFormat), FSM(RVV_vectorTypeMap[(lmul, ML_UInt64)]), FSM(RVV_VectorSize_T), debug=True):
                    RVVIntrinsic("vluxei64_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[((lmul, eltType))]) for (lmul, eltType) in RVV_vectorTypeMap
            },
            # mapping to cast int32 indices to uint32
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_TableFormat), FSM(RVV_vectorTypeMap[(lmul, ML_Int32)]), FSM(RVV_VectorSize_T), debug=True):
                    ComplexOperator(optree_modifier=typeCastInput(opIndex=1, vlIndex=2, castType=RVV_vectorTypeMap[(lmul, ML_UInt32)] )) for (lmul, eltType) in RVV_vectorTypeMap
            },
            # mapping to cast int64 indices to uint64
            lambda optree: True: {
                type_custom_match(FSM(RVV_vectorTypeMap[((lmul, eltType))]), TCM(ML_TableFormat), FSM(RVV_vectorTypeMap[(lmul, ML_Int64)]), FSM(RVV_VectorSize_T), debug=True):
                    ComplexOperator(optree_modifier=typeCastInput(opIndex=1, vlIndex=2, castType=RVV_vectorTypeMap[(lmul, ML_UInt64)] )) for (lmul, eltType) in RVV_vectorTypeMap
            },
        },
        TableStore: {
            lambda optree: True: {
                type_custom_match(FSM(ML_Void), TCM(ML_Pointer_Format), FSM(RVV_vectorTypeMap[((lmul, eltType))]), FSM(RVV_VectorSize_T)):
                    RVVIntrinsic("vse%d_v_%sm%d" % (eltType.get_bit_size(), RVVIntrSuffix[eltType], lmul), arity=3, output_precision=ML_Void, void_function=True) for (lmul, eltType) in RVV_vectorTypeMap
            }
        },
        Select: {
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vmerge_vvm_%sm%d" % (RVVIntrSuffix[eltType], lmul), arg_map={0: FO_Arg(0), 1: FO_Arg(2), 2: FO_Arg(1), 3: FO_Arg(3)}, arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)]) # op[1] and op[2] may need to be inverted
                    for (lmul, eltType) in RVV_vectorTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=expandOpSplat(RVV_vectorTypeMap[(lmul, eltType)], 2, 3)) # TODO/optme: could be implemented by swaping vec ops and negating mask
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfmerge_vfm_%sm%d" % (RVVIntrSuffix[eltType], lmul), arg_map={0: FO_Arg(0), 1: FO_Arg(2), 2: FO_Arg(1), 3: FO_Arg(3)}, arity=4, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], eltType, eltType, RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=expandOpSplat(RVV_vectorTypeMap[(lmul, eltType)], 1, 3))
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        (Comparison, Comparison.Greater): {
            lambda optree: True: {
                # generating mapping for all vf version of gt comparison
                type_strict_match(RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorFloatTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmfgt_vf_%sm%d_b%d" % (RVVIntrSuffix[eltType], lmul, eltType.get_bit_size() / lmul), arity=3, output_precision=RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vmfgt_vv_%sm%d_b%d" % (RVVIntrSuffix[eltType], lmul, eltType.get_bit_size() / lmul), arity=3, output_precision=RVV_vectorFloatTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        (Comparison, Comparison.Less): {
            lambda optree: True: {
                # generating mapping for all vf version of gt comparison
                type_strict_match(RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorFloatTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmflt_vf_%sm%d_b%d" % (RVVIntrSuffix[eltType], lmul, eltType.get_bit_size() / lmul), arity=3, output_precision=RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
            lambda optree: True: {
                # generating mapping for all vv version of vfadd
                type_strict_match(RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vmflt_vv_%sm%d_b%d" % (RVVIntrSuffix[eltType], lmul, eltType.get_bit_size() / lmul), arity=3, output_precision=RVV_vectorFloatTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            },
        },
        (Comparison, Comparison.Equal): {
            lambda optree: True: {
                # generating mapping for all vf version of gt comparison
                type_strict_match(RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmseq_vx_%sm%d_b%d" % (RVVIntrSuffix[eltType], lmul, eltType.get_bit_size() / lmul), arity=3, output_precision=RVV_vectorBoolTypeMap[(lmul, eltType.get_bit_size())])
                    for (lmul, eltType) in RVV_vectorIntTypeMap
            },
        },
        ReciprocalSeed: {
            lambda optree: True: {
                type_strict_match(RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfrec7_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorFloatTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            }
        },
        Abs: {
            lambda optree: True: {
                type_strict_match(RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_vectorFloatTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfabs_v_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=2, output_precision=RVV_vectorFloatTypeMap[(lmul, eltType)])
                    for (lmul, eltType) in RVV_vectorFloatTypeMap
            }
        },
        Min: {
             lambda optree: True: {
                 # generating mapping for all vv version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfmin_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for all vf version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vfmin_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for fv (swapped vf) version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=swapOperand(2, {0: 1}))
                     for (lmul, eltType) in RVV_vectorTypeMap
            },
             lambda optree: True: {
                # generating mapping for all vx version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmin_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorIntTypeMap
             },
         },
        Max: {
             lambda optree: True: {
                 # generating mapping for all vv version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    RVVIntrinsic("vfmax_vv_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for all vf version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vfmax_vf_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorFloatTypeMap
             },
             lambda optree: True: {
                # generating mapping for fv (swapped vf) version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_vectorTypeMap[(lmul, eltType)], RVV_VectorSize_T):
                    ComplexOperator(optree_modifier=swapOperand(2, {0: 1}))
                     for (lmul, eltType) in RVV_vectorTypeMap
            },
             lambda optree: True: {
                # generating mapping for all vx version of vfmul
                type_strict_match(RVV_vectorTypeMap[(lmul, eltType)], RVV_vectorTypeMap[(lmul, eltType)], eltType, RVV_VectorSize_T):
                    RVVIntrinsic("vmax_vx_%sm%d" % (RVVIntrSuffix[eltType], lmul), arity=3, output_precision=RVV_vectorTypeMap[(lmul, eltType)])
                     for (lmul, eltType) in RVV_vectorIntTypeMap
             },
         },
    },
    TypeCast: {
        None: {
            lambda optree: True: {
                # generating mapping for all vf version of vfadd
                # TODO/FIXME: TypeCast should not have length operand
                type_strict_match(RVV_vectorTypeMap[(lmul, dstEltType)], RVV_vectorTypeMap[(lmul, eltType)]):
                    RVVIntrinsic("vreinterpret_v_%sm%d_%sm%d" % (RVVIntrSuffix[eltType], lmul, RVVIntrSuffix[dstEltType], lmul), arity=1, output_precision=RVV_vectorTypeMap[(lmul, dstEltType)])
                    for (lmul, eltType) in RVV_vectorTypeMap for dstEltType in RVV_castEltTypeMapping[eltType]
            }
        }
    }
}

@UniqueTargetDecorator
class RISCV_RVV64(RISCV_RV64_CLANG):
    target_name = "rv64gv"
    vectorSizeType = RVV_VectorSize_T

    code_generation_table = {
        C_Code: rvv64_CCodeGenTable,
    }

    # approximation table map
    approx_table_map = rvv_approx_table_map


    def __init__(self):
        super().__init__()

    def get_compilation_options(self, ML_SRC_DIR):
        RISCV_ENV = self.getRiscvEnv()
        extraOpts = [f"-L{RISCV_ENV}/riscv64-unknown-elf/lib/",
                     f"--gcc-toolchain={RISCV_ENV}/ ",
                     "-menable-experimental-extensions",
                     "-march=rv64gcv0p10",
                     "-target riscv64"]
        return super(RISCV_RVV64, self).get_compilation_options(ML_SRC_DIR) + extraOpts

    def getGem5ExeCmd(self, test_file):
        """ build an execution command for a GEM5 runner"""
        try:
            riscvGem5Bin = os.environ["GEM5_BIN"]
        except KeyError:
            Log.report(Log.Warning, "GEM5_BIN env var must point to a valid gem5 program")
            riscvGem5Bin = "<GEM5_BIN undef>"

        try:
            gem5Cfg = os.environ["GEM5_CFG"]
        except KeyError:
            Log.report(Log.Warning, "GEM5_CFG env var must point to a valid gem5 configuration")
            gem5Cfg = "<GEM5_CFG undef>"
        cmd = f"{riscvGem5Bin} {gem5Cfg} --cmd {test_file}"
        return cmd

    def get_execution_command(self, test_file, runner="spike"):
        """ build an execution command which can be configured by a runner option """
        if runner == "spike":
            return self.getSpikeExecCmd(test_file, "RV64gcv")
        elif runner == "gem5":
            return self.getGem5ExeCmd(test_file)
        else:
            Log.report(Log.Error, f"unknown runner {runner} in target {self.__class__}")

# debug message
Log.report(LOG_BACKEND_INIT, "initializing RISC-V Vector target")