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
# created:          Sep  5th, 2021
# last-modified:    Sep  5th, 2021
#
# Author(s):        Nicolas Brunie <metalibmdev@gmail.com>
###############################################################################

import os

from metalibm_core.core.target import UniqueTargetDecorator
from metalibm_core.core.ml_operations import (
    ReadTimeStamp, NearestInteger, Conversion)
from metalibm_core.core.ml_formats import (
    ML_Int64, ML_Int32,
    ML_Binary64, ML_Binary32)

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import (
    AsmInlineOperator, FO_Result, FO_Arg, type_strict_match)
from metalibm_core.code_generation.complex_generator import ComplexOperator

from metalibm_core.utility.log_report import Log

from metalibm_core.targets.common.vector_backend import VectorBackend


rdcycleOperator = AsmInlineOperator(
"""{
    unsigned long cycles;
    asm volatile ("rdcycle %%0 " : "=r" (cycles));
    %s = cycles;
}""",
    arg_map = {0: FO_Result(0)},
    arity = 0
)

def RV_singleOpAsmTemplate(insn, regDst="r", regSrc="f"):
    singleOpOperator = AsmInlineOperator(
   """asm volatile ("{insn}" : "={regDst}" (%s) : "{regSrc}"(%s));\n""".format(insn=insn, regDst=regDst, regSrc=regSrc),
        arg_map = {0: FO_Result(0), 1: FO_Arg(0)},
        arity=1
    )
    return singleOpOperator

def lowerNearestInteger(intFormat, targetFormat):
    """ expand conversion into a conversion from
        conv's input to <intFormat> and then to <targetFormat> """
    def modifier(conv):
        op = conv.get_input(0)
        return Conversion(NearestInteger(op, precision=intFormat), precision=targetFormat)
    return modifier

rv64CCodeGenTable = {
    ReadTimeStamp: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int64): rdcycleOperator
            }
        }
    },
    # Conversion are mapped to function by default
    # so we lower them explicity to less-contrained
    # implementation
    NearestInteger: {
        None: {
            lambda optree: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    RV_singleOpAsmTemplate("fcvt.w.s %%0, %%1, rne"),
                type_strict_match(ML_Binary32, ML_Binary32):
                    ComplexOperator(optree_modifier=lowerNearestInteger(ML_Int32, ML_Binary32)),
                type_strict_match(ML_Int64, ML_Binary64):
                    RV_singleOpAsmTemplate("fcvt.l.d %%0, %%1, rne"),
                type_strict_match(ML_Int32, ML_Binary64):
                    ComplexOperator(optree_modifier=lowerNearestInteger(ML_Int64, ML_Int32)),
                type_strict_match(ML_Binary64, ML_Binary64):
                    ComplexOperator(optree_modifier=lowerNearestInteger(ML_Int64, ML_Binary64)),
            },
        },
    },
}

def buildRVCompilerPath():
    try:
        RISCV_CC = os.environ["RISCV_CC"]
    except KeyError:
        Log.report(Log.Warning, "RISCV_CC env variable must be set such than $RISCV/bin/riscv64-unknown-elf-gcc is accessible")
        RISCV_CC = "<RISCV_CC undef>"
    compiler = RISCV_CC #"{}/bin/riscv64-unknown-elf-gcc".format(RISCV)
    return compiler


class RISCV_RV64_Common(VectorBackend):
    default_compiler = buildRVCompilerPath()
    # only cross-compilation (not binary embedding in python) is currently supported
    support_embedded_bin = False
    cross_platform = True

    code_generation_table = {
        C_Code: rv64CCodeGenTable,
    }

    def __init__(self):
        super().__init__()

    def get_compilation_options(self, ML_SRC_DIR):
        return super(RISCV_RV64_Common, self).get_compilation_options(ML_SRC_DIR) + ["-march=rv64gc"]

    def get_execution_command(self, test_file):
        return self.getSpikeExecCmd(test_file, "RV64gc")

    def getSpikeExecCmd(self, test_file, isa="RV64gc"):
        try:
            pk_bin = os.environ["PK_BIN"]
        except KeyError:
            Log.report(Log.Warning, "PK_BIN env var must point to proxy-kernel image")
            pk_bin = "<PK_BIN undef>"

        try:
            spike_bin = os.environ["SPIKE_BIN"]
        except KeyError:
            Log.report(Log.Warning, "SPIKE_BIN env var must point to spike simulator binary")
            spike = "<SPIKE_BIN undef>"
        cmd = f"{spike_bin} --isa={isa} {pk_bin} {test_file}"
        return cmd


@UniqueTargetDecorator
class RISCV_RV64(RISCV_RV64_Common):
    target_name = "rv64g"


@UniqueTargetDecorator
class RISCV_RV64_CLANG(RISCV_RV64_Common):
    target_name = "rv64g-clang"

    def getRiscvEnv(self):
        try:
            RISCV_ENV = os.environ["RISCV"]
        except KeyError:
            Log.report(Log.Warning, "RISCV env variable must be set such than $RISCV/riscv64-unknown-elf/lib/ is accessible")
            RISCV_ENV = "<RISCV undef>"
        return RISCV_ENV

    def get_compilation_options(self, ML_SRC_DIR):
        RISCV_ENV = self.getRiscvEnv()
        extraOpts = [f"-L{RISCV_ENV}/riscv64-unknown-elf/lib/",
                    f"--gcc-toolchain={RISCV_ENV}/ ",
                    "-march=rv64gc",
                    "-target riscv64"]
        return super(RISCV_RV64_CLANG, self).get_compilation_options(ML_SRC_DIR) + extraOpts 

# debug message
Log.report(LOG_BACKEND_INIT, "initializing RISC-V targets")
