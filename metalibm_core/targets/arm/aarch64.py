# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2022 Nicolas Brunie
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
# created:          Apr 25th, 2022
# last-modified:    Apr 25th, 2022
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


# TODO: this should be filled with an assembly snippet to read the main time counter
# it is used to evaluate performance in function benchmarks
rdcycleOperator = AsmInlineOperator(
"""{
    unsigned long cycles;
    asm volatile ("rdcycle %%0 " : "=r" (cycles));
    %s = cycles;
}""",
    arg_map = {0: FO_Result(0)},
    arity = 0
)

# code generation table for a C-backend
aarch64CCodeGenTable = {
    ReadTimeStamp: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int64): rdcycleOperator
            }
        }
    },
    # Empty example of function mapping
    # <Operation-class>: {
    #   <Specifier>: {
    #      <predicate-function> optree: bool: {
    #          <interface/type matching>: implementation
    #      }
    #   }
    # }
    NearestInteger: {
        None: {
            lambda optree: True: {
                # TODO: uncomment and add proper implementation mapping
                # type_strict_match(ML_Int32, ML_Binary32):
            },
        },
    },
}

def buildArmCompilerPath():
    """ helper to build a path to a valid arm compiler """
    try:
        ARM_CC = os.environ["ARM_CC"]
    except KeyError:
        Log.report(Log.Warning, "ARM_CC env variable must be set such than $ARM_CC is a valid aarch64 compiler")
        ARM_CC = "<ARM_CC undef>"
    compiler = ARM_CC
    return compiler


class ARM_Aarch64_Common(VectorBackend):
    default_compiler = buildArmCompilerPath()
    # only cross-compilation (not binary embedding in python) is currently supported
    support_embedded_bin = False
    cross_platform = True

    code_generation_table = {
        C_Code: aarch64CCodeGenTable,
    }

    def __init__(self):
        super().__init__()

    def get_compilation_options(self, ML_SRC_DIR):
        # TODO: update default compilation options
        return super(ARM_Aarch64_Common, self).get_compilation_options(ML_SRC_DIR) + ["-march=aarch64"]

    def get_execution_command(self, test_file):
        # TODO: build a command line string to execute <test_file>
        raise NotImplementedError



@UniqueTargetDecorator
class ARM_Aarch64(ARM_Aarch64_Common):
    target_name = "aarch64"


@UniqueTargetDecorator
class ARM_Aarch64_CLANG(ARM_Aarch64_Common):
    target_name = "aarch64-clang"


    def get_compilation_options(self, ML_SRC_DIR):
        extraOpts = ["-march=aarch64"]
        return super(ARM_Aarch64_CLANG, self).get_compilation_options(ML_SRC_DIR) + extraOpts 

# debug message
Log.report(LOG_BACKEND_INIT, "initializing ARM targets")
