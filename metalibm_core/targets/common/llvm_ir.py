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
# created:          Apr  4th, 2018
# last-modified:    Apr  4th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import os, inspect
from sollya import S2


from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Int64, ML_Binary32, ML_Binary64, v4float32, v4float64
)
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_operations import (
    Addition, Return
)
from metalibm_core.core.legalizer import min_legalizer, max_legalizer

from metalibm_core.code_generation.generator_utility import (
    ConstantOperator, FunctionOperator,
    type_strict_match
)
from metalibm_core.code_generation.code_constant import LLVM_IR_Code
from metalibm_core.code_generation.abstract_backend import (
    AbstractBackend, LOG_BACKEND_INIT
)
from metalibm_core.code_generation.llvm_utils import llvm_ir_format

from metalibm_core.utility.log_report import Log



class LLVMIrFunctionOperator(FunctionOperator):
    default_prefix = "%tmp"
    def generate_call_code(self, result_arg_list):
        return "{function_name} {output_precision} {arg_list}".format(
            output_precision=llvm_ir_format(self.output_precision),
            function_name=self.function_name,
            arg_list = ", ".join(
                [var_arg.get() for var_arg in result_arg_list]
            )
        )


llvm_ir_code_generation_table = {
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Int32, ML_Int32):
                    LLVMIrFunctionOperator("add", arity=2, output_precision=ML_Int32),
            }
        },
    },
    Return: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Int32):
                    LLVMIrFunctionOperator("ret", arity=1, void_function=True, output_precision=ML_Int32),
            }
        },
    },
}




## Generic C Capable Backend
class LLVMBackend(AbstractBackend):
    """ Generic class for instruction selection,
        corresponds to a portable C-implementation """
    target_name = "llvm"
    TargetRegister.register_new_target(target_name, lambda _: LLVMBackend)

    default_compiler = "clang"


    # code generation table map
    code_generation_table = {
        LLVM_IR_Code: llvm_ir_code_generation_table,
    }

    def __init__(self, *args):
        AbstractBackend.__init__(self, *args)
        self.simplified_rec_op_map[LLVM_IR_Code] = self.generate_supported_op_map(language=LLVM_IR_Code)


    ## return the compiler command line program to use to build
    #  test programs
    def get_compiler(self):
        return LLVMBackend.default_compiler

    ## Return a list of compiler option strings for the @p self target
    def get_compilation_options(self):
        """ return list of compiler options """
        return [" "]


# debug message
Log.report(LOG_BACKEND_INIT, "Initializing llvm backend target")
