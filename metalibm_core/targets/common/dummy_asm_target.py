# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          Apr 26th, 2020
# last-modified:    Apr 26th, 2020
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from ..core.target import UniqueTargetDecorator

from metalibm_core.core.ml_formats import (
    ML_Bool,
    v2bool, v4bool, v8bool,
    ML_Int32, ML_Int64, ML_Binary32, ML_Binary64,
    ML_Int128, ML_Int256,
    v2int32, v2int64, v2float32, v2float64,
    v4int32, v4int64, v4float32, v4float64,
    v8int32, v8int64, v8float32, v8float64,
    ML_FP_Format,
)
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_operations import (
    Addition, Subtraction, Multiplication,
    Negation,
    BitLogicRightShift, BitLogicLeftShift,
    BitLogicAnd,
    NearestInteger,
    ExponentInsertion,
    LogicalAnd, LogicalNot, LogicalOr,
    Test, Comparison,
    Return,
    FunctionObject,
    Conversion, TypeCast,
    VectorElementSelection,
    Constant,
    FusedMultiplyAdd,
    ReciprocalSeed,
)

from metalibm_core.code_generation.generator_utility import (
    TemplateOperatorFormat,
    ConstantOperator, FunctionOperator,
    type_strict_match, type_strict_match_list
)
from metalibm_core.code_generation.complex_generator import (
    ComplexOperator
)
from metalibm_core.code_generation.code_constant import ASM_Code
from metalibm_core.code_generation.abstract_backend import (
    LOG_BACKEND_INIT
)
from metalibm_core.code_generation.generic_processor import (
    GenericProcessor
)

from metalibm_core.utility.log_report import Log



asm_code_generation_table = {
    Conversion: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    TemplateOperatorFormat("fixedw.rn {} = {}, 0", arity=2),
            },
        },
    },
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    TemplateOperatorFormat("faddww {} = {}, {}", arity=2),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    TemplateOperatorFormat("fmulw {} = {}, {}", arity=2),
            },
        },
    },
}




@UniqueTargetDecorator
class DummyAsmBackend(AbstractBackend):
    """ Dummy class to generate an abstract low-level assembly program stream """
    target_name = "dummy_asm_backend"

    # code generation table map
    code_generation_table = {
        ASM_Code: asm_code_generation_table,
    }

    def __init__(self, *args):
        super().__init__(*args)
        self.simplified_rec_op_map[ASM_Code] = self.generate_supported_op_map(language=ASM_Code)


    ## return the compiler command line program to use to build
    #  test programs
    def get_compiler(self):
        return DummyAsmBackend.default_compiler

    ## Return a list of compiler option strings for the @p self target
    def get_compilation_options(self, ML_SRC_DIR):
        """ return list of compiler options """
        return [" "]

    def instanciate_pass_pipeline(self, pass_scheduler, processor, extra_passes, language=LLVM_IR_Code):
        """ instanciate an optimization pass pipeline for VectorBackend targets """
        EXTRA_VECTOR_PASSES = [
            "beforecodegen:gen_basic_block",
            "beforecodegen:basic_block_simplification",
            "beforecodegen:ssa_translation",
        ]
        return GenericProcessor.instanciate_pass_pipeline(self, pass_scheduler, processor,
                                                          EXTRA_VECTOR_PASSES + extra_passes,
                                                          language=language)


# debug message
Log.report(LOG_BACKEND_INIT, "Initializing llvm backend target")
