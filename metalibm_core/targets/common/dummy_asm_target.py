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


from metalibm_core.core.target import UniqueTargetDecorator

from metalibm_core.core.ml_formats import (
    ML_Bool,
    v2bool, v4bool, v8bool,
    ML_Int32, ML_Int64, ML_Binary32, ML_Binary64,
    ML_Int128, ML_Int256,
    v2int32, v2int64, v2float32, v2float64,
    v4int32, v4int64, v4float32, v4float64,
    v8int32, v8int64, v8float32, v8float64,
    ML_FP_Format,
    ML_Void,
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
    FO_Result, FO_Arg,
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
from metalibm_core.code_generation.abstract_backend import (
    AbstractBackend)
from metalibm_core.code_generation.generic_processor import (
    instanciate_extra_passes)

from metalibm_core.utility.log_report import Log

import asmde.allocator as asmde

class DummyArchitecture(asmde.Architecture):
    def __init__(self, std_reg_num=16):
        asmde.Architecture.__init__(self,
            set([
                asmde.RegFileDescription(
                    asmde.Register.Std, std_reg_num,
                    asmde.PhysicalRegister,
                    asmde.VirtualRegister),
            ]),
            None)


def DummyAsmOperator(pattern, arity=1, **kw):
    return TemplateOperatorFormat(
        pattern, arg_map=({index: arg_obj for (index, arg_obj) in [(0, FO_Result())] + [(i+1, FO_Arg(i)) for i in range(arity)]}),
        **kw)

asm_code_generation_table = {
    Conversion: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    DummyAsmOperator("fixedw.rn {} = {}, 0", arity=1),
            },
        },
    },
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    DummyAsmOperator("faddww {} = {}, {}", arity=2),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    DummyAsmOperator("fmulw {} = {}, {}", arity=2),
            },
        },
    },
    Return: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Void):
                    TemplateOperatorFormat("ret", arity=0, void_function=True),
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
        # asmde.Architecture object
        self.architecture = DummyArchitecture()

    def generate_register(self, machine_register):
        return "$r{}".format(machine_register.register_id)

    ## return the compiler command line program to use to build
    #  test programs
    def get_compiler(self):
        return DummyAsmBackend.default_compiler

    ## Return a list of compiler option strings for the @p self target
    def get_compilation_options(self, ML_SRC_DIR):
        """ return list of compiler options """
        return [" "]

    def instanciate_pass_pipeline(self, pass_scheduler, processor, extra_passes, language=ASM_Code):
        """ instanciate an optimization pass pipeline for VectorBackend targets """
        EXTRA_PASSES = [
            # "beforecodegen:gen_basic_block",
            # "beforecodegen:basic_block_simplification",
            # "beforecodegen:ssa_translation",
        ]
        return instanciate_extra_passes(pass_scheduler, processor,
                                        EXTRA_PASSES + extra_passes,
                                        language=language,
                                        pass_slot_deps={})


# debug message
Log.report(LOG_BACKEND_INIT, "Initializing llvm backend target")
