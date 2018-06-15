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
    BitLogicRightShift,
    BitLogicLeftShift,
    BitLogicAnd,
    LogicalNot,
    NearestInteger,
    ExponentInsertion,
    LogicalAnd,
    TypeCast,
    Test, Comparison,
    Return,
    FunctionObject
)
from metalibm_core.core.legalizer import (
    min_legalizer, max_legalizer, legalize_test, legalize_exp_insertion
)

from metalibm_core.code_generation.generator_utility import (
    ConstantOperator, FunctionOperator,
    type_strict_match, type_strict_match_list
)
from metalibm_core.code_generation.complex_generator import (
    ComplexOperator
)
from metalibm_core.code_generation.code_constant import LLVM_IR_Code
from metalibm_core.code_generation.abstract_backend import (
    LOG_BACKEND_INIT
)
from metalibm_core.code_generation.generic_processor import (
    GenericProcessor
)
from metalibm_core.code_generation.llvm_utils import llvm_ir_format

from metalibm_core.utility.log_report import Log

def llvm_negation_function(precision):
    """ build code generation operator for Negation operation.
        As LLVM-IR does not have neg we must build a subtraction from 0 """
    op = "fsub" if ML_FP_Format.is_fp_format(precision) else "sub"
    zero = "0.0" if ML_FP_Format.is_fp_format(precision) else "0"
    return LLVMIrTemplateOperator(
        "{op} {precision} {zero}, {{}}".format(
            zero=zero,
            op=op,
            precision=llvm_ir_format(precision),
        ),
        arity=1
    )

def llvm_not_function(precision):
    """ build a code generation operator for LogicalNot operation (unary)
        from the binary operations supported in LLVM-IR """
    op = "xor"
    one = 1
    return LLVMIrTemplateOperator(
        "{op} {precision} {one}, {{}}".format(
            op=op,
            one=one,
            precision=llvm_ir_format(precision)
        ), arity=1
    )

def llvm_ret_function(precision):
    return LLVMIrFunctionOperator(
        "ret", arity=1, void_function=True, output_precision=precision
    )

def llvm_bitcast_function(dst_precision, src_precision):
    return LLVMIrTemplateOperator(
        "bitcast {src_format} {{}} to {dst_format}".format(
            src_format=llvm_ir_format(src_precision),
            dst_format=llvm_ir_format(dst_precision)
        ),
        arity=1
    )

def llvm_fcomp_function(predicate, precision):
    return LLVMIrFunctionOperator("fcmp {}".format(predicate), arity=2, output_precision=precision)

def llvm_icomp_function(predicate, precision):
    return LLVMIrFunctionOperator("icmp {}".format(predicate), arity=2, output_precision=precision)

def llvm_op_function(name, precision, arity=2):
    return LLVMIrFunctionOperator(name, arity=2, output_precision=precision)


class LLVMIrFunctionOperator(FunctionOperator):
    default_prefix = "tmp"
    def generate_call_code(self, result_arg_list):
        return "{function_name} {output_precision} {arg_list}".format(
            output_precision=llvm_ir_format(self.output_precision),
            function_name=self.function_name,
            arg_list = ", ".join(
                [var_arg.get() for var_arg in result_arg_list]
            )
        )

class LLVMIrIntrinsicOperator(LLVMIrFunctionOperator):
    def __init__(self, function_name, input_formats=None, **kw):
        self.input_formats = [] if input_formats is None else input_formats
        LLVMIrFunctionOperator.__init__(self, function_name, **kw)


    def register_prototype(self, optree, code_object):
        if self.declare_prototype:
            code_object.declare_function(
                self.function_name, self.declare_prototype
            )

    @property
    def declare_prototype(self):
        return FunctionObject(
            self.function_name,
            self.input_formats,
            self.output_precision,
            self,
        )
    @declare_prototype.setter
    def declare_prototype(self, value):
        # discard declare_prototype change
        pass

    def generate_call_code(self, result_arg_list):
        return "call {output_precision} @{function_name}({arg_list})".format(
            output_precision=llvm_ir_format(self.output_precision),
            function_name=self.function_name,
            arg_list = ", ".join(
                ["%s %s" % (llvm_ir_format(var_arg.precision), var_arg.get()) for var_arg in result_arg_list]
            )
        )

class LLVMIrTemplateOperator(LLVMIrFunctionOperator):
    def generate_call_code(self, result_arg_list):
        return self.function_name.format(
            *tuple(var_arg.get() for var_arg in result_arg_list)
        )

def generate_comp_mapping(predicate, fdesc, idesc):
     return dict(
        # floating-point comparison mapping
         [(
             type_strict_match_list([ML_Bool, ML_Int32], [precision], [precision]),
             llvm_fcomp_function(fdesc, precision)
         ) for precision in [
             ML_Binary32, ML_Binary64,
         ]] +
         # vectorial floating-point comparison mapping
         [(
             type_strict_match(v4bool, precision, precision),
             llvm_fcomp_function(fdesc, precision)
         ) for precision in [
             v4float32, v4float64,
         ]] +
         # integer comparison mapping
         [(
             type_strict_match_list([ML_Bool, ML_Int32], [precision], [precision]),
             llvm_icomp_function(idesc, precision)
         ) for precision in [
             ML_Int32, ML_Int64, ML_Int128, ML_Int256
         ]] +
         # vectorial integer comparison mapping
         [(
             type_strict_match(v4bool, precision, precision),
             llvm_icomp_function(idesc, precision)
         ) for precision in [
             v4int32
         ]] 
     )

def legalize_integer_nearest(optree):
    """ transform a NearestInteger node floating-point to integer 
        into a sequence of floating-point NearestInteger and Conversion.
        This conversion is lossy """
    op_input = optree.get_input(0)
    int_precision = {
        v4float32: v4int32,
        ML_Binary32: ML_Int32
    }[optree.get_precision()]
    return Conversion(
        NearestInteger(
            op_input,
            precision=int_precision
        ),
        precision=optree.get_precision()
    )

llvm_ir_code_generation_table = {
    NearestInteger: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    LLVMIrTemplateOperator("fptosi float {} to i32", arity=1),
                type_strict_match(ML_Binary32, ML_Binary32):
                    LLVMIrIntrinsicOperator("llvm.nearbyint.f32", arity=1, output_precision=ML_Binary32, input_formats=[ML_Binary32]),
                # vector version
                type_strict_match(v4float32, v4float32):
                    LLVMIrIntrinsicOperator("llvm.nearbyint.f32", arity=1, output_precision=v4float32, input_formats=[v4float32]),
                type_strict_match(v4int32, v4float32):
                    ComplexOperator(optree_modifier=legalize_integer_nearest),
            },
        },
    },
    Addition: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("add", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ] + [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("fadd", precision),
                    ) for precision in [
                        ML_Binary32, ML_Binary64,
                        v2float32, v4float32, v8float32,
                        v2float64, v4float64, v8float64,
                    ]
                ]
                )
        },
    },
    Negation: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision),
                            llvm_negation_function(precision)
                    ) for precision in [
                        ML_Binary32, ML_Binary64,
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,

                        v2float32, v4float32, v8float32,
                        v2float64, v4float64, v8float64,
                    ]
                ]
                )
        },
    },
    BitLogicAnd: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("and", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ]
                )
        },
    },
    LogicalAnd: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("and", precision)
                    ) for precision in [
                        ML_Bool,
                        v2bool, v4bool, v8bool,
                    ]
                ]
                )
        },
    },
    LogicalNot: {
        None: {
            (lambda _: True): 
                dict(
                [
                    (
                        type_strict_match(precision, precision),
                            llvm_not_function(precision)
                    ) for precision in [
                        ML_Bool,
                        v2bool, v4bool, v8bool,
                    ]
                ]
                )
        },
    },
    BitLogicRightShift: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("lshr", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ]
                )
        },
    },
    BitLogicLeftShift: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("shl", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ]
                )
        },
    },
    TypeCast: {
        None: {
            (lambda _: True): {
                type_strict_match(ML_Int32, ML_Binary32):
                    llvm_bitcast_function(ML_Int32, ML_Binary32),
                type_strict_match(ML_Binary32, ML_Int32):
                    llvm_bitcast_function(ML_Binary32, ML_Int32),
            },
        },
    },
    Subtraction: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("sub", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ] + [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("fsub", precision),
                    ) for precision in [
                        ML_Binary32, ML_Binary64,
                        v2float32, v4float32, v8float32,
                        v2float64, v4float64, v8float64,
                    ]
                ]
                )
        },
    },
    Multiplication: {
        None: {
            (lambda _: True):
                dict(
                [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("mul", precision)
                    ) for precision in [
                        ML_Int32, ML_Int64,
                        v2int32, v4int32, v8int32,
                        v2int64, v4int64, v8int64,
                    ]
                ] + [
                    (
                        type_strict_match(precision, precision, precision),
                            llvm_op_function("fmul", precision),
                    ) for precision in [
                        ML_Binary32, ML_Binary64,
                        v2float32, v4float32, v8float32,
                        v2float64, v4float64, v8float64,
                    ]
                ]
                )
        },
    },
    # operation class
    ExponentInsertion: {
        # operation specifier
        ExponentInsertion.Default: {
            # predicate : ML_Operation -> bool
            (lambda _: True): {
                type_strict_match(ML_Binary32, ML_Int32):
                    ComplexOperator(
                        optree_modifier=legalize_exp_insertion(ML_Binary32)
                    ),
                type_strict_match(v4float32, v4int32):
                    ComplexOperator(
                        optree_modifier=legalize_exp_insertion(v4float32)
                    ),
            },
        },
    },
    Return: {
        None: {
            lambda _: True:
                dict(
                    (
                        type_strict_match(precision, precision),
                        llvm_ret_function(precision)
                    ) for precision in [
                        ML_Int32, ML_Int64, ML_Binary32, ML_Binary64,
                        v2float32, v4float32, v8float32,
                        v2float64, v4float64, v8float64,
                    ]
                )
        },
    },
    Comparison: {
        Comparison.GreaterOrEqual: {
            lambda _: True :
                generate_comp_mapping(Comparison.GreaterOrEqual, "oge", "ge")
        },
        Comparison.Greater: {
            lambda _: True :
                generate_comp_mapping(Comparison.Greater, "ogt", "sgt")
        },
        Comparison.Less: {
            lambda _: True :
                generate_comp_mapping(Comparison.Less, "olt", "slt")
        },
        Comparison.LessOrEqual: {
            lambda _: True :
                generate_comp_mapping(Comparison.LessOrEqual, "ole", "sle")
        },
        Comparison.Equal: {
            lambda _: True :
                generate_comp_mapping(Comparison.Equal, "oeq", "eq")
        },
        Comparison.NotEqual: {
            lambda _: True :
                generate_comp_mapping(Comparison.NotEqual, "ne", "ne")
        },
    },
    Test: {
        Test.IsInfOrNaN: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]):
                    ComplexOperator(legalize_test),
                type_strict_match_list([v4bool], [v4float32]):
                    ComplexOperator(legalize_test),
            }
        },
        Test.IsSubnormal: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]):
                    ComplexOperator(legalize_test),
            }
        },
        Test.IsNaN: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]):
                    ComplexOperator(legalize_test),
            }
        },
        Test.IsInfty: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]):
                    ComplexOperator(legalize_test),
            }
        },
        Test.IsSignalingNaN: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]):
                    ComplexOperator(legalize_test),
            }
        },
        Test.IsMaskNotAnyZero: {
            lambda _: True: {
                type_strict_match(ML_Bool, v4bool):
            }
        },
    },
}




## Generic C Capable Backend
class LLVMBackend(GenericProcessor):
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
        GenericProcessor.__init__(self, *args)
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
