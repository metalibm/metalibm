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
    ML_UInt32,
    ML_Int128, ML_Int256,
    v2int32, v2int64, v2float32, v2float64,
    v4int32, v4int64, v4float32, v4float64,
    ML_FP_Format,
    ML_Void,
    ML_Integer,
    ML_Standard_FixedPoint_Format,
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
    TableStore, TableLoad,
    VectorBroadcast,
)

from metalibm_core.core.bb_operations import SequentialBlock
from metalibm_core.core.machine_operations import (
    MaterializeConstant, RegisterCopy, RegisterAssign, SubRegister, VirtualRegister)
from metalibm_core.core.ml_complex_formats import is_pointer_format, ML_Pointer_Format
from metalibm_core.core.static_vectorizer import vectorize_format

from metalibm_core.code_generation.generator_utility import (
    TemplateOperatorFormat,
    FO_Result, FO_Arg,
    ConstantOperator, FunctionOperator,
    type_strict_match, type_strict_match_list,
    type_all_match, type_custom_match, FSM, TCM,
    type_table_index_match,
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


from metalibm_core.opt.generic_lowering import (
    GenericLoweringBackend, Pass_GenericLowering, LoweringAction)
from metalibm_core.core.passes import METALIBM_PASS_REGISTER

from metalibm_core.utility.log_report import Log

import asmde.allocator as asmde

class MachineEXU:
    def __init__(self, latency, pipeline=True, repeat_latency=1):
        self.latency = latency
        # is the unit pipelined ? (can it have multiple operations executing
        # at the same time ?)
        self.pipeline = pipeline
        # how many cycles must be waited before a new operation
        # can start executing ?
        self.repeat_latency = repeat_latency

DA_FPU = MachineEXU(4)
DA_BCU = MachineEXU(2)
DA_ALU = MachineEXU(1)
DA_LSU = MachineEXU(15)
DA_MAU = MachineEXU(2)

class DummyArchitecture(asmde.Architecture):
    REG_SIZE = 64
    # 64-bit addresses
    ADDR_SIZE = 64
    def __init__(self, std_reg_num=64):
        asmde.Architecture.__init__(self,
            set([
                asmde.RegFileDescription(
                    asmde.Register.Std, std_reg_num,
                    asmde.PhysicalRegister,
                    asmde.VirtualRegister),
            ]),
            None)

    def generate_ABI_physical_input_reg_tuples(self, ordered_input_regs):
        """ generate a list of physical registers for each
            virtual input register, enforcing architecture ABI """
        # argument allocation starts at register 0
        current_reg_index = 0

        phys_reg_list = []

        for reg in ordered_input_regs:
            ml_format = reg.precision
            if is_pointer_format(ml_format):
                reg_index = current_reg_index
                current_reg_index += 1
                phys_reg_list.append((self.get_unique_phys_reg_object(reg_index, asmde.Register.Std),))
            elif ml_format.get_bit_size() <= self.REG_SIZE:
                reg_index = current_reg_index
                current_reg_index += 1
                phys_reg_list.append((self.get_unique_phys_reg_object(reg_index, asmde.Register.Std),))
            else:
                raise NotImplementedError

        return phys_reg_list

    def generate_ABI_physical_output_reg_tuples(self, ordered_input_regs):
        """ generate a list of physical registers for each
            virtual input register, enforcing architecture ABI """
        # return value (retval) allocation starts at register 0
        current_reg_index = 0

        phys_reg_list = []

        for reg in ordered_input_regs:
            ml_format = reg.precision
            if ml_format.get_bit_size() <= self.REG_SIZE:
                reg_index = current_reg_index
                current_reg_index += 1
                phys_reg_list.append((self.get_unique_phys_reg_object(reg_index, asmde.Register.Std),))
            else:
                raise NotImplementedError

        return phys_reg_list

    def generate_virtual_reg(self, ml_reg):
        """ generate a tuple of asm virtual regs suitable to store
            ml_reg value """
        ml_format = ml_reg.precision
        if ml_format.get_bit_size() <= self.REG_SIZE:
            virt_reg = self.get_unique_virt_reg_object(ml_reg.get_tag(), asmde.Register.Std)
            return (virt_reg,)
        elif ml_format.get_bit_size() == 2 * self.REG_SIZE:
            return self.generate_virtual_pair_reg(ml_reg.get_tag())
        else:
            Log.report(Log.Error, "no virtual register suitable for format {}", ml_format)


    def generate_virtual_pair_reg(self, tag):
        """ Generate a linked pair (even, odd) of two registers """
        lo_reg = self.get_unique_virt_reg_object(tag + "_lo", reg_class=asmde.Register.Std, reg_constraint=asmde.even_indexed_register)
        hi_reg = self.get_unique_virt_reg_object(tag + "_hi", reg_class=asmde.Register.Std, reg_constraint=asmde.odd_indexed_register)
        lo_reg.add_linked_register(hi_reg, lambda color_map: [color_map[hi_reg] - 1])
        hi_reg.add_linked_register(lo_reg, lambda color_map: [color_map[lo_reg] + 1])
        return (lo_reg, hi_reg)


class MachineInstruction:
    """ hybrid between TemplateOperatorFormat and instruction
        with scheduling and bundling information """
    def __init__(self, exu=None):
        self.exu = exu


class StandardAsmOperator(MachineInstruction, TemplateOperatorFormat):
    def __init__(self, pattern, arity=1, exu=None, **kw):
        TemplateOperatorFormat.__init__(self, pattern, arg_map=({index: arg_obj for (index, arg_obj) in [(0, FO_Result())] + [(i+1, FO_Arg(i)) for i in range(arity)]}), *kw)
        MachineInstruction.__init__(self, exu)
class AdvancedAsmOperator(MachineInstruction, TemplateOperatorFormat):
    def __init__(self, pattern, exu=None, **kw):
        TemplateOperatorFormat.__init__(self, pattern, **kw)
        MachineInstruction.__init__(self, exu)

#def StandardAsmOperator(pattern, arity=1, **kw):
#    return TemplateOperatorFormat(
#        pattern, arg_map=({index: arg_obj for (index, arg_obj) in [(0, FO_Result())] + [(i+1, FO_Arg(i)) for i in range(arity)]}),
#        **kw)

asm_code_generation_table = {
    Conversion: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Int32, ML_Binary32):
                    StandardAsmOperator("fixedw.rn {} = {}, 0", arity=1, exu=DA_FPU),
            },
        },
    },
    Addition: {
        None: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_UInt32], [ML_UInt32, ML_Int32], [ML_Int32, ML_UInt32]):
                    StandardAsmOperator("addw {} = {}, {}", arity=2, exu=DA_ALU),

                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    StandardAsmOperator("faddw {} = {}, {}", arity=2, exu=DA_FPU),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64):
                    StandardAsmOperator("faddd {} = {}, {}", arity=2, exu=DA_FPU),

                type_strict_match(v4float32, v4float32, v4float32):
                    StandardAsmOperator("faddwq {} = {}, {}", arity=2, exu=DA_FPU),
            },
        },
    },
    Subtraction: {
        None: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_UInt32], [ML_UInt32, ML_Int32], [ML_Int32, ML_UInt32]):
                    # subtract from: op1 - op0 (reverse)
                    AdvancedAsmOperator("sbfw {} = {}, {}", arg_map={0: FO_Result(), 1: FO_Arg(1), 2: FO_Arg(0)}, arity=2, exu=DA_ALU),

                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    # subtract from: op1 - op0 (reverse)
                    AdvancedAsmOperator("fsbfw {} = {}, {}", arg_map={0: FO_Result(), 1: FO_Arg(1), 2: FO_Arg(0)}, arity=2, exu=DA_FPU),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64):
                    # subtract from: op1 - op0 (reverse)
                    AdvancedAsmOperator("fsbfd {} = {}, {}", arg_map={0: FO_Result(), 1: FO_Arg(1), 2: FO_Arg(0)}, arity=2, exu=DA_FPU),
            },
        },
    },
    Multiplication: {
        None: {
            lambda _: True: {
                type_strict_match_list([ML_Int32, ML_UInt32], [ML_UInt32, ML_Int32], [ML_Int32, ML_UInt32]):
                    StandardAsmOperator("mulw {} = {}, {}", arity=2, exu=DA_MAU),

                type_strict_match(ML_Binary32, ML_Binary32, ML_Binary32):
                    StandardAsmOperator("fmulw {} = {}, {}", arity=2, exu=DA_FPU),
                type_strict_match(ML_Binary64, ML_Binary64, ML_Binary64):
                    StandardAsmOperator("fmuld {} = {}, {}", arity=2, exu=DA_FPU),

                type_strict_match(v4float32, v4float32, v4float32):
                    StandardAsmOperator("fmulwq {} = {}, {}", arity=2, exu=DA_FPU),
            },
        },
    },
    Return: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Void):
                    AdvancedAsmOperator("ret", arity=0, void_function=True, exu=DA_BCU),
            },
        },
    },
    Comparison: {
        Comparison.Less: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_Int32, ML_Int32):
                    StandardAsmOperator("compw.lt {} = {}, {}", arity=2, exu=DA_ALU),
            }
        },
        Comparison.LessOrEqual: {
            lambda _: True: {
                type_strict_match(ML_Bool, ML_Int32, ML_Int32):
                    StandardAsmOperator("compw.le {} = {}, {}", arity=2, exu=DA_ALU),
            },
        },
    },
    MaterializeConstant: {
        None: {
            lambda _: True: {
                type_strict_match_list([ML_Integer, ML_Int32, ML_Int64, ML_Binary32, ML_Binary64, v2float32]):
                    StandardAsmOperator("maked {} = {}", arity=1, exu=DA_ALU),
            },
        },
    },
    RegisterCopy: {
        None: {
            lambda _: True: {
                # TODO/FIXME: should be distinguished based on format size
                type_all_match:
                    StandardAsmOperator("copyd {} = {}", arity=1, exu=DA_ALU),
            },
        },
    },
    VectorBroadcast: {
        None: {
            lambda _: True: {
                # TODO/FIXME: should be distinguished based on format size
                type_strict_match(v2float32, ML_Binary32):
                    StandardAsmOperator("vbcast {} = {}", arity=1, exu=DA_MAU),
            },
        },
    },
    TableStore: {
        None: {
            # 32-bit data store
            (lambda optree: optree.get_input(0).get_precision().get_bit_size() == 32): {
                type_custom_match(FSM(ML_Void), type_all_match, is_pointer_format, type_table_index_match):
                    AdvancedAsmOperator("sw {1}[{2}] = {0}", arity=3, void_function=True, exu=DA_LSU), 
            },
            # 64-bit data store
            (lambda optree: optree.get_input(0).get_precision().get_bit_size() == 64): {
                type_custom_match(FSM(ML_Void), type_all_match, is_pointer_format, type_table_index_match):
                    AdvancedAsmOperator("sd {1}[{2}] = {0}", arity=3, void_function=True, exu=DA_LSU), 
            },
            # 128-bit data store
            (lambda optree: optree.get_input(0).get_precision().get_bit_size() == 128): {
                type_custom_match(FSM(ML_Void), type_all_match, is_pointer_format, type_table_index_match):
                    AdvancedAsmOperator("sq {1}[{2}] = {0}", arity=3, void_function=True, exu=DA_LSU), 
            },
        },
    },
    TableLoad: {
        None: {
            # 32-bit data load
            (lambda optree: optree.get_precision().get_bit_size() == 32): {
                type_custom_match(FSM(ML_Binary32), is_pointer_format, type_table_index_match):
                    StandardAsmOperator("lw {} = {}[{}]", arity=2, exu=DA_LSU),
            },
            # 64-bit data load
            (lambda optree: optree.get_precision().get_bit_size() == 64): {
                type_custom_match(FSM(ML_Binary64), is_pointer_format, type_table_index_match):
                    StandardAsmOperator("ld {} = {}[{}]", arity=2, exu=DA_LSU),
            },
            # 128-bit data load
            (lambda optree: optree.get_precision().get_bit_size() == 128): {
                type_custom_match(FSM(v4float32), is_pointer_format, type_table_index_match):
                    StandardAsmOperator("lq {} = {}[{}]", arity=2, exu=DA_LSU),
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
        if isinstance(machine_register, VirtualRegister):
            if machine_register.is_named_register:
                return "$v-{}".format(machine_register.var_tag)
            else:
                return "$v{}".format(machine_register.register_id)
        elif isinstance(machine_register, SubRegister):
            super_reg = machine_register.super_register
            if isinstance(super_reg, VirtualRegister):
                return "$v{}[{}]".format(machine_register.super_register.register_id, machine_register.sub_id)
            else:
                return "$r{}".format(machine_register.super_register.register_id[machine_register.sub_id])
        else:
            return "${}".format("".join("r%d" % sub_id for sub_id in machine_register.register_id))

    def get_operation_resource(self, node):
        """ return the resources used to implement an operation """
        implementation = self.get_recursive_implementation(node, language=ASM_Code)
        return implementation.exu

    def generate_constant_expr(self, constant_node):
        """ generate the assembly value of a given Constant node """
        cst_format = constant_node.get_precision()
        if cst_format.is_vector_format():
            return self.generate_vector_constant_expr(constant_node)
        elif cst_format is ML_Integer:
            return "%d" % constant_node.get_value()
        elif isinstance(cst_format, ML_Standard_FixedPoint_Format):
            return "%d" % constant_node.get_value()
        elif isinstance(cst_format, ML_FP_Format):
            return "0x%x" % constant_node.get_precision().get_integer_coding(constant_node.get_value())
        else:
            return cst_format.get_cst(
                    constant_node.get_value(), language=ASM_Code)

    def generate_vector_constant_expr(self, constant_node):
        """ generate the assembly value of a given Constant node
            with vector values """
        cst_format = constant_node.get_precision()
        assert cst_format.is_vector_format()
        if isinstance(cst_format, ML_FP_Format):
            scalar_format = cst_format.get_scalar_format()
            hex_size = scalar_format.get_bit_size() / 8
            return "0x" + "".join(("{:0%dx}" % hex_size).format(scalar_format.get_integer_coding(value)) for value in constant_node.get_value())
        else:
            raise NotImplementedError

    def generate_conditional_branch(self, asm_generator, code_object, cb_node):
        cond = cb_node.get_input(0)
        if_bb = cb_node.get_input(1)
        if_label = asm_generator.get_bb_label(code_object, if_bb)

        cond_code = asm_generator.generate_expr(
            code_object, cond, language=ASM_Code)

        code_object << "cb.nez {cond}, {if_label}\n".format(
            cond=cond_code, # cond register
            if_label=if_label,
        )
    def generate_unconditional_branch(self, asm_generator, code_object, node):
        dest_bb = node.get_input(0)
        dest_label = asm_generator.get_bb_label(code_object, dest_bb)

        code_object << "goto {dest_label}\n".format(
            dest_label=dest_label,
        )

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
            "beforecodegen:gen_basic_block",
            "beforecodegen:basic_block_simplification",
            "beforecodegen:linearize_op_graph",
            # NEXT
            # "beforecodegen:dummy_asm_lowering",
            "beforecodegen:collapse_reg_copy",
            "beforecodegen:register_allocation",
            "beforecodegen:dump",
            "beforecodegen:simplify_bb_fallback",
        ]
        return instanciate_extra_passes(pass_scheduler, processor,
                                        EXTRA_PASSES + extra_passes,
                                        language=language,
                                        pass_slot_deps={})

def split_format(vector_format, num_chunk):
    """ split vector format into sub-format """
    return vectorize_format(vector_format.get_scalar_format(), num_chunk)

def split_register(register, num_chunk):
    """ split MachineRegister <register> into <num_chunk>
        sub-registers """
    reg_id = register.register_id
    sub_format = split_format(register.get_precision(), num_chunk)
    return [SubRegister(register, sub_id, reg_id, sub_format, register.var_tag) for sub_id in range(num_chunk)]

class SplitConstantAssign(LoweringAction):
    def __init__(self, num_chunk):
        self.num_chunk = num_chunk
    def lower_node(self, node):
        dst_reg = node.get_input(0)
        materialze_op = node.get_input(1)
        assert isinstance(materialze_op, MaterializeConstant)
        src_value = materialze_op.get_input(0)

        sub_regs = split_register(dst_reg, self.num_chunk)
        out_vsize = src_value.get_precision().get_vector_size() // self.num_chunk
        out_vformat = vectorize_format(src_value.get_precision().get_scalar_format(), out_vsize)
        sub_values = [MaterializeConstant(Constant([src_value.get_value()[chunk_id * out_vsize + sub_id] for sub_id in range(out_vsize)], precision=out_vformat), precision=out_vformat) for chunk_id in range(self.num_chunk)]

        # single RegisterAssign is lowered into a sequence
        # of sub-register assign
        lowered_sequence = SequentialBlock(
            *tuple(RegisterAssign(sub_reg, sub_value) for sub_reg, sub_value in zip(sub_regs, sub_values))
        )
        return lowered_sequence

    def __call__(self, node):
        return self.lower_node(node)

    def get_source_info(self):
        """ required as implementation origin indicator by AbstractBackend """
        return None

class SplitVectorBroadCast(LoweringAction):
    def __init__(self, num_chunk):
        self.num_chunk = num_chunk
    def lower_node(self, node):
        dst_reg = node.get_input(0)
        src_value = node.get_input(1).get_input(0)

        sub_regs = split_register(dst_reg, self.num_chunk)
        out_vsize = dst_reg.get_precision().get_vector_size() // self.num_chunk
        out_vformat = vectorize_format(src_value.get_precision(), out_vsize)
        sub_values = [VectorBroadcast(src_value, precision=out_vformat) for chunk_id in range(self.num_chunk)]

        # single RegisterAssign is lowered into a sequence
        # of sub-register assign
        lowered_sequence = SequentialBlock(
            *tuple(RegisterAssign(sub_reg, sub_value) for sub_reg, sub_value in zip(sub_regs, sub_values))
        )
        return lowered_sequence

    def __call__(self, node):
        return self.lower_node(node)

    def get_source_info(self):
        """ required as implementation origin indicator by AbstractBackend """
        return None


DUMMY_ASM_TARGET_LOWERING_TABLE = {
    MaterializeConstant: {
        None: {
            lambda _: True: {
                type_strict_match_list([v4float32, v4float32]):
                    SplitConstantAssign(2),
            },
        },
    },
    VectorBroadcast: {
        None: {
            lambda _: True: {
                type_strict_match_list([v4float32, ML_Binary32]):
                    SplitVectorBroadCast(2),
            },
        },
    },
}


def lowering_target_register(cls):
    """ Decorator for a Lowering target which generate the associated lowering
        optimization pass and registers it automatically """
    @METALIBM_PASS_REGISTER
    class TargetLowering(Pass_GenericLowering):
        """ Target specific lowering """
        pass_tag = "{}_lowering".format(cls.target_name)
        def __init__(self, _target):
            Pass_GenericLowering.__init__(self, cls(), description="lowering for specific target")

@lowering_target_register
class DummyAsmTargetLowering(GenericLoweringBackend):
    # adding first default level of indirection
    target_name = "dummy_asm"
    lowering_table = {
        None: DUMMY_ASM_TARGET_LOWERING_TABLE
    }

    @staticmethod
    def get_operation_keys(node):
        """ unwrap RegisterAssign to generate operation key from
            node's operation """
        if isinstance(node, RegisterAssign):
            # if node is a RegisterAssign, we look for an implementation
            # for the wrapped operation
            dst = node.get_input(0)
            operation = node.get_input(1)
            return GenericLoweringBackend.get_operation_keys(operation)
        else:
            return GenericLoweringBackend.get_operation_keys(node)

# debug message
Log.report(LOG_BACKEND_INIT, "Initializing llvm backend target")
