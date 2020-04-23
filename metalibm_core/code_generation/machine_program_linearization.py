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
# created:          Apr 23rd, 2020
# last-modified:    Apr 23rd, 2020
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import sys
import copy

from ..core.ml_operations import (
    Addition,
    Variable, Constant, ConditionBlock, Return, TableLoad, Statement,
    SpecificOperation, Conversion, FunctionObject,
    ReferenceAssign, Loop,
)
from ..core.bb_operations import (
    BasicBlockList,
    BasicBlock, ConditionalBranch, UnconditionalBranch,
    PhiNode,
)
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from .generator_utility import C_Code, Gappa_Code, RoundOperator
from .code_element import CodeVariable, CodeExpression
from .code_object import MultiSymbolTable

from ..utility.log_report import Log

from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.llvm_utils import llvm_ir_format


class MachineInsnGenerator(object):
    """ LLVM-IR language code generator """
    language = LLVM_IR_Code

    def __init__(self, processor):
        self.processor = processor
        self.memoization_map = {}
        # map of basic blocks (bb node -> label)
        self.bb_map = {}

        # mapping Variable -> MachineRegister
        self.var_register_map = {}
        # unique identifier for new virtual registers
        self.register_id = -1

    def get_memoization(self, node):
        return self.memoization_map[node]

    def add_memoization(self, node, node_reg):
        self.memoization_map[node] = node_reg

    def get_new_register(self, register_format, var_tag=None):
        self.register_id += 1
        return MachineRegister(register_id, precision=regiser_format, var_tag=var_tag)

    def get_var_register(self, var_node):
        if not var_node in self.var_register_map:
            self.var_register_map[var_node] = self.get_new_register(var_node.get_precision(), var_tag=var_node.get_tag())
        return self.var_register_map[var_node]

    # force_variable_storing is not supported
    def linearize_graph(self, current_bb, node):
        """ linearize operation graph """

        # search if <optree> has already been processed
        if self.has_memoization(node):
            return self.get_memoization(node)

        result = None
        # implementation generation
        if isinstance(node, CodeVariable):
            raise NotImplementedError

        elif isinstance(node, Variable):
            var_register = self.get_var_register(node)
            result = var_register

        elif isinstance(node, Constant):
            # constant are unmodified
            result = node

        elif isinstance(node, BasicBlock):
            linearized_bb = BasicBlock()
            for op in node.inputs:
                self.linearize_graph(linearized_bb, op)
            result = linearized_bb

        elif isinstance(node, ConditionalBranch):
            cond = node.get_input(0)
            if_bb = node.get_input(1)
            else_bb = node.get_input(2)

            cond_code = self.linearized_bb(current_bb, cond)
                code_object, cond, folded=folded, language=language)

            return None

        elif isinstance(node, UnconditionalBranch):
            dest_bb = node.get_input(0)
            code_object << "br label %{}\n".format(self.get_bb_label(code_object, dest_bb))
            return None

        elif isinstance(node, BasicBlockList):
            new_bb_list = BasicBlockList()
            for bb in node.inputs:
                linearized_bb = self.linearized_bb(None, bb)
                new_bb_list.add(linearized_bb)
            result = new_bb_list

        elif isinstance(node, (Statement, ConditionBlock, Loop)):
            Log.report(Log.Error, "{} are not supported in MI graph linearization "
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", node.__class__, node)

        elif isinstance(optree, PhiNode):
            output_var = optree.get_input(0)
            output_var_code = self.generate_expr(
                code_object, output_var, folded=folded, language=language)

            value_list = []
            for input_var, bb_var in zip(optree.get_inputs()[1::2], optree.get_inputs()[2::2]):
                assert isinstance(input_var, Variable)
                assert isinstance(bb_var, BasicBlock)
                input_var = self.generate_expr(
                    code_object, input_var, folded=folded, language=language
                )
                bb_label = self.get_bb_label(code_object, bb_var)
                value_list.append("[{var}, %{bb}]".format(var=input_var.get(), bb=bb_label))

            code_object << "{output_var} = phi {precision} {value_list}\n".format(
                output_var=output_var_code.get(),
                precision=llvm_ir_format(precision=output_var.get_precision()),
                value_list=(", ".join(value_list))
            )

            return None

        elif isinstance(node, ReferenceAssign):
            output_var = node.get_input(0)
            result_value = node.get_input(1)

            # TODO/FIXME: manage cases where ReferenceAssign is not directly
            #             a Variable (e.g. VectorElementSelection)
            var_register = self.get_var_register(output_var)

            result_value_reg = self.linearize_graph(current_bb, result_value)
            result = RegisterAssign(var_register, result_value_reg)

            # adding RegisterAssign to current basic-block
            current_bb.add(result)

        elif is_leaf_node(node):
            result = self.linearize_graph(current_bb, node)

        elif isinstance(node, GeneralArithmeticOperation):
            op_regs = [self.linearize_graph(op) for op in node.inputs]
            result_reg = self.get_new_register()
            result = RegisterAssign(
                result_reg,
                node.__class__(*tuple(op_regs))
            )
            # adding RegisterAssign to current basic-block
            current_bb.add(result)

        else:
            Log.report(Log.report, "node unsupported in linearize_graph: {}", node)


        # registering result into memoization table
        self.add_memoization(node, result)

        return result



