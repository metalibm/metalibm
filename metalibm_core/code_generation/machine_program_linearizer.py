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
    is_leaf_node,
    TableStore,
    ML_ArithmeticOperation,
)
from ..core.bb_operations import (
    BasicBlockList,
    BasicBlock, ConditionalBranch, UnconditionalBranch,
    PhiNode,
)
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from ..core.legalizer import is_constant

from .generator_utility import C_Code, Gappa_Code, RoundOperator
from .code_element import CodeVariable, CodeExpression
from .code_object import MultiSymbolTable

from ..utility.log_report import Log

from metalibm_core.code_generation.code_function import CodeFunction

from metalibm_core.core.machine_operations import (
    MachineRegister, RegisterAssign, MaterializeConstant,
    RegisterCopy, MaterializeConstant,
    MachineProgram
)

class MachineInsnGenerator(object):
    """ Machine Instruction generator """

    def __init__(self, processor):
        self.processor = processor
        self.memoization_map = {}
        # map of basic blocks (bb node -> label)
        self.bb_map = {}

        # mapping Variable -> MachineRegister
        self.var_register_map = {}
        # mapping Node -> MachineRegister (superset of self.var_register_map and subset
        # of memoization_map)
        self.node_to_register_map = {}
        # unique identifier for new virtual registers
        self.register_id = -1

        # mapping of pre-linearization to linearized BasicBlock-s
        self.prepost_bb_mapping = {}

    def has_memoization(self, node):
        return node in self.memoization_map

    def get_memoization(self, node):
        return self.memoization_map[node]

    def add_memoization(self, node, node_reg):
        if isinstance(node_reg, MachineRegister):
            self.node_to_register_map[node] = node_reg
        self.memoization_map[node] = node_reg

    def get_new_register(self, register_format, var_tag=None):
        self.register_id += 1
        if var_tag is None:
            reg_tag = "reg-{}".format(self.register_id)
        else:
            reg_tag = "reg-{}".format(var_tag)
        return MachineRegister(self.register_id, register_format, reg_tag, var_tag=var_tag)

    def get_var_register(self, var_node):
        if not var_node in self.var_register_map:
            self.var_register_map[var_node] = self.get_new_register(var_node.get_precision(), var_tag=var_node.get_tag())
        return self.var_register_map[var_node]

    def get_reg_from_node(self, node):
        # TODO/FIXME: support case where node is expanded into multiple registers
        return self.node_to_register_map[node]

    def get_post_linearization_bb(self, pre_linearization_bb):
        """ Return the post-linerazation basick-block which corresponds
            to the linearization of <pre_linearization_bb>, creates it
            if it does not exist yet """
        if not pre_linearization_bb in self.prepost_bb_mapping:
            self.set_post_linearization_bb(pre_linearization_bb, BasicBlock(tag=pre_linearization_bb.get_tag() + "_lin"))
        return self.prepost_bb_mapping[pre_linearization_bb]

    def set_post_linearization_bb(self, pre_linearization_bb, post_linearization_bb):
        self.prepost_bb_mapping[pre_linearization_bb] = post_linearization_bb

    # force_variable_storing is not supported
    def linearize_graph(self, node, current_bb=None):
        """ linearize operation graph node

            :params node: operation node to be linearized
            :params current_bb: current BasicBlock where node should be inserted (if any)
        """

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
            # if BasicBlock node has not been encountered yet, this will create it
            linearized_bb = self.get_post_linearization_bb(node)
            for op in node.inputs:
                self.linearize_graph(op, linearized_bb)
            result = linearized_bb

        elif isinstance(node, ConditionalBranch):
            cond = node.get_input(0)
            if_bb = self.get_post_linearization_bb(node.get_input(1))
            else_bb = self.get_post_linearization_bb(node.get_input(2))

            cond_value = self.linearize_graph(cond, current_bb)
            new_node = ConditionalBranch(cond_value, if_bb)
            current_bb.add(new_node)
            current_bb.add(UnconditionalBranch(else_bb))

            result = new_node

        elif isinstance(node, UnconditionalBranch):
            # unconditionnal branch are not modified
            new_node = UnconditionalBranch(self.get_post_linearization_bb(node.get_input(0)))
            result = new_node
            current_bb.add(new_node)

        elif isinstance(node, BasicBlockList):
            new_bb_list = MachineProgram()
            for bb in node.inputs:
                linearized_bb = self.linearize_graph(bb)
                new_bb_list.add(linearized_bb)
            result = new_bb_list

        elif isinstance(node, (Statement, ConditionBlock, Loop)):
            Log.report(Log.Error, "{} are not supported in MI graph linearization "
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", node.__class__, node)

        elif isinstance(node, PhiNode):
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

            result_value = self.linearize_graph(result_value, current_bb)
            if is_constant(result_value):
                result_assign = RegisterAssign(var_register, MaterializeConstant(result_value, precision=result_value.get_precision()))
            else:
                result_assign = RegisterAssign(var_register, RegisterCopy(result_value, precision=result_value.get_precision()))
            # expression's result is the output register
            result = var_register

            # adding RegisterAssign to current basic-block
            current_bb.add(result_assign)

        elif isinstance(node, Return):
            # TODO/FIXME: must implement processor ABI
            if len(node.inputs) == 1:
                self.linearize_graph(node.get_input(0), current_bb)
            elif len(node.inputs) == 0:
                assert node.get_precision() == None
                pass
            else:
                # only supports Return with 0 or one argument
                Log.report(Log.Error, "linearize_graph only supports Return with 0 or 1 args, not {}", node, error=NotImplementedError)
            result = Return(precision=ML_Void)
            current_bb.add(result)

        elif is_leaf_node(node):
            result = self.linearize_graph(node, current_bb)

        elif isinstance(node, TableStore):
            # TODO/FIXME: may need the generation of a shaddow dependency chain
            # to maintain TableStore/TableLoad relative order when required
            # TableStore are void operation (result is not stored into a machine register)
            op_regs = {op: self.linearize_graph(op, current_bb) for op in node.inputs}
            linearized_node = node.copy(copy_map=op_regs)
            node.finish_copy(linearized_node)
            current_bb.add(linearized_node)
            # TableStore does not have a result
            result = None

        elif isinstance(node, ML_ArithmeticOperation):
            op_regs = {op: self.linearize_graph(op, current_bb) for op in node.inputs}
            result_reg = self.get_new_register(node.get_precision())

            linearized_node = node.copy(copy_map=op_regs)
            #__class__(*tuple(op_regs), precision=node.get_precision())
            node.finish_copy(linearized_node)

            result_assign = RegisterAssign(
                result_reg,
                linearized_node,
                precision=ML_Void
            )
            # adding RegisterAssign to current basic-block
            current_bb.add(result_assign)
            # expression's result is the output register
            result = result_reg

        else:
            Log.report(Log.Error, "node unsupported in linearize_graph: {}", node)


        # registering result into memoization table
        self.add_memoization(node, result)

        return result



