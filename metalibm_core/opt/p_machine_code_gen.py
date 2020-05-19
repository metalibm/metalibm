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
# Desciption: set of optimization passes to:
#             - generate Machine Code level representation
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################


from metalibm_core.core.passes import FunctionPass, METALIBM_PASS_REGISTER
from metalibm_core.core.ml_operations import Return, is_leaf_node
from metalibm_core.core.bb_operations import UnconditionalBranch
from metalibm_core.core.machine_operations import MachineProgram

from metalibm_core.code_generation.machine_program_linearizer import MachineInsnGenerator
from metalibm_core.code_generation.asmde_translator import AssemblySynthesizer

from metalibm_core.utility.log_report import Log

# specific debug log levels for this file: (very) verbose and
# info(rmative)
LOG_MACHINE_CODE_VERBOSE = Log.LogLevel("MachineCodeVerbose")
LOG_MACHINE_CODE_INFO = Log.LogLevel("MachineCodeInfo")

def locate_return_values(op_graph):
    """ locate Return node in op_graph """
    return_values = set()
    processed = set()
    to_be_processed = [op_graph]
    while to_be_processed != []:
        node = to_be_processed.pop(0)
        processed.add(node)
        if isinstance(node, Return):
            return_values.add(node.get_input(0))
        if is_leaf_node(node):
            pass
        else:
            for op in node.inputs:
                if not op in processed:
                    to_be_processed.append(op)
    # TODO/FIXME: explicit conversion to list may not be required here
    return list(return_values)
            

@METALIBM_PASS_REGISTER
class Pass_LinearizeOperationGraph(FunctionPass):
    """ Linearize operation tree to basic blocks,
        ensuring a RegisterAssign is associated to each operation node
    """
    pass_tag = "linearize_op_graph"
    def __init__(self, target, description = "linearize operation graph pass"):
        FunctionPass.__init__(self, description, target)
        self.machine_insn_linearizer = MachineInsnGenerator(target)

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        """ return the head basic-block, i.e. the entry bb for the current node
            implementation """
        raise NotImplementedError

    def execute_on_graph(self, op_graph):
        """ program linearization on complete operation graph, generating
            a final BasicBlockList as result """
        linearized_program = self.machine_insn_linearizer.linearize_graph(op_graph)
        return linearized_program

    def execute_on_function(self, fct, fct_group):
        """ execute basic-block generation pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_MACHINE_CODE_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        # extracting operation graph
        op_graph = fct.get_scheme()
        # extracting graph outputs
        linearized_program = self.execute_on_graph(op_graph)
        # extracting graph inputs and setting program inputs
        input_reg_list = [self.machine_insn_linearizer.get_reg_from_node(arg) for arg in fct.arg_list]
        linearized_program.ordered_input_regs = input_reg_list
        # setting outputs
        retval_list = locate_return_values(op_graph)
        output_reg_list = [self.machine_insn_linearizer.get_reg_from_node(retval) for retval in retval_list]
        # TODO/FIXME should be fixed when ABI lowering is performed
        linearized_program.output_regs = output_reg_list 

        fct.set_scheme(linearized_program)


@METALIBM_PASS_REGISTER
class Pass_RegisterAllocation(FunctionPass):
    """ perform register allocation in Machine-level code """
    pass_tag = "register_allocation"
    def __init__(self, target, description="register allocation pass"):
        FunctionPass.__init__(self, description, target)
        self.asm_synthesizer = AssemblySynthesizer(target.architecture)

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        """ return the head basic-block, i.e. the entry bb for the current node
            implementation """
        raise NotImplementedError

    def execute_on_graph(self, linearized_program):
        """ BB generation on complete operation graph, generating
            a final BasicBlockList as result """
        # translating to asmde program and performing register allocation
        Log.report(LOG_MACHINE_CODE_VERBOSE, "performing register allocation\n"
            "  input reg list is {}\n"
            "  output_reg_list is {}",
            linearized_program.ordered_input_regs,
            linearized_program.output_regs)
        asmde_program = self.asm_synthesizer.translate_to_asmde_program(
            linearized_program, linearized_program.ordered_input_regs, linearized_program.output_regs)
        color_map = self.asm_synthesizer.perform_register_allocation(asmde_program)
        # instanciating physical register
        self.asm_synthesizer.transform_to_physical_reg(color_map, linearized_program)
        return linearized_program

    def execute_on_function(self, fct, fct_group):
        """ execute basic-block generation pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_MACHINE_CODE_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        linearized_program = fct.get_scheme()
        assert isinstance(linearized_program, MachineProgram)
        allocated_program = self.execute_on_graph(linearized_program)
        fct.set_scheme(allocated_program)

LOG_SIMPLIFY_BB_FALLBACK_INFO = Log.LogLevel("SimplifyBBFallbackInfo")

@METALIBM_PASS_REGISTER
class Pass_SimplifyBBFallback(FunctionPass):
    """ remove unecessary goto """
    pass_tag = "simplify_bb_fallback"
    def __init__(self, target, description="bb fallback simplification pass"):
        FunctionPass.__init__(self, description, target)

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        """ return the head basic-block, i.e. the entry bb for the current node
            implementation """
        raise NotImplementedError

    def execute_on_graph(self, linearized_program):
        """ BB generation on complete operation graph, generating
            a final BasicBlockList as result """
        # translating to asmde program and performing register allocation
        assert isinstance(linearized_program, MachineProgram)
        for index, bb in enumerate(linearized_program.inputs):
            if len(bb.inputs) <= 0:
                continue
            last_insn = bb.inputs[-1]
            if isinstance(last_insn, UnconditionalBranch) and index + 1 < len(linearized_program.inputs):
                dest_bb = last_insn.get_input(0)
                if dest_bb == linearized_program.inputs[index + 1]:
                    Log.report(LOG_SIMPLIFY_BB_FALLBACK_INFO, "removing last UnconditionalBranch from BB: {}", bb)
                    # removing last UnconditionalBranch has the fallback is enough
                    bb.inputs = bb.inputs[:-1]
        return linearized_program

    def execute_on_function(self, fct, fct_group):
        """ execute basic-block generation pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_MACHINE_CODE_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        linearized_program = fct.get_scheme()
        assert isinstance(linearized_program, MachineProgram)
        allocated_program = self.execute_on_graph(linearized_program)
        fct.set_scheme(allocated_program)
