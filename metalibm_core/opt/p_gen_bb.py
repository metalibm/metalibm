# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

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

from metalibm_core.core.ml_operations import (
    Statement, ConditionBlock, Return, Loop
)

from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO
from metalibm_core.core.bb_operations import (
    ConditionalBranch, UnconditionalBranch, BasicBlock
)

from metalibm_core.utility.log_report import Log


class Pass_GenerateBasicBlock(FunctionPass):
    """ pre-Linearize operation tree to basic blocks
        Control flow construction are transformed into linked basic blocks
        Dataflow structure of the operation graph is kept for unambiguous
        construct (basically non-control flow nodes) """
    pass_tag = "gen_basic_block"
    def __init__(self, target, description = "generate basic-blocks pass"):
        FunctionPass.__init__(self, description, target)
        self.memoization_map = {}
        self.bb_list = []
        self.top_bb = BasicBlock(tag="main")
        self.current_bb_stack = [self.top_bb]

    def set_top_bb(self, bb):
        """ define the top basic block and reset the basic block stack so it
            only contains this block """
        self.top_bb = bb
        self.current_bb_stack = [self.top_bb]
        return self.top_bb

    def push_to_current_bb(self, node):
        """ add a new node at the end of the current basic block
            which is the topmost on the BB stack """
        assert len(self.current_bb_stack) > 0
        self.current_bb_stack[-1].push(node)

    def push_new_bb(self):
        self.current_bb_stack.append(BasicBlock(tag="new"))

    def add_to_bb(self, bb, node):
        if not bb.final:
            bb.add(node)

    def pop_current_bb(self):
        """ remove the topmost basic block from the BB stack
            and add it to the list of basic blocks """
        if len(self.current_bb_stack) >= 1:
            self.bb_list.append(self.current_bb_stack.pop(-1))

    def get_current_bb(self):
        return self.current_bb_stack[-1]

    ## Recursively traverse operation graph from @p optree
    #  to check that every node has a defined precision
    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        """ return the head basic-block, i.e. the entry BB for the current node
            implementation """
        entry_bb =  self.get_current_bb()
        if isinstance(optree, ConditionBlock):
            # When encoutering
            # if <cond>:
            #       if_branch
            # else:
            #       else_branch
            #
            #   BB0:
            #      cb <cond>, if_block
            #   if_block:
            #       if_branch
            #   else_block:     (if any)
            #       else_branch
            #   fallback:
            #       ...
            cond = optree.get_input(0)
            if_branch = optree.get_input(1)
            if optree.get_input_num() > 2:
                else_branch = optree.get_input(2)
            else:
                else_branch = None
            # create new basic block to generate if branch
            self.push_new_bb()
            if_bb = self.execute_on_optree(if_branch, fct, fct_group, memoization_map)

            self.pop_current_bb()
            self.push_new_bb()
            if not else_branch is None:
                print "ELSE BRANCH"
                else_bb = self.execute_on_optree(else_branch, fct, fct_group, memoization_map)
                self.pop_current_bb()
                self.push_new_bb()
                next_bb = self.get_current_bb()
                self.add_to_bb(else_bb, UnconditionalBranch(next_bb))
            else:
                else_bb = self.get_current_bb()
                
            next_bb = self.get_current_bb()
            # adding end of block instructions
            cb = ConditionalBranch(cond, if_bb, else_bb)
            self.add_to_bb(entry_bb, cb)
            # adding end of if block
            self.add_to_bb(if_bb, UnconditionalBranch(next_bb))
        elif isinstance(optree, Loop):
            pass
        elif isinstance(optree, Return):
            # Return must be processed separately as it finishes a basic block
            self.push_to_current_bb(optree)
            self.get_current_bb().final = True

        elif isinstance(optree, Statement):
            for op in optree.get_inputs():
                self.execute_on_optree(op, fct, fct_group, memoization_map)
        else:
            self.push_to_current_bb(optree)
        return entry_bb


    def execute_on_function(self, fct, fct_group):
        """ """
        Log.report(Log.Info, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        optree = fct.get_scheme()
        memoization_map = {}
        top_bb = self.set_top_bb(BasicBlock(tag="top"))
        last_bb = self.execute_on_optree(optree, fct, fct_group, memoization_map)
        fct.set_scheme(top_bb)



Log.report(LOG_PASS_INFO, "Registering generate Basic-Blocks pass")
Pass.register(Pass_GenerateBasicBlock)
