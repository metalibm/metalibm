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
#
# Created:     June 13th, 2018
# Description: Specific operations to describe basic block constructs
#

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_operations import (
    AbstractOperation,
    Statement, ControlFlowOperation,
    GeneralArithmeticOperation,
)


class ConditionalBranch(ControlFlowOperation):
    """ branch <cond> <true_dest> <false_dest> """
    arity = 3
    name = "ConditionalBranch"
    @property
    def destination_list(self):
        """ return the list of BB targeted by the instruction """
        return [self.get_input(1), self.get_input(2)]
    def get_str(
            self, depth=2, display_precision=False,
            tab_level=0, memoization_map=None,
            display_attribute=False, display_id=False,
            custom_callback=lambda op: "",
            display_interval=None,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        new_depth = None
        if depth != None:
            if  depth < 0:
                return ""
        new_depth = (depth - 1) if depth != None else None

        tab_str = AbstractOperation.str_del * tab_level + custom_callback(self)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        desc_str = self.get_str_descriptor(display_precision, display_id, display_attribute, tab_level)
        memoization_map[self] = str_tag

        node_str = tab_str + "{name}{desc} -------> {tag}\n{args}".format(
            name=self.get_name(), 
            desc=desc_str,
            tag=str_tag,
            args="".join(
                inp.get_str(
                    op_depth, display_precision,
                    tab_level=tab_level + 1,
                    memoization_map=memoization_map,
                    display_attribute=display_attribute,
                    display_id=display_id,
                    custom_callback=custom_callback,
                    display_interval=display_interval,
                ) for inp, op_depth in zip(self.inputs, [new_depth, 0, 0]))
        )
        return node_str
class UnconditionalBranch(ControlFlowOperation):
    """ goto <dest> """
    arity = 1
    name = "UnconditionalBranch"
    @property
    def destination_list(self):
        """ return the list of BB targeted by the instruction """
        return [self.get_input(0)]
    def get_str(
            self, depth=2, display_precision=False,
            tab_level=0, memoization_map=None,
            display_attribute=False, display_id=False,
            custom_callback=lambda op: "",
            display_interval=None,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        new_depth = None
        if depth != None:
            if  depth < 0:
                return ""
        new_depth = (depth - 1) if depth != None else None

        tab_str = AbstractOperation.str_del * tab_level + custom_callback(self)
        if self in memoization_map:
            return tab_str + "%s\n" % memoization_map[self]
        str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        desc_str = self.get_str_descriptor(display_precision, display_id, display_attribute, tab_level)
        memoization_map[self] = str_tag

        node_str = tab_str + "{name}{desc} -------> {tag}\n{args}".format(
            name=self.get_name(), 
            desc=desc_str,
            tag=str_tag,
            args="".join(
                inp.get_str(
                    op_depth, display_precision,
                    tab_level=tab_level + 1,
                    memoization_map=memoization_map,
                    display_attribute=display_attribute,
                    display_id=display_id,
                    custom_callback=custom_callback,
                    display_interval=display_interval
                ) for inp, op_depth in zip(self.inputs, [0]))
        )
        return node_str
class BasicBlock(Statement):
    name = "BasicBlock"
    def __init__(self, *args, **kw):
        Statement.__init__(self, *args, **kw)
        # indicate that the current basic block is final (end with
        # a Return like statement)
        self.final = False
    def finish_copy(self, new_copy, copy_map = {}):
        """ Propagating final attribute during copy """
        new_copy.final = self.final

    @property
    def empty(self):
        """ predicate if BasicBlock has any instruction node """
        return len(self.inputs) == 0

    @property
    def successors(self):
        last_op = self.get_input(-1)
        if not isinstance(last_op, ControlFlowOperation):
            Log.report(Log.Info, "last operation of BB is not a ControlFlowOperation: BB is {}", self)
            return []
        else:
            return last_op.destination_list

    def get_str(
            self, depth=2, display_precision=False,
            tab_level=0, memoization_map=None,
            display_attribute=False, display_id=False,
            custom_callback=lambda op: "",
            display_interval=None,
        ):
        memoization_map = {} if memoization_map is None else memoization_map
        new_depth = None
        if depth != None:
            if  depth < 0:
                return ""
        new_depth = (depth - 1) if depth != None else None

        tab_str = AbstractOperation.str_del * tab_level + custom_callback(self)
        if self in memoization_map:
            #return tab_str + "%s\n" % memoization_map[self]
            str_tag = memoization_map[self]
        else:
            str_tag = self.get_tag() if self.get_tag() else ("tag_%d" % len(memoization_map))
        desc_str = self.get_str_descriptor(display_precision, display_id, display_attribute, tab_level)
        memoization_map[self] = str_tag

        node_str = tab_str + "{name}{desc} -------> {tag}\n{args}".format(
            name=self.get_name(),
            desc=desc_str,
            tag=str_tag,
            args="".join(
                inp.get_str(
                    new_depth, display_precision,
                    tab_level=tab_level + 1,
                    memoization_map=memoization_map,
                    display_attribute=display_attribute,
                    display_id=display_id,
                    custom_callback=custom_callback,
                    display_interval=display_interval
                ) for inp in self.inputs)
        )
        return node_str


class BasicBlockList(Statement):
    """ list of BasicBlock, first basic block should be the code entry point """
    name = "BasicBlockList"
    @property
    def entry_bb(self):
        """ Return the entry block """
        return self.inputs[0]
    @entry_bb.setter
    def entry_bb(self, value):
        self.push(value)


class PhiNode(GeneralArithmeticOperation):
    name = "PhiNode"
    """ Implement Phi-Node required in SSA form
        PhiNode([v<i>, bb<i>) returns v<i> if predecessor of current basic-block
        in execution flow was bb<i>. Each predecessor must appear in the list
        """
    def __init__(self, *args, **kw):
        GeneralArithmeticOperation.__init__(self, *args, **kw)
        self.arity = len(args) / 2

    def get_value(self, index):
        return self.get_input(2 * index)
    def get_predecessor(self, index):
        return self.get_input(2 * index +1)


