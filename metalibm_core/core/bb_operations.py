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


from metalibm_core.core.ml_operations import (
    Statement, ControlFlowOperation
)


class ConditionalBranch(ControlFlowOperation):
    """ branch <cond> <true_dest> <false_dest> """
    arity = 3
    name = "ConditionalBranch"
class UnconditionalBranch(ControlFlowOperation):
    """ goto <dest> """
    arity = 1
    name = "UnconditionalBranch"
class BasicBlock(Statement):
    name = "BasicBlock"
    def __init__(self, *args, **kw):
        Statement.__init__(self, *args, **kw)
        # indicate that the current basic block is final (end with
        # a Return like statement)
        self.final = False

