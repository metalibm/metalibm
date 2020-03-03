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
import collections

from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import *
from metalibm_core.core.bb_operations import (
        UnconditionalBranch, ConditionalBranch, BasicBlock
)

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.log_report import Log


class Pass_TagNode(FunctionPass):
    """ Verify that each node has a precision assigned to it """
    pass_tag = "tag_node"

    def __init__(self, target, language=C_Code):
        FunctionPass.__init__(self, "tag_node", target)
        self.language = language
        self.tag_id = collections.defaultdict(lambda: 0)


    def get_new_tag(self, optree):
        self.tag_id[optree.__class__] += 1
        tag = "{}_{}".format(str(optree.__class__.name), self.tag_id[optree.__class__])
        return tag

    def tag_node(self, optree, memoization_map=None, debug=False, language=C_Code):
        """ provide a random tag to each node with no tags """
        memoization_map = {} if memoization_map is None else memoization_map
        if  optree in memoization_map:
            return None
        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs:
                self.tag_node(inp, memoization_map, debug=debug)

        if optree.get_tag() is None:
            optree.set_tag(self.get_new_tag(optree))

        # memoization
        memoization_map[optree] = True
        return None

    def execute_on_optree(self, optree, fct, fct_group, memoization_map):
        return self.tag_node(optree, memoization_map, language=self.language)



Log.report(LOG_PASS_INFO, "Registering tag_node pass")
# register pass
Pass.register(Pass_TagNode)
