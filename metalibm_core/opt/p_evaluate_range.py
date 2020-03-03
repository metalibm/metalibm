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
# Description: evaluating node range accross the graph
###############################################################################

from metalibm_core.core.ml_operations import is_leaf_node
from metalibm_core.utility.log_report import Log
from metalibm_core.core.passes import FunctionPass, LOG_PASS_INFO, Pass
from metalibm_core.opt.opt_utils import evaluate_range

LOG_VERBOSE_EVALUATE_RANGE = Log.LogLevel("EvaluateRangeVerbose")

class Pass_EvaluateRange(FunctionPass):
    """ Verify that each node has a precision assigned to it """
    pass_tag = "evaluate_range"

    def __init__(self, target):
        super().__init__("evaluate_range", target)

    def evaluate_set_range(self, optree, memoization_map=None):
        """ check if all precision-instantiated operation are supported by the processor """
        # memoization map is used to store node's range/interval
        memoization_map = {} if memoization_map is None else memoization_map
        if  optree in memoization_map:
            return optree
        else:
            if not is_leaf_node(optree):
                for op in optree.inputs:
                    _ = self.evaluate_set_range(op, memoization_map=memoization_map)

            if optree.get_interval() is None:
                op_range = evaluate_range(optree, update_interval=True)
            else:
                op_range = optree.get_interval()
            if not op_range is None:
                Log.report(LOG_VERBOSE_EVALUATE_RANGE, "range for {} has been evaluated to {}", optree, op_range)
            # memoization
            memoization_map[optree] = op_range
            return optree

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        return self.evaluate_set_range(optree, memoization_map)



Log.report(LOG_PASS_INFO, "Registering evaluate_range pass")
# register pass
Pass.register(Pass_EvaluateRange)
