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
from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import *
from metalibm_core.core.bb_operations import (
    UnconditionalBranch, ConditionalBranch, BasicBlock
)

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.log_report import Log

## Test if @p optree is supported by self.target
def check_target_support(target, optree, memoization_map=None, debug=False, language=C_Code):
    """ check if all precision-instantiated operation are supported by the processor <target>
    
        :param optree: operation tree to be tested
        :type optree: ML_Operation
        :param memoization_map: memoization map of parallel executions
        :type memoization_map: dict
        :param debug: enable debug messages
        :type debug: bool
        :rtype: bool
        :return: boolean support
    
    """
    memoization_map = {} if memoization_map is None else memoization_map
    if debug:
        print("checking processor support: ", target.__class__) # Debug print
    if  optree in memoization_map:
        return True
    if not isinstance(optree, ML_LeafNode):
        for inp in optree.inputs:
            check_target_support(target, inp, memoization_map, debug = debug)

        if isinstance(optree, ConditionBlock):
            check_target_support(target, optree.get_pre_statement(), memoization_map, debug = debug)
        elif isinstance(optree, ConditionalBranch):
            pass
        elif isinstance(optree, UnconditionalBranch):
            pass
        elif isinstance(optree, Statement):
            pass
        elif isinstance(optree, Loop):
            pass
        elif isinstance(optree, Return):
            pass
        elif isinstance(optree, ReferenceAssign):
            pass
        elif isinstance(optree, PlaceHolder):
            pass
        elif isinstance(optree, SwitchBlock):
            pass

            for op in optree.get_extra_inputs():
                # TODO: assert case is integer constant
                check_target_support(target, op, memoization_map, debug = debug)
        elif not target.is_supported_operation(optree, language=language, debug=debug):
            print("languages is {}".format(language))
            print("Operation' keys are: {}".format(target.get_operation_keys(optree))) # Error print
            print("Operation tree is:")
            print(optree.get_str(display_precision = True, display_id = True, memoization_map = {})) # Error print
            Log.report(Log.Error, "unsupported operation in Pass_CheckSupport's check_processor_support {}:\n{}", target, optree)

    # memoization
    memoization_map[optree] = True
    return True

class Pass_CheckSupport(OptreeOptimization):
  """ Verify that each node has a precision assigned to it """
  pass_tag = "check_target_support"

  def __init__(self, target, language=C_Code):
    OptreeOptimization.__init__(self, "check_target_support", target)
    self.language = language


  def execute(self, optree):
    memoization_map = {}
    return check_target_support(self.processor, optree, memoization_map, language=self.language)


Log.report(LOG_PASS_INFO, "Registering check_target_support pass")
# register pass
Pass.register(Pass_CheckSupport)
