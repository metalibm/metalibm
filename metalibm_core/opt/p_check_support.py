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

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.log_report import Log


## Check support of operation graph on a given target
class Pass_CheckSupport(OptreeOptimization):
  pass_tag = "check_target_support"

  def __init__(self, target, language=C_Code):
    OptreeOptimization.__init__(self, "check_target_support", target)
    self.language = language

  ## Test if @p optree is supported by self.target
  #  @param optree operation tree to be tested
  #  @param memoization_map memoization map of parallel executions
  #  @param debug enable debug messages
  #  @return boolean support
  def check_processor_support(self, optree, memoization_map=None, debug=False, language=C_Code):
    """ check if all precision-instantiated operation are supported by the processor """
    memoization_map = {} if memoization_map is None else memoization_map
    if debug:
      print("checking processor support: ", self.get_target().__class__) # Debug print
    if  optree in memoization_map:
      return True
    if not isinstance(optree, ML_LeafNode):
      for inp in optree.inputs:
        self.check_processor_support(inp, memoization_map, debug = debug)

      if isinstance(optree, ConditionBlock):
        self.check_processor_support(optree.get_pre_statement(), memoization_map, debug = debug)
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
        #self.check_processor_support(optree.get_pre_statement(), memoization_map)

        for op in optree.get_extra_inputs():
          # TODO: assert case is integer constant
          self.check_processor_support(op, memoization_map, debug = debug)
      elif not self.get_target().is_supported_operation(optree, language=language, debug=debug):
        print(language, self.processor.get_operation_keys(optree)) # Error print
        print(optree.get_str(display_precision = True, display_id = True, memoization_map = {})) # Error print
        Log.report(Log.Error, "unsupported operation\n")
    # memoization
    memoization_map[optree] = True
    return True

  def execute(self, optree):
    memoization_map = {}
    return self.check_processor_support(optree, memoization_map, language=self.language)



Log.report(LOG_PASS_INFO, "Registering check_target_support pass")
# register pass
Pass.register(Pass_CheckSupport)
