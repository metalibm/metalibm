# -*- coding: utf-8 -*-
#
from metalibm_core.core.passes import OptreeOptimization, Pass
from metalibm_core.core.ml_operations import *


## Check support of operation graph on a given target
class Pass_CheckSupport(OptreeOptimization):
  pass_tag = "check_target_support"

  def __init__(self, target):
    OptreeOptimization.__init__(self, "check_target_support", target)

  ## Test if @p optree is supported by self.target
  #  @param optree operation tree to be tested
  #  @param memoization_map memoization map of parallel executions
  #  @param debug enable debug messages
  #  @return boolean support
  def check_processor_support(self, optree, memoization_map = {}, debug = False):
    """ check if all precision-instantiated operation are supported by the processor """
    if debug:
      print "checking processor support: ", self.get_target().__class__ # Debug print
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
      elif not self.get_target().is_supported_operation(optree, debug = debug):
        print self.processor.get_operation_keys(optree) # Error print
        print optree.get_str(display_precision = True, display_id = True, memoization_map = {}) # Error print
        Log.report(Log.Error, "unsupported operation\n")
    # memoization
    memoization_map[optree] = True
    return True

  def execute(self, optree):
    memoization_map = {}
    return self.check_processor_support(optree, memoization_map)



print "Registering check_target_support pass"
# register pass
Pass.register(Pass_CheckSupport)
