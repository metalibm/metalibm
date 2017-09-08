# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import (
    ML_LeafNode, Statement, ConditionBlock, ReferenceAssign
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Loop, ComponentInstance, Assert, Wait
)
from metalibm_core.core.passes import OptreeOptimization, Pass


## Check if @p optree has a valid precision
def check_precision_validity(optree):
  none_class_list = [
    Statement, ConditionBlock, Process,
    ReferenceAssign, Loop, ComponentInstance,
    Assert, Wait,
  ]
  if reduce(lambda x, y: x or y, [isinstance(optree, none_class) for none_class in none_class_list], False):
    return optree.get_precision() is None
  else:
    return not optree.get_precision() is None

## Generic vector promotion pass
class Pass_CheckGeneric(OptreeOptimization):
  pass_tag = "check_generic"
  def __init__(self, target, check_function = lambda optree: True, description = "check_generic pass"):
    OptreeOptimization.__init__(self, description, target)
    self.memoization_map = {}
    self.check_function = check_function

  ## Recursively traverse operation graph from @p optree
  #  to check that every node has a defined precision
  def execute(self, optree):
    if optree in self.memoization_map: 
      return self.memoization_map[optree]
    else:
      check_result = self.check_function(optree)
      self.memoization_map[optree] = check_result
      if not check_result:
        Log.report(Log.Info, 
          "the following node check failed: {}".format(
            optree.get_str(
              depth = 2, 
              display_precision = True, 
              memoization_map = {}
            )
          )
        )
      if not isinstance(optree, ML_LeafNode):
        for op_input in optree.get_inputs():
          check_result &= self.execute(op_input)
      return check_result



## Generic vector promotion pass
class Pass_CheckPrecision(Pass_CheckGeneric):
  pass_tag = "check_precision"
  def __init__(self, target):
    Pass_CheckGeneric.__init__(self, target, check_precision_validity, "check_precision pass")
    self.memoization_map = {}


# register pass
print "Registering check_generic pass"
Pass.register(Pass_CheckGeneric)

print "Registering check_precision pass"
Pass.register(Pass_CheckPrecision)
