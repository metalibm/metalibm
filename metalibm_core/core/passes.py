# -*- coding: utf-8 -*-

## System to manage dynamically defined optimization pass
class Pass:
  pass_tag = None
  ## map of all registered pass
  pass_map = {}
  @staticmethod
  def register(pass_class):
    tag = pass_class.pass_tag
    if not tag in Pass.pass_map:
      print "registering pass {} associated to tag {}".format(pass_class, tag)
      Pass.pass_map[tag] = pass_class

  ## return the pass class associated with name @p tag
  #  @param tag[str] pass name
  #  @return[Pass] pass object 
  @staticmethod
  def get_pass_by_tag(tag):
    return Pass.pass_map[tag]

  ## return the list of tags of registered passes
  @staticmethod
  def get_pass_tag_list():
    return [tag for tag in Pass.pass_map]


## Abstract parent to optimization pass
class OptimizationPass:
  """ Virtual parent to all optjmization pass """
  def __init__(self, descriptor = ""):
    self.descriptor = descriptor

  def get_descriptor(self):
    return self.descriptor

## Operation tree Optimization pass
class OptreeOptimization(OptimizationPass):
  def __init__(self, descriptor, target):
    OptimizationPass.__init__(self, descriptor)
    # Processor target
    self.target = target

  def get_target(self):
    return self.target

  ## main function to executon optree optimization pass
  # on the given operation sub-graph @p optree
  def execute(self, optree):
    raise NotImplemented
