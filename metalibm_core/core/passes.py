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

import sys
from metalibm_core.utility.log_report import Log

""" custom warning log level for pass management """
LOG_PASS_INFO = Log.LogLevel("Info", "passes")

## Parent class for all pass dependency
class PassDependency:
    ## test if the  @p self dependency is resolved
    #  @p param pass_scheduler scheduler which requires dependency checking
    #  @return boolean True if dependency is resolved, False otherwise
    def is_dep_resolved(self, pass_scheduler):
        return True

class AfterPassByClass(PassDependency):
  def __init__(self, pass_class):
    self.pass_class = pass_class

  def is_dep_resolved(self, pass_scheduler):
    for pass_obj in pass_scheduler.get_executed_passes():
      if isinstance(pass_obj, self.pass_class):
        return True
    return False

class AfterPassById(PassDependency):
  def __init__(self, pass_id):
    self.pass_id = pass_id

  def is_dep_resolved(self, pass_scheduler):
    for pass_obj in pass_scheduler.get_executed_passes():
      if pass_obj.get_pass_id() == self.pass_id:
        return True
    return False

class CombineAnd(PassDependency):
  def __init__(self, lhs, rhs):
    self.ops = lhs, rhs

  def is_dep_resolved(self, pass_scheduler):
    lhs, rhs = self.ops
    return lhs.is_dep_resolved(pass_scheduler) and \
           rhs.is_dep_resolved(pass_scheduler)

class CombineOr(PassDependency):
  def __init__(self, lhs, rhs):
    self.ops = lhs, rhs

  def is_dep_resolved(self, pass_scheduler):
    lhs, rhs = self.ops
    return lhs.is_dep_resolved(pass_scheduler) or \
           rhs.is_dep_resolved(pass_scheduler)

class PassWrapper:
  def __init__(self, pass_object, dependency):
    self.pass_object = pass_object
    self.dependency  = dependency
  def get_dependency(self):
    return self.dependency
  def get_pass_object(self):
    return self.pass_object


## default execution pass function
def default_execute_pass(pass_scheduler, pass_object, inputs):
  return [pass_object.execute(pass_input) for pass_input in inputs]

class PassScheduler:
  class Start: 
    tag = "start"
  class Whenever:
    tag = "whenever"
  class BeforePipelining: 
    tag = "beforepipelining"
  class AfterPipelining: 
    tag = "afterpipelining"
  class AfterTargetCheck:
    tag = "aftertargetcheck"
  class JustBeforeCodeGen: 
    tag = "beforecodegen"
  class Typing:
    tag = "typing"
  class Optimization:
    tag = "optimization"

  @staticmethod
  def get_tag_class(tag):
    return {
      PassScheduler.Start.tag: PassScheduler.Start,
      PassScheduler.Whenever.tag: PassScheduler.Whenever,
      PassScheduler.JustBeforeCodeGen.tag: PassScheduler.JustBeforeCodeGen,
      PassScheduler.BeforePipelining.tag: PassScheduler.BeforePipelining,
      PassScheduler.AfterPipelining.tag: PassScheduler.AfterPipelining,
    }[tag]

  def __init__(self, pass_tag_list=None):
    pass_tag_list = pass_tag_list or [
        PassScheduler.Start, PassScheduler.Whenever, PassScheduler.JustBeforeCodeGen,
        PassScheduler.BeforePipelining, PassScheduler.AfterPipelining
    ]
    self.pass_map = {
      None: [], # should remain empty
    }
    for pass_tag in pass_tag_list:
        self.pass_map[pass_tag] = []
    self.executed_passes = []
    self.ready_passes    = []
    self.waiting_pass_wrappers  = []

  def register_pass(self, pass_object, pass_dep=PassDependency(), pass_slot=None):
    """ Register a new pass to be executed
        @return the pass id """
    Log.report(LOG_PASS_INFO,
        "PassScheduler: registering pass {} at {}".format(
            pass_object,
            pass_slot
        )
    )
    self.pass_map[pass_slot].append(PassWrapper(pass_object, pass_dep)) 
    return pass_object.get_pass_id()

  def get_executed_passes(self):
    return self.executed_passes

  def get_rdy_pass_list(self):
    annotated_list = [(pass_wrapper, pass_wrapper.get_dependency().is_dep_resolved(self)) for pass_wrapper in self.waiting_pass_wrappers ]
    self.ready_passes += [pass_wrapper.get_pass_object() for (pass_wrapper, rdy_flag) in annotated_list if rdy_flag]
    self.waiting_pass_wrappers = [pass_wrapper for (pass_wrapper, rdy_flag) in annotated_list if not rdy_flag]
    return self.ready_passes


  def update_rdy_pass_list(self):
    self.ready_passes = self.get_rdy_pass_list()
    return self.ready_passes

  def enqueue_slot_to_waiting(self, pass_slot = None):
    self.waiting_pass_wrappers += self.pass_map[pass_slot]
    self.pass_map[pass_slot] = []

  ## @param pass_slot, add all remaining passes supposed to 
  #  start after pass_slot to the waiting list
  #  than update the ready passe list and execute ready passes
  #  each updating @p pass_input in turn
  #  the final result is returned
  def execute_pass_list(self, pass_list, inputs, execution_function):
    inter_values = inputs
    for pass_object in pass_list:
      Log.report(LOG_PASS_INFO, "executing pass: {}", pass_object.pass_tag)
      inter_values = execution_function(self, pass_object, inputs)
    return inter_values

  def flush_rdy_pass_list(self):
    ready_passes = self.ready_passes
    self.ready_passes = []
    return ready_passes


  ## Execute all the ready passes from the given @p pass_slot
  #  @param execution_function takes a pass and inputs as arguments and return 
  #         the corresponding  outputs
  def get_full_execute_from_slot(
      self, 
      inputs, 
      pass_slot = None, 
      execution_function = default_execute_pass
    ):
    self.enqueue_slot_to_waiting(pass_slot = pass_slot)
    passes_to_execute = self.update_rdy_pass_list()
    intermediary_values = inputs
    while len(passes_to_execute) > 0:
      intermediary_values = self.execute_pass_list(
        passes_to_execute, 
        intermediary_values,
        execution_function
      )
      self.executed_passes += passes_to_execute
      self.flush_rdy_pass_list()
      passes_to_execute = self.update_rdy_pass_list()
    return intermediary_values


## System to manage dynamically defined optimization pass
class Pass:
  pass_tag = None
  ## map of all registered pass
  pass_map = {}
  pass_id_iterator = -1

  @staticmethod
  def get_new_pass_id():
    Pass.pass_id_iterator += 1
    return Pass.pass_id_iterator

  ## instance a new pass object and allocate it a
  #  unique pass identifier
  def __init__(self):
    self.pass_id = Pass.get_new_pass_id()

  def get_pass_id(self):
    return self.pass_id

  @staticmethod
  def register(pass_class):
    tag = pass_class.pass_tag
    if not tag in Pass.pass_map:
      Log.report(LOG_PASS_INFO, "registering pass {} associated to tag {}".format(pass_class, tag))
      Pass.pass_map[tag] = pass_class
    else:
      Log.report(Log.Error, "a pass with name {} has already been registered while trying to register {}".format(tag, pass_class))

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
class OptimizationPass(Pass):
  """ Virtual parent to all optjmization pass """
  def __init__(self, descriptor = ""):
    Pass.__init__(self)
    self.descriptor = descriptor

  def set_descriptor(self, descriptor):
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


class FunctionPass(OptreeOptimization):
    """ pass which execute on functions node:
        (ML_Operation, CodeFunction or FunctionGroup) """
    def __init__(self, descriptor="", target=None):
        OptreeOptimization.__init__(self, descriptor, target)

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        raise NotImplementedError

    def execute_on_function(self, fct, fct_group):
        Log.report(Log.Info, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        optree = fct.get_scheme()
        memoization_map = {}
        new_scheme = self.execute_on_optree(optree, fct, fct_group, memoization_map)
        if not new_scheme is None:
            fct.set_scheme(new_scheme)

    def execute_on_fct_group(self, fct_group):
        Log.report(Log.Info, "executing pass {} on fct group {}".format(self.pass_tag, fct_group))
        def local_fct_apply(group, fct):
            return self.execute_on_function(fct, group)
        return fct_group.apply_to_all_functions(local_fct_apply)


class PassQuit(FunctionPass):
    pass_tag = "quit"
    def __init__(self, *args):
        OptimizationPass.__init__(self, "quit")

    def execute(self, *args):
        sys.exit(1)

    def execute_on_function(self, fct, fct_group):
        Log.report(Log.Info, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        self.execute(self, fct, fct_group)

    def execute_on_fct_group(self, fct_group):
        Log.report(Log.Info, "executing pass {} on fct group {}".format(self.pass_tag, fct_group))
        self.execute(self, fct_group)

class PassDump(FunctionPass):
  pass_tag = "dump"
  def __init__(self, *args):
    OptimizationPass.__init__(self, "dump")

  def execute(self, optree):
    Log.report(Log.Info, "executing PassDump")
    print(optree.get_str(
        depth = None, display_precision = True, memoization_map = {}
    ))

  def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
    return self.execute(optree)

class PassDumpWithStages(OptreeOptimization):
  pass_tag = "dump_with_stages"
  def __init__(self, *args):
    OptimizationPass.__init__(self, "dump_with_stages")

  def execute(self, optree):
    Log.report(Log.Info, "executing PassDumpWithStages")
    print(optree.get_str(
        depth = None, display_precision = True, memoization_map = {},
        custom_callback = lambda op: " [S={}] ".format(op.attributes.init_stage)
    ))

Log.report(LOG_PASS_INFO, "registerting basic passes (Quit,Dump,DumpWithStages)")
# registering commidity pass
Pass.register(PassQuit)
Pass.register(PassDump)
Pass.register(PassDumpWithStages)
