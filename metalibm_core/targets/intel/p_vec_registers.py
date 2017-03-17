# -*- coding: utf-8 -*-
# optimization pass to promote a scalar DAG into vector registers

from .x86_processor import *
from metalibm_core.core.ml_formats import *

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

## _m128 register promotion
class Pass_M128_Promotion(OptreeOptimization):
  ## Translation table between standard formats
  #  and __m128-based register formats
  trans_table = {
    ML_Binary32: ML_SSE_m128_v1float32,
    ML_Binary64: ML_SSE_m128_v1float64,
    v2float64:   ML_SSE_m128_v2float64,
    v4float32:   ML_SSE_m128_v4float32
  }

  def __init__(self, target):
    OptreeOptimization.__init__(self, "__m128 promotion pass", target)
    ## memoization map for promotion optree
    self.memoization_map = {}
    # memoization map for copy
    self.copy_map = {}

  ## Evaluate the latency of a converted operation
  #  graph to determine whether the conversion
  #  is worth it
  def evaluate_converted_graph_cost(self, optree):
    pass

  def get_conv_format(self, precision):
    return self.trans_table[precision]

  ## test wether optree's operation is supported on 
  #  __m128-based formats
  def does_m128_support_operation(self, optree):
    # check that the output format is supported
    if not optree.get_precision() in self.trans_table:
      return False
    if isinstance(optree, ML_LeafNode):
      return False
    # check that the format of every input is supported 
    for arg in optree.get_inputs():
      if not arg.get_precision() in self.trans_table:
        return False
    ## This local ket getter modifies on the fly
    # the optree precision to __m128-based formats
    # to determine if the converted node is supported
    def key_getter(target_obj, optree):
      op_class = optree.__class__
      result_type = (self.get_conv_format(optree.get_precision().get_match_format()),)
      arg_type = tuple((self.get_conv_format(arg.get_precision().get_match_format()) if not arg.get_precision() is None else None) for arg in optree.inputs)
      interface = result_type + arg_type
      codegen_key = optree.get_codegen_key()
      return op_class, interface, codegen_key

    return self.target.is_supported_operation(optree, key_getter = key_getter)

  def memoize(self, force, optree, new_optree):
    self.memoization_map[(force, optree)] = new_optree
    return new_optree

  ## Convert a graph of operation to exploit
  #  the _m128 registers
  #  @param force indicate that the result must be __m128 formats 
  #   possibly through a conversion if the operation is not supported
  def convert_node_graph_to_m128(self, optree, force = False):
    if (force, optree) in self.memoization_map:
      return self.memoization_map[(force, optree)]
    else:
      if self.does_m128_support_operation(optree):
        new_optree = optree.copy(copy_map = self.copy_map)

        new_inputs = [self.convert_node_graph_to_m128(op, force = True) for op in new_optree.get_inputs()]
        new_optree.inputs = new_inputs
        new_optree.set_precision(self.get_conv_format(optree.get_precision()))

        ## must be converted back to initial format
        if not force:
          new_optree = Conversion(new_optree, precision = optree.get_precision())

        return self.memoize(force, optree, new_optree)
      elif isinstance(optree, ML_LeafNode):
        if force and optree.get_precision() in self.trans_table:
          new_optree = Conversion(optree, precision = self.get_conv_format(optree.get_precision()))
          return self.memoize(force, optree, new_optree)
        else:
          return self.memoize(force, optree, optree)
      else:
        new_optree = optree.copy(copy_map = self.copy_map)

        # propagate conversion to inputs
        new_inputs = [self.convert_node_graph_to_m128(op) for op in new_optree.get_inputs()]
        # register modified inputs
        new_optree.inputs = new_inputs

        if force and optree.get_precision() in self.trans_table:
          new_optree = Conversion(new_optree, precision = self.get_conv_format(optree.get_precision()))
        return self.memoize(force, optree, new_optree)
          

  # standard Opt pass API
  def execute(self, optree):
    return self.convert_node_graph_to_m128(optree)


 
