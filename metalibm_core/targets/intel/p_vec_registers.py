# -*- coding: utf-8 -*-
# optimization pass to promote a scalar DAG into vector registers

from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass

from metalibm_core.opt.check_support import Pass_CheckSupport


## _m128 register promotion
class Pass_M128_Promotion(OptreeOptimization):
  pass_tag = "m128_promotion"
  ## Translation table between standard formats
  #  and __m128-based register formats
  trans_table = {
    ML_Binary32: ML_SSE_m128_v1float32,
    ML_Binary64: ML_SSE_m128_v1float64,
    v2float64:   ML_SSE_m128_v2float64,
    v4float32:   ML_SSE_m128_v4float32,
    v4int32:     ML_SSE_m128_v4int32,
    ML_Int32:    ML_SSE_m128_v1int32,
  }

  def __init__(self, target):
    OptreeOptimization.__init__(self, "__m128 promotion pass", target)
    ## memoization map for promoted optree
    self.memoization_map = {}
    ## memoization map for converted promoted optree
    self.conversion_map = {}
    # memoization map for copy
    self.copy_map = {}

    self.support_checker = Pass_CheckSupport(target) 

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

    support_status = self.target.is_supported_operation(optree, key_getter = key_getter)
    if not support_status:
      print "not supported in m128_promotion: ", optree.get_str(depth = 2, display_precision = True, memoization_map = {})
    return support_status

  ## memoize converted     
  def memoize(self, force, optree, new_optree):
    self.memoization_map[(force, optree)] = new_optree
    #if not isinstance(optree, ML_LeafNode) and not self.target.is_supported_operation(optree):
    #  print optree.get_str(display_precision = True, memoization_map = {})
    #  print new_optree.get_str(display_precision = True, memoization_map = {})
    #  #raise Exception()
    return new_optree

  ## memoize conversion    
  def memoize_conv(self, force, optree, new_optree):
    self.conversion_map[(force, optree)] = new_optree
    #if not isinstance(optree, ML_LeafNode) and not self.target.is_supported_operation(optree):
    #  raise Exception()
    return new_optree

  def get_converted_node(self, optree):
    if optree in self.conversion_map:
      return self.conversion_map[optree]
    else:
      if self.does_m128_support_operation(optree):
        new_optree = optree.copy(copy_map = self.copy_map)
        new_inputs = [self.get_converted_node(op) for op in new_optree.get_inputs()]
        new_optree.inputs = new_inputs
        new_optree.set_precision(self.get_conv_format(optree.get_precision()))
        self.conversion_map[optree] = new_optree
    

  ## Convert a graph of operation to exploit the _m128 registers
  #  @param parent_converted indicates that the result must 
  #         be in __m128 formats else it must be in input format
  #         In case of __m128, the return value may need to be 
  #         a conversion if the operation is not supported
  def convert_node_graph_to_m128(self, optree, parent_converted = False):
    #if (parent_converted, optree) in self.memoization_map:
    #  if parent_converted:
    #  else:
    #    return self.memoization_map[(parent_converted, optree)]
    #else:
    if 1:
      new_optree = optree.copy(copy_map = self.copy_map)
      if self.does_m128_support_operation(optree):

        new_inputs = [self.convert_node_graph_to_m128(op, parent_converted = True) for op in optree.get_inputs()]
        new_optree.inputs = new_inputs
        new_optree.set_precision(self.get_conv_format(optree.get_precision()))

        # must be converted back to initial format
        # before being returned
        if not parent_converted:
          new_optree = Conversion(new_optree, precision = optree.get_precision())

        return self.memoize(parent_converted, optree, new_optree)
      elif isinstance(optree, ML_LeafNode):
        if parent_converted and optree.get_precision() in self.trans_table:
          new_optree = Conversion(optree, precision = self.get_conv_format(optree.get_precision()))
          return self.memoize(parent_converted, optree, new_optree)
        elif parent_converted:
          raise NotImplementedError
        else:
          return self.memoize(parent_converted, optree, optree)
      else:
        # new_optree = optree.copy(copy_map = self.copy_map)

        # propagate conversion to inputs
        new_inputs = [self.convert_node_graph_to_m128(op) for op in optree.get_inputs()]


        # register modified inputs
        new_optree.inputs = new_inputs

        if parent_converted and optree.get_precision() in self.trans_table:
          new_optree = Conversion(new_optree, precision = self.get_conv_format(optree.get_precision()))
          return new_optree
        elif parent_converted:
          print optree.get_precision()
          raise NotImplementedError
        return self.memoize(parent_converted, optree, new_optree)
          

  # standard Opt pass API
  def execute(self, optree):
    return self.convert_node_graph_to_m128(optree)


print "Registering m128_conversion pass"
# register pass
Pass.register(Pass_M128_Promotion)
