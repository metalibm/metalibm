# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.opt.p_check_support import Pass_CheckSupport


## Generic vector promotion pass
class Pass_Vector_Promotion(OptreeOptimization):
  pass_tag = "vector_promotion"
  ## Return the translation table of formats
  #  to be used for promotion
  def get_translation_table(self):
    raise NotImplementedError

  def __init__(self, target):
    OptreeOptimization.__init__(self, "vector promotion pass", target)
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
    # table precision are left unchanged
    if isinstance(precision, ML_TableFormat):
      return precision
    else:
      return self.get_translation_table()[precision]

  def is_convertible_format(self, precision):
    # Table format is always convertible
    if isinstance(precision, ML_TableFormat):
      return True
    else:
      return precision in self.get_translation_table()

  ## test wether optree's operation is supported on 
  #  promoted formats
  def does_target_support_promoted_op(self, optree):
    # check that the output format is supported
    if not self.is_convertible_format(optree.get_precision()):
      return False
    if isinstance(optree, ML_LeafNode) \
       or isinstance(optree, VectorElementSelection) \
       or isinstance(optree, FunctionCall):
      return False
    # check that the format of every input is supported 
    for arg in optree.get_inputs():
      if not self.is_convertible_format(arg.get_precision()):
        return False
    ## This local key getter modifies on the fly
    # the optree precision to promotion-based formats
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
      print "not supported in vector_promotion: ", optree.get_str(depth = 2, display_precision = True, memoization_map = {})
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
      if self.does_target_support_promoted_op(optree):
        new_optree = optree.copy(copy_map = self.copy_map)
        new_inputs = [self.get_converted_node(op) for op in new_optree.get_inputs()]
        new_optree.inputs = new_inputs
        new_optree.set_precision(self.get_conv_format(optree.get_precision()))
        self.conversion_map[optree] = new_optree
    

  ## Convert a graph of operation to exploit the promoted registers
  #  @param parent_converted indicates that the result must 
  #         be in promoted formats else it must be in input format
  #         In case of promoted-format, the return value may need to be 
  #         a conversion if the operation is not supported
  def promote_node(self, optree, parent_converted = False):
    #if (parent_converted, optree) in self.memoization_map:
    #  if parent_converted:
    #  else:
    #    return self.memoization_map[(parent_converted, optree)]
    #else:
    if 1:
      new_optree = optree.copy(copy_map = self.copy_map)
      if self.does_target_support_promoted_op(optree):

        new_inputs = [self.promote_node(op, parent_converted = True) for op in optree.get_inputs()]
        new_optree.inputs = new_inputs
        new_optree.set_precision(self.get_conv_format(optree.get_precision()))

        # must be converted back to initial format
        # before being returned
        if not parent_converted:
          new_optree = Conversion(new_optree, precision = optree.get_precision())

        return self.memoize(parent_converted, optree, new_optree)
      elif isinstance(optree, ML_NewTable):
        return self.memoize(parent_converted, optree, optree)
      elif isinstance(optree, ML_LeafNode):
        if parent_converted and optree.get_precision() in self.get_translation_table():
          new_optree = Conversion(optree, precision = self.get_conv_format(optree.get_precision()))
          return self.memoize(parent_converted, optree, new_optree)
        elif parent_converted:
          raise NotImplementedError
        else:
          return self.memoize(parent_converted, optree, optree)
      else:
        # new_optree = optree.copy(copy_map = self.copy_map)

        # propagate conversion to inputs
        new_inputs = [self.promote_node(op) for op in optree.get_inputs()]


        # register modified inputs
        new_optree.inputs = new_inputs

        if parent_converted and optree.get_precision() in self.get_translation_table():
          new_optree = Conversion(new_optree, precision = self.get_conv_format(optree.get_precision()))
          return new_optree
        elif parent_converted:
          print optree.get_precision()
          raise NotImplementedError
        return self.memoize(parent_converted, optree, new_optree)
          

  # standard Opt pass API
  def execute(self, optree):
    return self.promote_node(optree)


print "Registering vector_conversion pass"
# register pass
Pass.register(Pass_Vector_Promotion)
