# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Feb  3rd, 2016
# last-modified:    Feb  5th, 2016
#
# author(s):     Nicolas Brunie (nicolas.brunie@kalray.eu)
# desciprition:  Static Vectorizer implementation for Metalibm
###############################################################################

from .ml_formats import *
from .ml_operations import *


## 
class StaticVectorizer(object):
  ## initialize static vectorizer object
  #  @param OptimizationEngine object
  def __init__(self, opt_engine):
    self.opt_engine = opt_engine


  ## optree static vectorization 
  #  @param optree ML_Operation object, root of the DAG to be vectorized
  #  @param arg_list list of ML_Operation objects used as arguments by optree
  #  @param vector_size integer size of the vectors to be generated
  #  @param call_externalizer function to handle call_externalization process
  #  @param output_precision scalar precision to be used in scalar callback
  #  @return paire ML_Operation, CodeFunction of vectorized scheme and scalar callback 
  def vectorize_scheme(self, optree, arg_list, vector_size, call_externalizer, output_precision, sub_vector_size = None):
    # defaulting sub_vector_size to vector_size  when undefined
    sub_vector_size = vector_size if sub_vector_size is None else sub_vector_size

    def fallback_policy(cond, cond_block, if_branch, else_branch):
      return if_branch, [cond]
    def and_merge_conditions(condition_list, bool_precision = ML_Bool):
      assert(len(condition_list) >= 1)
      if len(condition_list) == 1:
        return condition_list[0]
      else:
        half_size = len(condition_list) / 2
        first_half  = and_merge_conditions(condition_list[:half_size])
        second_half = and_merge_conditions(condition_list[half_size:])
        return LogicalAnd(first_half, second_half, precision = bool_precision)

    # instanciate intermediary variable according
    # to the association indicated by variable_mapping
    # @param optree ML_Operation object root of the input operation graph
    # @param variable_mapping dict ML_Operation -> ML_Operation 
    #        mapping a variable to its sub-graph
    # @return an updated version of optree with variables replaced
    #         by the corresponding sub-graph if any
    def instanciate_variable(optree, variable_mapping):
      if isinstance(optree, Variable) and optree in variable_mapping:
        return variable_mapping[optree]
      elif isinstance(optree, ML_LeafNode):
        return optree
      else:
        for index, op_in in enumerate(optree.get_inputs()):
          optree.set_input(index, instanciate_variable(op_in, variable_mapping))
        return optree

    vectorized_path = self.opt_engine.extract_vectorizable_path(optree, fallback_policy)
    linearized_most_likely_path = vectorized_path.linearized_optree
    validity_list = vectorized_path.validity_mask_list
    # replacing temporary variables by their latest assigned values
    linearized_most_likely_path = instanciate_variable(linearized_most_likely_path, vectorized_path.variable_mapping)

    vector_paths    = []
    vector_masks    = []
    vector_arg_list = []

    # Assembling a vector from sub-vectors, simplify to identity
    # if there is only ONE sub-vector
    def assembling_vector(args, precision = None, tag = None):
      if len(args) == 1:
        return args[0]
      else:
        return VectorAssembling(*args, precision = precision, tag = tag)

    def extract_const(in_dict):
      result_dict = {}
      for keys, values in in_dict.items():
        if isinstance(keys, Constant):
          result_dict.update({keys:values})
      return result_dict
 
    vec_arg_dict = dict((arg_node, Variable("vec_%s" % arg_node.get_tag(), precision = self.vectorize_format(arg_node.get_precision(), vector_size))) for arg_node in arg_list)
    constant_dict = {}

    for i in xrange(vector_size / sub_vector_size):
      if sub_vector_size == vector_size :
        arg_list_copy = dict((arg_node, Variable("vec_%s" % arg_node.get_tag() , precision = arg_node.get_precision())) for arg_node in arg_list)
        sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
        vectorization_map = {}
      else :
        arg_list_copy = dict(
          (arg_node, assembling_vector(tuple((VectorElementSelection(vec_arg_dict[arg_node], i*sub_vector_size + j, precision = arg_node.get_precision())) for j in range(sub_vector_size)), precision = self.vectorize_format(arg_node.get_precision(), sub_vector_size), tag = "%s%d" %(vec_arg_dict[arg_node].get_tag(), i)))
          for arg_node in arg_list)
        sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
        vectorization_map = dict((arg, arg) for arg in sub_vec_arg_list)

      if (i > 0):
        vect_const_dict = dict((constant_dict[key], constant_dict[key]) for key in constant_dict.keys())
        vectorization_map.update(vect_const_dict)

      vector_arg_list.append(sub_vec_arg_list)
      arg_list_copy.update(constant_dict)
      sub_vector_path = linearized_most_likely_path.copy(arg_list_copy)
      
      # print sub_vector_path.get_str(depth = None, display_precision = True, memoization_map = {}, display_id = True)
      sub_vector_path = self.vector_replicate_scheme_in_place(sub_vector_path, sub_vector_size, vectorization_map)
      vector_paths.append(sub_vector_path)

      # no validity condition for vectorization (always valid)
      if len(validity_list) == 0:
        Log.report(Log.Info, "empty validity list encountered during vectorization")
        sub_vector_mask = Constant(True, precision = ML_Bool) 
      else:
        sub_vector_mask = and_merge_conditions(validity_list).copy(arg_list_copy)

      sub_vector_mask = self.vector_replicate_scheme_in_place(sub_vector_mask, sub_vector_size, vectorization_map)
      vector_masks.append(sub_vector_mask)

      if i == 0:
        constant_dict = extract_const(arg_list_copy)
        
    vector_path = assembling_vector(tuple(vector_paths), precision = self.vectorize_format(linearized_most_likely_path.get_precision(), vector_size))
    vec_arg_list = [vec_arg_dict[arg_node] for arg_node in arg_list]
    vector_mask = assembling_vector(tuple(vector_masks), precision = self.vectorize_format(ML_Bool, vector_size))
    return vec_arg_list, vector_path, vector_mask


  def vectorize_format(self, scalar_format, vector_size):
    return {
      ML_Binary32: {
        2: v2float32,
        3: v3float32,
        4: v4float32,
        8: v8float32
      },
      ML_Binary64: {
        2: v2float64,
        3: v3float64,
        4: v4float64,
        8: v8float64
      },
      ML_UInt32: {
        2: v2uint32,
        3: v3uint32,
        4: v4uint32,
        8: v8uint32
      },
      ML_Int32: {
        2: v2int32,
        3: v3int32,
        4: v4int32,
        8: v8int32
      },
      ML_UInt64: {
        2: v2uint64,
        3: v3uint64,
        4: v4uint64,
        8: v8uint64
      },
      ML_Int64: {
        2: v2int64,
        3: v3int64,
        4: v4int64,
        8: v8int64
      },
      ML_Bool: {
        2: v2bool,
        3: v3bool,
        4: v4bool,
        8: v8bool
      },
    }[scalar_format][vector_size]

  def is_vectorizable(self, optree):
    arith_flag = isinstance(optree, ML_ArithmeticOperation)
    cst_flag   = isinstance(optree, Constant) 
    var_flag   = isinstance(optree, Variable)
    if arith_flag or cst_flag or var_flag:
      return True
    elif isinstance(optree, SpecificOperation) and optree.get_specifier() in [SpecificOperation.DivisionSeed, SpecificOperation.InverseSquareRootSeed]:
      return True
    return False

  def vector_replicate_scheme_in_place(self, optree, vector_size, _memoization_map = None):
    memoization_map = {} if _memoization_map is None else _memoization_map

    if optree in memoization_map:
      return memoization_map[optree]
    else:
      if self.is_vectorizable(optree):
        optree_precision = optree.get_precision()
        if optree_precision is None:
          print optree.get_str(display_precision = True, memoization_map = {})
        optree.set_precision(self.vectorize_format(optree.get_precision(), vector_size))
        if isinstance(optree, Constant):
          optree.set_value([optree.get_value() for i in xrange(vector_size)])
        elif isinstance(optree, Variable):
          # TODO: does not take into account intermediary variables
          pass
        elif not isinstance(optree, ML_LeafNode):
          for optree_input in optree.get_inputs():
            self.vector_replicate_scheme_in_place(optree_input, vector_size, memoization_map)
        memoization_map[optree] = optree
        return optree
      else:
        Log.report(Log.Info, "optree not vectorizable: {}".format(optree.get_str(display_precision = True)))

    
