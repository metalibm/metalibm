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
  def vectorize_scheme(self, optree, arg_list, vector_size, call_externalizer, output_precision):
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

    linearized_most_likely_path, validity_list = self.opt_engine.extract_vectorizable_path(optree, fallback_policy)

    arg_list_copy = dict((arg_node, Variable("vec_%s" % arg_node.get_tag(), precision = arg_node.get_precision())) for arg_node in arg_list)
    vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]

    vector_path = linearized_most_likely_path.copy(arg_list_copy)
    vector_mask = and_merge_conditions(validity_list).copy(arg_list_copy)

    vectorization_map = {}
    vector_path = self.vector_replicate_scheme_in_place(vector_path, vector_size, vectorization_map)
    vector_mask = self.vector_replicate_scheme_in_place(vector_mask, vector_size, vectorization_map)

    return vec_arg_list, vector_path, vector_mask

    #assert(isinstance(linearized_most_likely_path, ML_ArithmeticOperation))
    #likely_result = linearized_most_likely_path

    #callback_function = call_externalizer.externalize_call(optree, arg_list, tag = "scalar_callback", result_format = output_precision)
    #callback = callback_function.get_function_object()(*arg_list)

    #vectorized_scheme = Statement(
    #  likely_result,
    #  ConditionBlock(and_merge_conditions(validity_list),
    #    Return(likely_result),
    #    Return(callback)
    #  )
    #)
    #return vectorized_scheme, callback_function

  def vectorize_format(self, scalar_format, vector_size):
    return {
      ML_Binary32: {
        2: ML_Float2,
        4: ML_Float4,
        8: ML_Float8
      },
      ML_Binary64: {
        2: ML_Double2,
        4: ML_Double4,
        8: ML_Double8
      },
      ML_UInt32: {
        2: ML_UInt2,
        4: ML_UInt4,
        8: ML_UInt8
      },
      ML_Int32: {
        2: ML_Int2,
        4: ML_Int4,
        8: ML_Int8
      },
      ML_Bool: {
        2: ML_Bool2,
        4: ML_Bool4,
        8: ML_Bool8
      },
    }[scalar_format][vector_size]

  def is_vectorizable(self, optree):
    return isinstance(optree, ML_ArithmeticOperation) or isinstance(optree, Constant) or isinstance(optree, Variable)

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
          pass
        elif not isinstance(optree, ML_LeafNode):
          for optree_input in optree.get_inputs():
            self.vector_replicate_scheme_in_place(optree_input, vector_size, memoization_map)
        memoization_map[optree] = optree
        return optree
    
