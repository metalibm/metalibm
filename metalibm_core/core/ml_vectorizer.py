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
# created:                    Feb    3rd, 2016
# last-modified:        Mar    7th, 2018
#
# author(s):         Nicolas Brunie (nicolas.brunie@kalray.eu)
# desciprition:    Static Vectorizer implementation for Metalibm
###############################################################################

from .ml_formats import VECTOR_TYPE_MAP, ML_Bool
from .ml_operations import *
from metalibm_core.core.ml_table import ML_NewTable


##
class StaticVectorizer(object):
    """ Mapping of size, scalar format to vector format """
    ## initialize static vectorizer object
    #    @param OptimizationEngine object
    def __init__(self, opt_engine):
        self.opt_engine = opt_engine


    def vectorize_scheme(self, optree, arg_list, vector_size, call_externalizer,
                         output_precision, sub_vector_size=None):
        """ optree static vectorization
            @param optree ML_Operation object, root of the DAG to be vectorized
            @param arg_list list of ML_Operation objects used as arguments by
                   optree
            @param vector_size integer size of the vectors to be generated
            @param call_externalizer function to handle call_externalization
                   process
            @param output_precision scalar precision to be used in scalar
                   callback
            @return pair ML_Operation, CodeFunction of vectorized scheme and
                    scalar callback
        """
        # defaulting sub_vector_size to vector_size    when undefined
        sub_vector_size = vector_size if sub_vector_size is None else sub_vector_size

        def fallback_policy(cond, cond_block, if_branch, else_branch):
            return if_branch, [cond]
        def and_merge_conditions(condition_list, bool_precision = ML_Bool):
            assert(len(condition_list) >= 1)
            if len(condition_list) == 1:
                return condition_list[0]
            else:
                half_size = int(len(condition_list) / 2)
                first_half    = and_merge_conditions(condition_list[:half_size])
                second_half = and_merge_conditions(condition_list[half_size:])
                return LogicalAnd(first_half, second_half, precision = bool_precision)

        def instanciate_variable(optree, variable_mapping, processed_map=None):
            """ instanciate intermediary variable according
                to the association indicated by variable_mapping
                @param optree ML_Operation root of the input operation graph
                @param variable_mapping dict ML_Operation -> ML_Operation
                               mapping a variable to its sub-graph
                @param processed_map dictionnary of node -> mapping storing
                      already processed node
                @return an updated version of optree with variables replaced
                                by the corresponding sub-graph if any
            """
            processed_map = {} if processed_map is None else processed_map
            if optree in processed_map:
                return processed_map[optree]
            elif isinstance(optree, Variable) and optree in variable_mapping:
                processed_map[optree] = variable_mapping[optree]
                return variable_mapping[optree]
            elif isinstance(optree, ML_LeafNode):
                processed_map[optree] = optree
                return optree
            else:
                for index, op_in in enumerate(optree.get_inputs()):
                    optree.set_input(
                        index,
                        instanciate_variable(
                            op_in, variable_mapping, processed_map
                        )
                    )
                processed_map[optree] = optree
                return optree

        vectorized_path = self.opt_engine.extract_vectorizable_path(optree, fallback_policy)
        linearized_most_likely_path = vectorized_path.linearized_optree
        validity_list = vectorized_path.validity_mask_list
        # replacing temporary variables by their latest assigned values
        linearized_most_likely_path = instanciate_variable(linearized_most_likely_path, vectorized_path.variable_mapping)

        vector_paths        = []
        vector_masks        = []
        vector_arg_list = []

        def assembling_vector(args, precision = None, tag = None):
            """ Assembling a vector from sub-vectors, simplify to identity
                if there is only ONE sub-vector """
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

        # dictionnary of arg_node (scalar) -> new variable (vector) mapping
        vec_arg_dict = dict(
            (arg_node, Variable("vec_%s" % arg_node.get_tag(), precision=self.vectorize_format(arg_node.get_precision(), vector_size))) for arg_node in arg_list)
        constant_dict = {}

        for i in range(int(vector_size / sub_vector_size)):
            if sub_vector_size == vector_size:
                arg_list_copy = dict((arg_node, Variable("vec_%s" % arg_node.get_tag() , precision = arg_node.get_precision())) for arg_node in arg_list)
                sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
                vectorization_map = {}
            else :
                # selection of a subset of the large vector to be the
                # sub-vector operand of this sub-vector path
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
        """ Return the vector version of size @p vectori_size of the scalar precision
            @p scalar_format """
        return VECTOR_TYPE_MAP[scalar_format][vector_size]

    def is_vectorizable(self, optree):
        """ Predicate to test if @p optree can be vectorized """
        arith_flag = isinstance(optree, ML_ArithmeticOperation)
        cst_flag     = isinstance(optree, Constant)
        var_flag     = isinstance(optree, Variable)
        if arith_flag or cst_flag or var_flag:
            return True
        elif isinstance(optree, DivisionSeed) or isinstance(optree, ReciprocalSquareRootSeed):
            return True
        return False

    def vector_replicate_scheme_in_place(self, optree, vector_size, _memoization_map=None):
        """ update optree to replace scalar precision by vector precision of size
            @p vector_size Replacement is made in-place: node are kept unchanged 
                      except for the precision
            @return modified operation graph
        """
        memoization_map = {} if _memoization_map is None else _memoization_map

        if optree in memoization_map:
            return memoization_map[optree]
        else:
            if self.is_vectorizable(optree):
                optree_precision = optree.get_precision()
                if optree_precision is None:
                    Log.report(Log.Error, "operation node precision is None for {}", optree)
                optree.set_precision(self.vectorize_format(optree.get_precision(), vector_size))
                if isinstance(optree, Constant):
                    # extend consntant value from scalar to replicated constant vector
                    optree.set_value([optree.get_value() for i in range(vector_size)])
                elif isinstance(optree, Variable):
                    # TODO: does not take into account intermediary variables
                    Log.report(Log.Warning, "Variable not supported in vector_replicate_scheme_in_place: {}", optree)
                    pass
                elif not isinstance(optree, ML_LeafNode):
                    for optree_input in optree.get_inputs():
                        self.vector_replicate_scheme_in_place(optree_input, vector_size, memoization_map)
                memoization_map[optree] = optree
                return optree
            elif isinstance(optree, ML_NewTable):
                # TODO: does not take into account intermediary variables
                Log.report(Log.Info, "skipping ML_NewTable node in vector_replicate_scheme_in_place: {} ", optree)
                pass
            else:
                Log.report(Log.Error, "optree not vectorizable: {}", optree)


