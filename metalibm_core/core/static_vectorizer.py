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

from functools import reduce

from metalibm_core.core.ml_formats import VECTOR_TYPE_MAP, ML_Bool, ML_Integer
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.opt.opt_utils import extract_tables
from metalibm_core.core.legalizer import is_constant


# high verbosity log-level for optimization engine
LOG_LEVEL_VECTORIZER_VERBOSE = Log.LogLevel("VectorizerVerbose")


def fallback_policy(cond, cond_block, if_branch, else_branch):
    """ default fallback policy used by StaticVectorizer: if no
        likely branch can be found then the if-branch is selected """
    return if_branch, [cond]


def and_merge_conditions(condition_list, bool_precision=ML_Bool):
    """ merge predicates listed in @p condition_list using a conjonctive form
        (logical and) """
    assert(len(condition_list) >= 1)
    if len(condition_list) == 1:
        return condition_list[0]
    else:
        half_size = int(len(condition_list) / 2)
        first_half    = and_merge_conditions(condition_list[:half_size])
        second_half = and_merge_conditions(condition_list[half_size:])
        return LogicalAnd(first_half, second_half, precision = bool_precision)


def assembling_vector(args, precision=None, tag=None):
    """ Assembling a vector from sub-vectors, simplify to identity
        if there is only ONE sub-vector """
    if len(args) == 1:
        return args[0]
    else:
        return VectorAssembling(*args, precision=precision, tag=tag)

def extract_const(in_dict):
    """ extract all constant node from node:node dict @p in_dict """
    result_dict = {}
    for keys, values in in_dict.items():
        if isinstance(keys, Constant):
            result_dict.update({keys:values})
    return result_dict


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


def no_scalar_fallback_required(mask):
    """ Test whether a vector-mask is fully set to True, in order to
        disable scalar fallback generation """
    if isinstance(mask, VectorAssembling):
        # recursive application for sub-vector support
        return reduce(lambda acc, v: (no_scalar_fallback_required(v) and acc), mask.get_inputs(), True)
    elif isinstance(mask, Conversion):
        return no_scalar_fallback_required(mask.get_input(0))
    elif isinstance(mask, Constant):
        if mask.get_precision().is_vector_format():
            return reduce(lambda v, acc: (v and acc), mask.get_value(), True)
        else:
            # required to supported degenerate case (sub_)vector_size=1
            return mask.get_value()
    # fallback
    return False

def vectorize_format(scalar_format, vector_size, bool_specifier=None):
    """ Return the vector version of size @p vector_size of the scalar precision
        @p scalar_format
        bool_specifier indicates which types of boolean vector format must be chosen:
            None -> virtual bool
            int  -> size boolean formats

    """
    if vector_size == 1:
        # degenerate case vector-size = 1 => scalar fallback
        return scalar_format
    elif scalar_format is ML_Bool:
        # boolean format must be mapped to virtual boolean format
        return VECTOR_TYPE_MAP[scalar_format][bool_specifier][vector_size]
    else:
        return VECTOR_TYPE_MAP[scalar_format][vector_size]


def split_vectorial_op(node, output_vsize=2):
    """ Split a vectorial node <node> into a list of
        sub-vectors, each of size output_vsize

        input <node> vector-size must be a multiple of <output_vsize> """
    input_vsize = node.get_precision().get_vector_size()
    scalar_format = node.get_precision().get_scalar_format()
    if is_constant(node):
        sub_ops = [Constant(
            [node.get_value()[sub_id * output_vsize + j] for j in range(output_vsize)],
            precision=vectorize_format(scalar_format, output_vsize)
        ) for sub_id in range(input_vsize // output_vsize)]
    else:
        CI = lambda v: Constant(v, precision=ML_Integer)
        bool_specifier = None
        if scalar_format is ML_Bool:
            bool_specifier = node.get_precision().boolean_bitwidth
        sub_vector_fmt = vectorize_format(scalar_format, output_vsize, bool_specifier=bool_specifier)
        sub_ops = [SubVectorExtract(node, *tuple(CI(sub_id * output_vsize +j) for j in range(output_vsize)), precision=sub_vector_fmt) for sub_id in range(input_vsize // output_vsize)]
        #split_ops = [VectorElementSelection(node, Constant(i, precision=ML_Integer), precision=scalar_format) for i in range(input_vsize)]

        #sub_ops = [VectorAssembling(
        #    *tuple(split_ops[sub_id * output_vsize + j] for j in range(output_vsize)),
        #    precision=vectorize_format(scalar_format, output_vsize)
        #) for sub_id in range(input_vsize // output_vsize)]
    return sub_ops

def v4_to_v2_split(node):
    """ split node from v4 format to 2x v2 format """
    # TODO/FIXME: bool_specifier

    split_inputs = [split_vectorial_op(op, output_vsize=2) for op in node.inputs]
    half_node_format = vectorize_format(node.get_precision().get_scalar_format(), 2, bool_specifier=32)
    low = node.copy(copy_map={op: sub_op[0] for op, sub_op in zip(node.inputs, split_inputs)})
    low.set_precision(half_node_format)
    high = node.copy(copy_map={op: sub_op[1] for op, sub_op in zip(node.inputs, split_inputs)})
    high.set_precision(half_node_format)
    result = VectorAssembling(low, high, precision=node.get_precision())
    print("splitting {} into {}".format(node.get_str(), result.get_str(display_precision=True, depth=3)))
    return result

##
class StaticVectorizer(object):
    """ Mapping of size, scalar format to vector format """
    def vectorize_scheme(self, optree, arg_list, vector_size, sub_vector_size=None):
        """ optree static vectorization
            @param optree ML_Operation object, root of the DAG to be vectorized
            @param arg_list list of ML_Operation objects used as arguments by
                   optree
            @param vector_size integer size of the vectors to be generated
                   process
            @return pair ML_Operation, CodeFunction of vectorized scheme and
                    scalar callback
        """
        # TODO/FIXME: const table should not be copied
        table_set = extract_tables(optree)
        init_local_mapping = {table:table for table in table_set if table.const}

        # defaulting sub_vector_size to vector_size    when undefined
        sub_vector_size = vector_size if sub_vector_size is None else sub_vector_size
        vectorized_path = self.extract_vectorizable_path(optree, fallback_policy, local_mapping=init_local_mapping)
        linearized_most_likely_path = vectorized_path.linearized_optree
        validity_list = vectorized_path.validity_mask_list

        Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "validity_list: {}", validity_list)
        Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "linearized_most_likely_path: {}", linearized_most_likely_path.get_str(depth=None, display_precision=True))

        # replacing temporary variables by their latest assigned values
        linearized_most_likely_path = instanciate_variable(linearized_most_likely_path, vectorized_path.variable_mapping)

        vector_paths        = []
        vector_masks        = []
        vector_arg_list = []

        # dictionnary of arg_node (scalar) -> new variable (vector) mapping
        vec_arg_dict = dict(
            (arg_node, Variable("vec_%s" % arg_node.get_tag(), precision=vectorize_format(arg_node.get_precision(), vector_size))) for arg_node in arg_list)
        constant_dict = {}

        for i in range(vector_size // sub_vector_size):
            if sub_vector_size == vector_size:
                # if there is only one sub_vector, we must be carreful not to replicate input variable
                # in a new Variable node, as it break node unicity required to detect scheme
                # input variables properly
                arg_list_copy = dict((arg_node, vec_arg_dict[arg_node]) for arg_node in arg_list)
                sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
                # vector argument variables are already vectorized and should
                # be used directly in vectorized scheme
                vectorization_map = dict((vec_arg_dict[arg_node], vec_arg_dict[arg_node]) for arg_node in arg_list)
            elif sub_vector_size == 1:
                # degenerate case of scalar sub-vector
                arg_list_copy = dict(
                    (
                        arg_node,
                        VectorElementSelection(
                           vec_arg_dict[arg_node],
                           i,
                           precision=arg_node.get_precision()
                        )
                    )
                    for arg_node in arg_list)
                sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
                vectorization_map = dict((arg, arg) for arg in sub_vec_arg_list)
            else :
                # selection of a subset of the large vector to be the
                # sub-vector operand of this sub-vector path
                arg_list_copy = dict(
                    (
                        arg_node,
                        assembling_vector(
                            tuple(
                                (VectorElementSelection(
                                    vec_arg_dict[arg_node],
                                    i*sub_vector_size + j,
                                    precision=arg_node.get_precision())
                                ) for j in range(sub_vector_size)
                            ),
                            precision=vectorize_format(
                                arg_node.get_precision(), sub_vector_size
                            ),
                            tag="%s%d" %(vec_arg_dict[arg_node].get_tag(), i)
                        )
                    )
                    for arg_node in arg_list)
                sub_vec_arg_list = [arg_list_copy[arg_node] for arg_node in arg_list]
                vectorization_map = dict((arg, arg) for arg in sub_vec_arg_list)

            if (i > 0):
                vect_const_dict = dict((constant_dict[key], constant_dict[key]) for key in constant_dict.keys())
                vectorization_map.update(vect_const_dict)

            vector_arg_list.append(sub_vec_arg_list)
            arg_list_copy.update(constant_dict)

            # adding const table in pre-copied map toi avoid replication
            arg_list_copy.update({table:table for table in table_set if table.const})
            sub_vector_path = linearized_most_likely_path.copy(arg_list_copy)

            sub_vector_path = self.vector_replicate_scheme_in_place(sub_vector_path, sub_vector_size, vectorization_map)
            vector_paths.append(sub_vector_path)

            # no validity condition for vectorization (always valid)
            if len(validity_list) == 0:
                Log.report(Log.Warning, "empty validity list encountered during vectorization")
                sub_vector_mask = Constant(True, precision = ML_Bool)
            else:
                sub_vector_mask = and_merge_conditions(validity_list).copy(arg_list_copy)

            sub_vector_mask = self.vector_replicate_scheme_in_place(sub_vector_mask, sub_vector_size, vectorization_map)
            vector_masks.append(sub_vector_mask)

            # for the first iteration (first sub-vector), we extract
            if i == 0:
                constant_dict = extract_const(arg_list_copy)

        vector_path = assembling_vector(tuple(vector_paths), precision = vectorize_format(linearized_most_likely_path.get_precision(), vector_size))
        vec_arg_list = [vec_arg_dict[arg_node] for arg_node in arg_list]
        vector_mask = assembling_vector(tuple(vector_masks), precision = vectorize_format(ML_Bool, vector_size))
        return vec_arg_list, vector_path, vector_mask



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
                optree.set_precision(vectorize_format(optree.get_precision(), vector_size))
                if isinstance(optree, Constant):
                    # extend consntant value from scalar to replicated constant vector
                    if vector_size > 1:
                        optree.set_value([optree.get_value() for i in range(vector_size)])
                elif isinstance(optree, Variable):
                    # TODO: does not take into account intermediary variables
                    Log.report(Log.Warning, "Variable not supported in vector_replicate_scheme_in_place: {}", optree)
                    pass
                elif not isinstance(optree, ML_LeafNode):
                    for input_id, optree_input in enumerate(optree.get_inputs()):
                        new_input = self.vector_replicate_scheme_in_place(optree_input, vector_size, memoization_map)
                        if new_input != optree_input and not new_input is None:
                            optree.set_input(input_id, new_input)
                memoization_map[optree] = optree
                return optree
            elif isinstance(optree, ML_NewTable):
                # TODO: does not take into account intermediary variables
                Log.report(Log.Info, "skipping ML_NewTable node in vector_replicate_scheme_in_place: {} ", optree)
                pass
            elif isinstance(optree, FunctionCall):
                # we assume function cannot be overloaded and are scalar
                for op in optree.inputs:
                    self.vector_replicate_scheme_in_place(op, vector_size, memoization_map)
                assert not optree.get_precision() is None
                func_obj = optree.get_function_object()
                if vector_size > 1:
                    new_node = VectorAssembling(
                        *tuple(
                            func_obj(
                                *tuple(op if not op.precision.is_vector_format() else VectorElementSelection(op, arg_id) for op in optree.inputs),
                                precision=optree.precision # scalar precision is unchanged
                            ) for arg_id in range(vector_size)
                        ),
                        precision=vectorize_format(optree.precision, vector_size)
                    )
                else:
                    new_node = func_obj(
                                *tuple(op for op in optree.inputs),
                                precision=optree.precision # scalar precision is unchanged
                    )
                Log.report(Log.Debug, "vectorizing {} to {}", optree, new_node)
                return new_node
            else:
                Log.report(Log.Error, "optree not vectorizable: {}", optree)

    def extract_vectorizable_path(self, optree, fallback_policy, bool_precision=ML_Bool, local_mapping=None):
        """ extract a linear execution path from optree by chosing
            most likely side on each conditional branch
            @param optree operation tree to extract fast path from
            @param fallback_policy lambda function
                  (cond, cond_block, if_branch, else_branch):
                         branch_to_consider, validity_mask_list
                 if else_branch == None, fallback_policy MUST return if_branch
            @return VectorizedPath object containing linearized optree,
                validity mask list and variable_mapping """
        # NOTES/ fallback_policy must return the if_branch in the case
        # the else_branch is undefined (None)
        # local mapping is used to store the most up-to-date mapping between Variable 
        # and their values
        local_mapping = {} if local_mapping is None else local_mapping
        class VectorizedPath:
            """ structure to store vectorizable path sub-graph """
            def __init__(self, linearized_optree, validity_mask_list, variable_mapping=None, final=False):
                self.linearized_optree = linearized_optree
                self.validity_mask_list = validity_mask_list
                self.variable_mapping = {} if variable_mapping is None else variable_mapping
                # does the path contains a Return value
                self.final = final

        """ look for the most likely Return statement """
        if isinstance(optree, ConditionBlock):
            cond   = optree.inputs[0]
            likely = cond.get_likely()
            vectorized_path = self.extract_vectorizable_path(optree.get_pre_statement(), fallback_policy, local_mapping=local_mapping)
            Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "extracting vectorizable path in {}", optree)
            # start by vectorizing the if-then-else pre-statement
            if not vectorized_path.linearized_optree is None:
              Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "   vectorizing pre-statement")
              return vectorized_path
            else:
              if likely:
                  Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "   cond likely: {}", cond)
                  if_branch = optree.inputs[1]
                  vectorized_path = self.extract_vectorizable_path(if_branch, fallback_policy, local_mapping=local_mapping)
                  return  VectorizedPath(
                            vectorized_path.linearized_optree,
                            vectorized_path.validity_mask_list + [cond],
                            vectorized_path.variable_mapping
                          )
              elif likely == False:
                  Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "   cond unlikely: {}", cond)
                  if len(optree.inputs) >= 3:
                    # else branch exists
                    else_branch = optree.inputs[2]
                    vectorized_path = self.extract_vectorizable_path(else_branch, fallback_policy, local_mapping=local_mapping)
                    extra_cond = [LogicalNot(cond, precision = bool_precision) ]
                    # return  linearized_optree, (validity_mask_list + [LogicalNot(cond, precision = bool_precision)])
                    return  VectorizedPath(
                              vectorized_path.linearized_optree,
                              vectorized_path.validity_mask_list + extra_cond,
                              vectorized_path.variable_mapping
                            )
                  else:
                    # else branch does not exists
                    return VectorizedPath(None, [])
              else:
                  # no likely identified => using fallback policy
                  Log.report(LOG_LEVEL_VECTORIZER_VERBOSE, "   cond likely undef: {}", cond)
                  if_branch = optree.inputs[1]
                  else_branch = optree.inputs[2] if len(optree.inputs) >= 3 else None
                  selected_branch, cond_mask_list = fallback_policy(cond, optree, if_branch, else_branch)
                  vectorized_path = self.extract_vectorizable_path(selected_branch, fallback_policy, local_mapping=local_mapping)
                  return  VectorizedPath(
                    vectorized_path.linearized_optree,
                    (cond_mask_list + vectorized_path.validity_mask_list),
                    vectorized_path.variable_mapping
                  )
        elif isinstance(optree, Statement):
            merged_variable_mapping = {}
            merged_validity_list = []
            result_path = VectorizedPath(None, [])
            for sub_stat in optree.inputs:
                vectorized_path = self.extract_vectorizable_path(sub_stat, fallback_policy, local_mapping=local_mapping)
                # The following ensures that the latest assignation
                # has priority over all the previous ones
                merged_variable_mapping.update(vectorized_path.variable_mapping)
                # The validity list is the accumulation of all the sub-statement
                # validity conditions
                merged_validity_list += vectorized_path.validity_mask_list
                if not vectorized_path.linearized_optree is None and result_path.linearized_optree is None: 
                    result_path =  vectorized_path
                if vectorized_path.final:
                    # if vectorized_path is final, we can exit now
                    break
            return VectorizedPath(result_path.linearized_optree, merged_validity_list, merged_variable_mapping)
        elif isinstance(optree, ReferenceAssign):
            var_dst   = optree.get_input(0)
            pre_var_value = optree.get_input(1)
            # to prevent Variable aliasing and reference towards out-of-date
            # value we copy the value assocaited with the variable, using
            # local_mapping to solve any pre-defined Variable-to-Value mapping
            var_value = pre_var_value.copy(local_mapping.copy())
            assert(var_value != None)

            local_mapping[var_dst] = var_value
            return VectorizedPath(None, [], {var_dst: var_value})
        elif isinstance(optree, Return):
            return VectorizedPath(optree.inputs[0].copy(local_mapping.copy()), [], final=True)
        else:
            return VectorizedPath(None, [])

