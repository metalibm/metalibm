# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:              Oct  3rd, 2021
# last-modified:        Oct  3rd, 2021
#
# author(s):    Nicolas Brunie
# description: Vector Length Agnostic Vectorizer implementation for Metalibm
###############################################################################

from metalibm_core.core.passes import PassScheduler
import sollya

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_function import FunctionGroup

from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.array_function import DefaultArrayFunctionArgTemplate, ML_ArrayFunction, ML_ArrayFunctionArgTemplate
from metalibm_core.core.vla_common import (
    VLA_FORMAT_MAP, VLA_Binary32_l1, VLAGetLength, VLAOperation)
from metalibm_core.core.ml_operations import (
    Addition,
    Conversion,
    Loop,
    ReferenceAssign,
    SpecifierOperation, Statement,
    Subtraction, TableLoad, TableStore, Variable, Constant, is_leaf_node)
from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Bool, ML_Format, ML_Int32, ML_Integer, ML_UInt32, ML_Void
from metalibm_core.core.static_vectorizer import (
    StaticVectorizer, and_merge_conditions, extract_const, extract_tables,
    fallback_policy, instanciate_variable)

from metalibm_core.opt.opt_utils import forward_attributes
from metalibm_core.opt.p_function_typing import PassInstantiateAbstractPrecision, PassInstantiatePrecision

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.ml_template import DefaultFunctionArgTemplate

from metalibm_core.targets.riscv.riscv_vector import RVV_VectorSize_T

from metalibm_functions.function_map import FUNCTION_MAP

from metalibm_core.utility.debug_utils import debug_multi

class VLAVectorizer(StaticVectorizer):
    """ vectorizer for vector-length agnostic architecture """
    def __init__(self, defaultGroupSize=1):
        """ """
        self.defaultGroupSize = defaultGroupSize


    def vectorize_scheme(self, optree, arg_list, vectorLen):
        """ optree static vectorization
            @param optree ML_Operation object, root of the DAG to be vectorized
            @param arg_list list of ML_Operation objects used as arguments by
                   optree
            @param vector_size integer size of the vectors to be generated
                   process
            @return pair ML_Operation, CodeFunction of vectorized scheme and
                    scalar callback
        """
        table_set = extract_tables(optree)
        init_local_mapping = {table:table for table in table_set if table.const}

        # defaulting sub_vector_size to vector_size when undefined
        vectorized_path = self.extract_vectorizable_path(optree, fallback_policy, local_mapping=init_local_mapping)
        linearized_most_likely_path = vectorized_path.linearized_optree
        validity_list = vectorized_path.validity_mask_list

        # replacing temporary variables by their latest assigned values
        linearized_most_likely_path = instanciate_variable(linearized_most_likely_path, vectorized_path.variable_mapping)

        vector_paths        = []
        vector_masks        = []
        vector_arg_list = []

        def vlaVectorizeType(eltType):
            """ return the corresponding VLA format for a given element type """
            # select group multiplier = 1 to for now
            return VLA_FORMAT_MAP[(eltType, self.defaultGroupSize)]

        # dictionnary of arg_node (scalar) -> new variable (vector) mapping
        vec_arg_dict = dict(
            (arg_node, Variable("vec_%s" % arg_node.get_tag(), precision=vlaVectorizeType(arg_node.get_precision()), var_type=Variable.Local)) for arg_node in arg_list)
        constant_dict = {}

        arg_list_copy = dict((arg_node, vec_arg_dict[arg_node]) for arg_node in arg_list)
        # vector argument variables are already vectorized and should
        # be used directly in vectorized scheme
        vectorization_map = dict((vec_arg_dict[arg_node], vec_arg_dict[arg_node]) for arg_node in arg_list)

        # adding const table in pre-copied map to avoid replication
        arg_list_copy.update({table:table for table in table_set if table.const})
        sub_vector_path = linearized_most_likely_path.copy(arg_list_copy)

        vector_path = self.vector_replicate_scheme_in_place(sub_vector_path, vectorLen, vectorization_map)

        # no validity condition for vectorization (always valid)
        if len(validity_list) == 0:
            Log.report(Log.Warning, "empty validity list encountered during vectorization")
            vector_mask = Constant(True, precision = ML_Bool)
        else:
            vector_mask = and_merge_conditions(validity_list).copy(arg_list_copy)

        # todo/fixme: implement proper length agnostic masking
        vector_mask = self.vector_replicate_scheme_in_place(vector_mask, vectorLen, vectorization_map)

        # for the first iteration (first sub-vector), we extract
        constant_dict = extract_const(arg_list_copy)
        vec_arg_list = [vec_arg_dict[arg_node] for arg_node in arg_list]
        return vec_arg_list, vector_path, vector_mask

    def vector_replicate_scheme_in_place(self, node, vectorLen, _memoization_map=None):
        """ update node to replace scalar precision by vector precision of size
            @p vector_size Replacement is made in-place: node are kept unchanged 
                      except for the precision
            @return modified operation graph
        """
        memoization_map = {} if _memoization_map is None else _memoization_map

        if node in memoization_map:
            return memoization_map[node]
        else:
            if self.is_vectorizable(node):
                optree_precision = node.get_precision()
                newNode = node # default: unmodified
                if isinstance(node, Constant):
                    # extend constant value from scalar to replicated constant vector
                    newNode = self.vectorizeConstInplace(node, vectorLen, memoization_map)
                elif isinstance(node, Variable):
                    # TODO: does not take into account intermediary variables
                    Log.report(Log.Warning, "Variable not supported in vector_replicate_scheme_in_place: {}", node)
                    pass
                elif not is_leaf_node(node):
                    opInputs = []
                    for input_id, optree_input in enumerate(node.get_inputs()):
                        new_input = self.vector_replicate_scheme_in_place(optree_input, vectorLen, memoization_map)
                        opInputs.append(new_input)
                    # extracting operation class
                    try:
                        opType = self.vectorizeFormat(optree_precision, self.defaultGroupSize)
                    except KeyError as e:
                        Log.report(Log.Error, "unable to determine a vector-format for node {}", node, error=e)
                    if isinstance(node, SpecifierOperation):
                        specifier = (node.__class__, node.specifier)
                    else:
                        specifier = node.__class__
                    assert not None in opInputs
                    newNode = VLAOperation(*opInputs, vectorLen, precision=opType, specifier=specifier)
                    forward_attributes(node, newNode)
                memoization_map[node] = newNode
                return newNode
            elif isinstance(node, ML_NewTable):
                # TODO: does not take into account intermediary variables
                Log.report(Log.Info, "skipping ML_NewTable node in vector_replicate_scheme_in_place: {} ", node)
                return node
            else:
                Log.report(Log.Error, "node not vectorizable: {}", node)

    def vectorizeConstInplace(self, constOp, groupSize=1, memoization_map=None):
        """ Processing of Constant op during vectorization """
        # vector length agnostic architecture often support operations
        # between vector and scalar constant, so constants are not modified
        # by the VLAVectorizer
        return constOp

    def vectorizeFormat(self, eltType, groupSize=1):
        return VLA_FORMAT_MAP[(eltType, groupSize)]