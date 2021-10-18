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

import sollya

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_function import FunctionGroup

from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.array_function import DefaultArrayFunctionArgTemplate, ML_ArrayFunction, ML_ArrayFunctionArgTemplate
from metalibm_core.core.vla_common import (
    VLA_FORMAT_MAP, VLA_Binary32_l1, VLAGetLength, VLAOperation)
from metalibm_core.core.ml_operations import (
    Loop,
    ReferenceAssign, Statement, TableLoad, TableStore, Variable, Constant, is_leaf_node)
from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64, ML_Bool, ML_Format, ML_Int32, ML_Integer, ML_UInt32
from metalibm_core.core.static_vectorizer import (
    StaticVectorizer, and_merge_conditions, extract_const, extract_tables,
    fallback_policy, instanciate_variable)

from metalibm_core.opt.opt_utils import forward_attributes
from metalibm_core.opt.p_function_typing import PassInstantiateAbstractPrecision, PassInstantiatePrecision

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.ml_template import DefaultFunctionArgTemplate

from metalibm_core.targets.riscv.riscv_vector import RVV_VectorSize_T

from metalibm_functions.function_map import FUNCTION_MAP

class VLAVectorizer(StaticVectorizer):
    """ vectorizer for vector-length agnostic architecture """
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
            return VLA_FORMAT_MAP[(eltType, 1)]

        # dictionnary of arg_node (scalar) -> new variable (vector) mapping
        vec_arg_dict = dict(
            (arg_node, Variable("vec_%s" % arg_node.get_tag(), precision=vlaVectorizeType(arg_node.get_precision()))) for arg_node in arg_list)
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

    def vectorizeConstInplace(self, constOp, groupSize=1, memoization_map=None):
        """ Processing of Constant op during vectorization """
        # vector length agnostic architecture often support operations
        # between vector and scalar constant, so constants are not modified
        # by the VLAVectorizer
        return constOp

    def vectorizeFormat(self, eltType, groupSize=1):
        return VLA_FORMAT_MAP[(eltType, groupSize)]


class VLAVectorialFunction(ML_ArrayFunction):
    """ Meta function to generate Vector Length Agnostic implementation
        of a scalar meta-function """
    function_name = "vla_vectorial_function"
    def __init__(self, args=DefaultFunctionArgTemplate):
        # initializing base class
        ML_ArrayFunction.__init__(self, args)
        self.arity = 3
        precision_ptr = ML_Pointer_Format(self.precision)
        index_format = ML_UInt32
        # self.input_precisions can no longer be modified directly
        self._input_precisions = [
            precision_ptr,
            precision_ptr,
            index_format
        ]
        self.group_size = args.group_size
        self.function_ctor = args.function_ctor
        self.scalar_emulate = args.scalar_emulate

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_VectorialFunction,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "vla_vectorial_function.c",
            "function_name": "vla_vectorial_function",
            "function_ctor": FUNCTION_MAP["exp"],
            "scalar_emulate": sollya.exp,
            "group_size": 1,
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_exp.update(kw)
        return DefaultArrayFunctionArgTemplate(**default_args_exp)

    def generate_scheme(self):
        # declaring target and instantiating optimization engine
        precision_ptr = self.get_input_precision(0)
        index_format = self.get_input_precision(2)
        multi_elt_num = self.multi_elt_num

        dst = self.implementation.add_input_variable("dst", precision_ptr)
        src = self.implementation.add_input_variable("src", precision_ptr)
        n = self.implementation.add_input_variable("len", index_format)

        # instantiating vectorizer to vectorize scalar scheme
        vectorizer = VLAVectorizer()

        element_format = self.precision

        self.function_list = []

        element_format = vectorizer.vectorizeFormat(self.precision)

        elt_input = VLAOperation(src, i, specifier=TableLoad, precision=element_format)

        local_res = Variable("local_res", precision=element_format, var_type=Variable.Local)

        scalar_result = Variable("scalar_result", precision=self.precision, var_type=Variable.Local)

        # building meta generator for scalar scheme
        fct_ctor_args = self.function_ctor.get_default_args(
            precision=self.precision,
            libm_compliant=False,
        )
        meta_function = self.function_ctor(fct_ctor_args)
        scalar_scheme = meta_function.generate_scheme()

        # instanciating required passes for typing
        pass_inst_abstract_prec = PassInstantiateAbstractPrecision(self.processor)
        pass_inst_prec = PassInstantiatePrecision(self.processor, default_precision=None)

        # exectuting format instantiation passes on optree
        scalar_scheme = pass_inst_abstract_prec.execute_on_optree(scalar_scheme)
        scalar_scheme = pass_inst_prec.execute_on_optree(scalar_scheme)

        # extracting scalar argument from scalar meta function
        scalar_input = meta_function.implementation.arg_list[0]

        vectorSizeType = RVV_VectorSize_T
        # local sub-vector length used internally in the loop body
        vectorLocalLen = Variable("l", var_type=Variable.Local, precision=ML_Int32)
        vectorOffset = Variable("offset", var_type=Variable.Local, precision=ML_Int32)
        # remaining vector length
        vectorRemLen = Variable("remLen", var_type=Variable.Local)
        vec_arg_list, vector_scheme, vector_mask = \
            self.vectorize_scheme(scalar_scheme, scalar_arg_list, vectorLocalLen)

        assert len(vec_arg_list) == 1, "currently a single vector argument is supported"

        # we build the strip mining Loop
        mainLoop = Statement(
            ReferenceAssign(vectorRemLen, n),
            ReferenceAssign(vectorOffset, 0),
            # strip mining
            Loop(vectorLocalLen >= 0,
                Statement(
                    # assigning local vector length
                    ReferenceAssign(vectorLocalLen, VLAGetLength(vectorLocalLen, precision=vectorSizeType)),
                    # assigning inputs
                    ReferenceAssign(vec_arg_list[0], VLAOperation(src, vectorOffset, vectorLocalLen, specifier=TableLoad)),
                    # computing and storing results
                    VLAOperation(vector_scheme, dst, vectorOffset, vectorLocalLen, specifier=TableStore),
                    # updating remaining vector length by subtracting the number of elements
                    # evaluated in the loop body
                    ReferenceAssign(vectorRemLen, vectorRemLen - vectorLocalLen),
                    # updating source/destination offset
                    ReferenceAssign(vectorOffset, vectorOffset, vectorLocalLen),
                )
            )
        )

        return mainLoop

    def generate_function_list(self):
        self.implementation.set_scheme(self.generate_scheme())
        return FunctionGroup([self.implementation], self.function_list)


    def numeric_emulate(self, input_value):
        """ Numeric emulation of scalar function """
        return self.scalar_emulate(input_value)

    standard_test_cases = []



if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArrayFunctionArgTemplate(default_arg=VLAVectorialFunction.get_default_args())
    arg_template.get_parser().add_argument(
        "--function", dest="function_ctor", default=FUNCTION_MAP["exp"], type=(lambda v: FUNCTION_MAP[v]),
        help="define the function to be applied elementwise")
    arg_template.get_parser().add_argument(
        "--use-libm-fct", dest="use_libm_function", default=None,
        action="store", help="use standard libm function to implement element computation")
    arg_template.get_parser().add_argument(
        "--multi-elt-num", dest="multi_elt_num", default=1, type=int,
        action="store", help="number of vector element to be computed at each loop iteration")
    arg_template.get_parser().add_argument(
        "--scalar-emulate", dest="scalar_emulate", default=sollya.exp,
        type=(lambda s: eval(s, {"sollya": sollya})),
        action="store", help="function to use to compute exact expected value")
    # argument extraction
    args = arg_template.arg_extraction()

    vlaVectorialFunction = VLAVectorialFunction(args)

    vlaVectorialFunction.gen_implementation()
