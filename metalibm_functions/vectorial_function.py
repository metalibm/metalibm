# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2019 Kalray
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
# created:          Jul 26th, 2019
# last-modified:    Jul 26th, 2019
###############################################################################
import sollya

S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import (
    Variable, Constant,
    Loop, ReferenceAssign, Statement,
    TableLoad, TableStore,
    Return,
    FunctionObject,
    VectorElementSelection, VectorAssembling,
    Division, Modulo,
)
from metalibm_core.core.ml_formats import (
    ML_UInt32, ML_Int32, ML_Binary32, ML_Void,
    VECTOR_TYPE_MAP, ML_Integer,
)
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.precisions import (
    ML_CorrectlyRounded, ML_Faithful,
)
from metalibm_core.core.ml_function import (
    generate_c_vector_wrapper
)
from metalibm_core.core.array_function import (
    ML_ArrayFunction, DefaultArrayFunctionArgTemplate,
    ML_ArrayFunctionArgTemplate
)
from metalibm_core.core.ml_call_externalizer import (
    generate_function_from_optree
)
from metalibm_core.core.ml_vectorizer import (
    StaticVectorizer, no_scalar_fallback_required
)


from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_function import FunctionGroup
from metalibm_core.code_generation.generator_utility import FunctionOperator


from metalibm_core.opt.p_function_inlining import inline_function
from metalibm_core.opt.p_function_typing import (
    PassInstantiateAbstractPrecision, PassInstantiatePrecision,
)

from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import (
    debug_multi
)


# TODO/FIXME: implement cleaner way to register and list meta-functions
from metalibm_functions.ml_exp import ML_Exponential
from metalibm_functions.ml_tanh import ML_HyperbolicTangent


def generate_inline_fct_scheme(FctClass, dst_var, input_arg, custom_class_params):
    """ generate the sub-graph corresponding to the implementation of
        @p FctClass with argument dict @p custom_class_params
        the result is stored in the node @p dst_var and the function's
        parameters are given in @p input_arg """
    # build argument dict for meta class
    meta_args = FctClass.get_default_args(**custom_class_params)

    meta_fct_object = FctClass(meta_args)

    # generate implementation DAG
    meta_scheme = meta_fct_object.generate_scheme()

    result_statement = inline_function(
        meta_scheme,
        dst_var,
        meta_fct_object.implementation.arg_list[0],
        input_arg
    )
    return result_statement

def vectorize_function_scheme(vectorizer, name_factory, scalar_scheme,
                              scalar_output_format,
                              scalar_arg_list, vector_size,
                              sub_vector_size=None):
    """ Use a vectorization engine @p vectorizer to vectorize the sub-graph @p
        scalar_scheme, that is transforming and inputs and outputs from scalar
        to vectors and performing required internal path duplication """

    sub_vector_size = vector_size if sub_vector_size is None else sub_vector_size

    vec_arg_list, vector_scheme, vector_mask = \
        vectorizer.vectorize_scheme(scalar_scheme, scalar_arg_list,
                                    vector_size, sub_vector_size)

    vector_output_format = vectorizer.vectorize_format(scalar_output_format,
                                                       vector_size)

    vec_res = Variable("vec_res", precision=vector_output_format,
                       var_type=Variable.Local)

    vector_mask.set_attributes(tag="vector_mask", debug = debug_multi)

    callback_name = "scalar_callback"
    scalar_callback_fct = generate_function_from_optree(name_factory,
                                                        scalar_scheme,
                                                        scalar_arg_list,
                                                        callback_name,
                                                        scalar_output_format)
    scalar_callback          = scalar_callback_fct.get_function_object()

    if no_scalar_fallback_required(vector_mask):
        function_scheme = Statement(
            Return(vector_scheme, precision=vector_output_format)
        )
    function_scheme = generate_c_vector_wrapper(vector_size,
                                                vec_arg_list, vector_scheme,
                                                vector_mask, vec_res,
                                                scalar_callback)

    return vec_res, vec_arg_list, function_scheme, scalar_callback, scalar_callback_fct

class ML_VectorialFunction(ML_ArrayFunction):
    function_name = "ml_vectorial_function"
    def __init__(self, args=DefaultArrayFunctionArgTemplate):
        # initializing base class
        ML_ArrayFunction.__init__(self, args)
        self.arity = 3
        precision_ptr = ML_Pointer_Format(self.precision)
        index_format = ML_UInt32
        self.input_precisions = [
            precision_ptr,
            precision_ptr,
            index_format
        ]
        self.use_libm_function = args.use_libm_function
        self.multi_elt_num = args.multi_elt_num
        self.function_ctor = args.function_ctor
        self.scalar_emulate = args.scalar_emulate

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_VectorialFunction,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_vectorial_function.c",
            "function_name": "ml_vectorial_function",
            "function_ctor": ML_Exponential,
            "use_libm_function": False,
            "scalar_emulate": sollya.exp,
            "multi_elt_num": 1,
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor()
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

        i = Variable("i", precision=index_format, var_type=Variable.Local)
        CU0 = Constant(0, precision=index_format)

        element_format = self.precision

        self.function_list = []

        if multi_elt_num > 1:
            element_format = VECTOR_TYPE_MAP[self.precision][multi_elt_num]

        elt_input = TableLoad(src, i, precision=element_format)

        local_exp = Variable("local_exp", precision=element_format, var_type=Variable.Local)

        if self.use_libm_function:
            libm_fct_operator = FunctionOperator(self.use_libm_function, arity=1)
            libm_fct = FunctionObject(self.use_libm_function, [ML_Binary32], ML_Binary32, libm_fct_operator)

            if multi_elt_num > 1:
                result_list = [libm_fct(VectorElementSelection(elt_input, Constant(elt_id, precision=ML_Integer), precision=self.precision)) for elt_id in range(multi_elt_num)]
                result = VectorAssembling(*result_list, precision=element_format)
            else:
                result = libm_fct(elt_input)
            elt_result = ReferenceAssign(local_exp, result)
        else:
            if multi_elt_num > 1:
                scalar_result = Variable("scalar_result", precision=self.precision, var_type=Variable.Local)
                fct_ctor_args = self.function_ctor.get_default_args(
                    precision=self.precision,
                    libm_compliant=False,
                )

                meta_function = self.function_ctor(fct_ctor_args)
                exponential_scheme = meta_function.generate_scheme()

                # instanciating required passes for typing
                pass_inst_abstract_prec = PassInstantiateAbstractPrecision(self.processor)
                pass_inst_prec = PassInstantiatePrecision(self.processor, default_precision=None)

                # exectuting format instanciation passes on optree
                exponential_scheme = pass_inst_abstract_prec.execute_on_optree(exponential_scheme)
                exponential_scheme = pass_inst_prec.execute_on_optree(exponential_scheme)

                vectorizer = StaticVectorizer()

                # extracting scalar argument from meta_exponential meta function
                scalar_input = meta_function.implementation.arg_list[0]

                # vectorize scalar scheme
                vector_result, vec_arg_list, vector_scheme, scalar_callback, scalar_callback_fct = vectorize_function_scheme(
                    vectorizer, self.get_main_code_object(),
                    exponential_scheme, element_format.get_scalar_format(),
                    [scalar_input], multi_elt_num)

                elt_result = inline_function(
                    vector_scheme,
                    vector_result,
                    vec_arg_list[0],
                    elt_input
                )

                local_exp = vector_result

                self.function_list.append(scalar_callback_fct)
                libm_fct = scalar_callback

            else:
                scalar_input = elt_input
                scalar_result = local_exp

                elt_result = generate_inline_fct_scheme(
                    self.function_ctor, scalar_result, scalar_input,
                    {"precision": self.precision, "libm_compliant": False}
                )

        CU1 = Constant(1, precision=index_format)

        local_exp_init_value = Constant(0, precision=self.precision)
        if multi_elt_num > 1:
            local_exp_init_value = Constant([0]*multi_elt_num, precision=element_format)
            remain_n = Modulo(n, multi_elt_num, precision=index_format)
            iter_n = n - remain_n
            CU_ELTNUM = Constant(multi_elt_num, precision=index_format)
            inc = i+CU_ELTNUM
        else:
            remain_n = None
            iter_n = n
            inc = i+CU1

        # main loop processing multi_elt_num element(s) per iteration
        main_loop = Loop(
            ReferenceAssign(i, CU0),
            i < iter_n,
            Statement(
                ReferenceAssign(local_exp, local_exp_init_value),
                elt_result,
                TableStore(local_exp, dst, i, precision=ML_Void),
                ReferenceAssign(i, inc)
            ),
        )
        # epilog to process remaining item (when the length is not a multiple
        # of multi_elt_num)
        if not remain_n is None:
            # TODO/FIXME: try alternative method for processing epilog
            #             by using full vector length and mask
            epilog_loop = Loop(
                Statement(),
                i < n,
                Statement(
                    TableStore(
                        libm_fct(TableLoad(src, i, precision=self.precision)),
                        dst, i, precision=ML_Void),
                    ReferenceAssign(i, i+CU1),
                )
            )
            main_loop = Statement(
                main_loop,
                epilog_loop
            )

        return main_loop

    def generate_function_list(self):
        self.implementation.set_scheme(self.generate_scheme())
        return FunctionGroup([self.implementation], self.function_list)


    def numeric_emulate(self, input_value):
        """ Numeric emaluation of exponential """
        return self.scalar_emulate(input_value)

    standard_test_cases = [
    ]

# dict of (str) -> tuple(ctor, dict(ML_Format -> str))
# the first level key is the function name
# the first value of value tuple is the meta-function constructor
# the second value of the value tuple is a dict which associates to a ML_Format
# the corresponding libm function
FUNCTION_MAP = {
    "exp": ML_Exponential,
    "tanh": ML_HyperbolicTangent,
}


if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArrayFunctionArgTemplate(default_arg=ML_VectorialFunction.get_default_args())
    arg_template.get_parser().add_argument(
        "--function", dest="function_ctor", default=ML_Exponential, type=(lambda v: FUNCTION_MAP[v]),
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

    ml_vectorial_function = ML_VectorialFunction(args)

    ml_vectorial_function.gen_implementation()
