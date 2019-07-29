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
    Return, ML_LeafNode,
    FunctionObject,
)
from metalibm_core.core.ml_formats import (
    ML_UInt32, ML_Int32, ML_Binary32, ML_Void,
)
from metalibm_core.core.ml_table import (
    ML_NewTable,
)
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.precisions import (
    ML_CorrectlyRounded, ML_Faithful,
)
from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.array_function import ML_ArrayFunction


from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator

from metalibm_core.opt.p_function_inlining import inline_function


from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import (
    debug_multi
)



from metalibm_functions.ml_exp import ML_Exponential


class ML_VectorialFunction(ML_ArrayFunction):
    function_name = "ml_vectorial_function"
    def __init__(self, args=DefaultArgTemplate):
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

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_VectorialFunction,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_vectorial_function.c",
            "function_name": "ml_vectorial_function",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor()
        }
        default_args_exp.update(kw)
        return DefaultArgTemplate(**default_args_exp)

    def generate_scheme(self):
        # declaring target and instantiating optimization engine
        precision_ptr = self.get_input_precision(0)
        index_format = self.get_input_precision(2)

        dst = self.implementation.add_input_variable("dst", precision_ptr)
        src = self.implementation.add_input_variable("src", precision_ptr)
        n = self.implementation.add_input_variable("len", index_format)

        i = Variable("i", precision=index_format, var_type=Variable.Local)
        CU1 = Constant(1, precision=index_format)
        CU0 = Constant(0, precision=index_format)
        inc = i+CU1

        elt_input = TableLoad(src, i, precision=self.precision)

        local_exp = Variable("local_exp", precision=self.precision, var_type=Variable.Local)

        if self.use_libm_function:
            libm_exp_operator = FunctionOperator("expf", arity=1)
            libm_exp = FunctionObject("expf", [ML_Binary32], ML_Binary32, libm_exp_operator)

            elt_result = ReferenceAssign(local_exp, libm_exp(elt_input))
        else:
            exponential_args = ML_Exponential.get_default_args(
                precision=self.precision,
                libm_compliant=False,
            )

            meta_exponential = ML_Exponential(exponential_args)
            exponential_scheme = meta_exponential.generate_scheme()

            elt_result = inline_function(
                exponential_scheme,
                local_exp,
                meta_exponential.implementation.arg_list[0],
                elt_input,
            )



        main_loop = Loop(
            ReferenceAssign(i, CU0),
            i < n,
            Statement(
                ReferenceAssign(local_exp, 0),
                elt_result,
                TableStore(local_exp, dst, i, precision=ML_Void),
                ReferenceAssign(i, inc)
            ),
        )


        return main_loop


    def numeric_emulate(self, input_value):
        """ Numeric emaluation of exponential """
        return sollya.exp(input_value)

    standard_test_cases = [
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_VectorialFunction.get_default_args())
    arg_template.get_parser().add_argument(
        "--use-libm-fct", dest="use_libm_function", default=False, const=True,
        action="store_const", help="use standard libm function to implement element computation")
    # argument extraction
    args = arg_template.arg_extraction()

    ml_vectorial_function = ML_VectorialFunction(args)

    ml_vectorial_function.gen_implementation()
