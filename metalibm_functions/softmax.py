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
)
from metalibm_core.core.ml_formats import (
    ML_UInt32, ML_Int32, ML_Binary32, ML_Void,
)
from metalibm_core.core.ml_table import (
    ML_NewTable,
)
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.precisions import (
    ML_CorrectlyRounded,
)
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_function import DefaultArgTemplate

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import (
    debug_multi
)

from metalibm_core.core.array_function import ML_ArrayFunction



class ML_SoftMax(ML_ArrayFunction):
    function_name = "ml_softmax"
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

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_SoftMax,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_exp = {
            "output_file": "ml_softmax.c",
            "function_name": "ml_softmax",
            "precision": ML_Binary32,
            "accuracy": ML_CorrectlyRounded,
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
        print(inc.get_str(display_precision=True))

        main_loop = Loop(
            ReferenceAssign(i, CU0),
            i < n,
            Statement(
                TableStore(TableLoad(src, i, precision=self.precision), dst, i, precision=ML_Void),
                ReferenceAssign(i, inc)
            ),
        )


        return main_loop


    def numeric_emulate(self, input_value):
        """ Numeric emaluation of exponential """
        return input_value

    standard_test_cases = [
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_SoftMax.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    ml_softmax = ML_SoftMax(args)

    ml_softmax.gen_implementation()
