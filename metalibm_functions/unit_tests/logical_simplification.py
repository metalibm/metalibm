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
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
""" Node range evaluation unit test """

import sollya

from sollya import parse as sollya_parse
from sollya import Interval, inf, sup
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import (
    Variable, Return,
    Min, Max,
    Comparison, Addition, Select, Constant, Conversion,
    MantissaExtraction, ExponentExtraction,
    ConditionBlock,
    LogicalOr, LogicalNot, LogicalAnd,
)
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Binary32,
)
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.utility.ml_template import \
    ML_NewArgTemplate
from metalibm_core.utility.log_report import Log
from metalibm_core.core.ml_hdl_format import fixed_point

from metalibm_functions.unit_tests.utils import TestRunner

from metalibm_core.opt.opt_utils import evaluate_range


from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)



class UT_LogicalSimplification(ML_FunctionBasis, TestRunner):
    function_name = "ml_num_simplification"
    """ Numerical Simplification unit-test """
    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Exponential,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_logical_simplification.c",
            "function_name": "ut_logical_simplification",
            "passes": ["beforecodegen:dump", "beforecodegen:numerical_simplification", "beforecodegen:dump"],
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def __init__(self, args):
        """ Initialize """
        # initializing base class
        ML_FunctionBasis.__init__(self, args=args)

        self.accuracy = args.accuracy
        self.precision = args.precision
        self.arity = 3

    def generate_scheme(self):
        """ main scheme generation """
        input_precision = self.precision
        output_precision = self.precision

        var_x = self.implementation.add_input_variable("x", input_precision)
        # declaring main input variable
        cond = LogicalAnd(
            LogicalNot(var_x > 3),
            LogicalNot(var_x < -3)
        )

        scheme = ConditionBlock(
            cond,
            Return(var_x),
            Return(var_x + var_x)
        )
        return scheme


    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        raise NotImplementedError


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_logical_simplification = UT_LogicalSimplification(args)
        ut_logical_simplification.gen_implementation()

        return True

run_test = UT_LogicalSimplification.__call__


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_NewArgTemplate(
        default_arg=UT_LogicalSimplification.get_default_args()
    )

    # argument extraction
    args = main_arg_template.arg_extraction()

    if run_test(args):
        exit(0)
    else:
        exit(1)
