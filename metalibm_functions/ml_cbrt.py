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
import sys

import sollya

from sollya import (
    round, RN, RD
)

try:
    from sollya import cbrt
except ImportError:
    from sollya_extra_functions import cbrt

from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_Cbrt(ScalarUnaryFunction):
    function_name = "ml_cbrt"
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Cbrt,
                builtin from a default argument mapping overloaded with @p kw """
        default_args_cbrt = {
                "output_file": "my_cbrt.c",
                "function_name": "my_cbrt",
                "precision": ML_Binary32,
                "accuracy": ML_Faithful,
                "target": GenericProcessor.get_target_instance()
        }
        default_args_cbrt.update(kw)
        return DefaultArgTemplate(**default_args_cbrt)

    def generate_scheme(self):
        # declaring main input variable
        vx = self.implementation.add_input_variable("x", self.precision) 

        # declaring approximation parameters
        index_size = 6
        num_iteration = 8

        Log.set_dump_stdout(True)

        Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
        if self.debug_flag: 
                Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
                kwords["arg_value"] = vx
                kwords["function_name"] = self.function_name
                return RaiseReturn(*args, **kwords)



        def cbrt_newton_iteration(current_approx, input_value, input_inverse):
            # Cubic root of A is approximated by a Newton-Raphson iteration
            # on f(x) = 1 - A / x^3
            # x_n+1 = 4/3 * x_n - x_n^4 / (3 * A)
            # x_n+1 = 1/3 * (x_n * (1 - x_n^3/A) + x_n)

            approx_triple = Multiplication(current_approx, Multiplication(current_approx, current_approx))

            diff            = FMSN(approx_triple, input_inverse, Constant(1, precision = self.precision))
            injection = FMA(
                Multiplication(
                    current_approx, 
                    Constant(1/3.0, precision = self.precision),
                ),
                diff, current_approx)

            new_approx = injection

            return new_approx


        reduced_vx = MantissaExtraction(vx, precision = self.precision)

        int_precision = self.precision.get_integer_format()


        cbrt_approx_table = ML_NewTable(dimensions = [2**index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cbrt_approx_table"))
        for i in range(2**index_size):
            input_value = 1 + i / SollyaObject(2**index_size) 

            cbrt_approx = cbrt(input_value)
            cbrt_approx_table[i][0] = round(cbrt_approx, self.precision.get_sollya_object(), RN)

        # Modulo operations will returns a reduced exponent within [-3, 2]
        # so we approximate cbrt on this interval (with index offset by -3)
        cbrt_mod_table = ML_NewTable(dimensions = [6, 1], storage_precision = self.precision, tag = self.uniquify_name("cbrt_mod_table"))
        for i in range(6):
            input_value = SollyaObject(2)**(i-3)
            cbrt_mod_table[i][0] = round(cbrt(input_value), self.precision.get_sollya_object(), RN)

        vx_int = TypeCast(reduced_vx, precision = int_precision)
        mask = BitLogicRightShift(vx_int, self.precision.get_precision() - index_size, precision = int_precision)
        mask = BitLogicAnd(mask, Constant(2**index_size - 1, precision = int_precision), precision = int_precision, tag = "table_index", debug=debug_multi)
        table_index = mask

        int_precision = self.precision.get_integer_format()

        exp_vx = ExponentExtraction(vx, precision=int_precision, tag = "exp_vx")
        exp_vx_third = Division(exp_vx, Constant(3, precision=int_precision), precision=int_precision, tag = "exp_vx_third")
        exp_vx_mod     = Modulo(exp_vx, Constant(3, precision=int_precision), precision=int_precision, tag = "exp_vx_mod", debug=debug_multi)

        # offset on modulo to make sure table index is positive
        exp_vx_mod = exp_vx_mod + 3

        cbrt_mod = TableLoad(cbrt_mod_table, exp_vx_mod, Constant(0), tag = "cbrt_mod")

        init_approx = Multiplication(
            Multiplication(
                # approx cbrt(mantissa)
                TableLoad(cbrt_approx_table, table_index, Constant(0, precision = ML_Int32), tag="seed", debug=debug_multi),
                # approx cbrt(2^(e%3))
                cbrt_mod,
                tag="init_mult", 
                debug=debug_multi,
                precision=self.precision
            ),
            # 2^(e/3)
            ExponentInsertion(exp_vx_third, precision = self.precision, tag="exp_vx_third", debug=debug_multi),
            tag="init_approx",
            debug=debug_multi,
            precision = self.precision
        )

        inverse_red_vx = Division(Constant(1, precision = self.precision), reduced_vx)
        inverse_vx = Division(Constant(1, precision = self.precision), vx)

        current_approx = init_approx

        for i in range(num_iteration):
            #current_approx = cbrt_newton_iteration(current_approx, reduced_vx, inverse_red_vx) 
            current_approx = cbrt_newton_iteration(current_approx, vx, inverse_vx) 
            current_approx.set_attributes(tag="approx_%d" % i, debug=debug_multi)

        result = current_approx
        result.set_attributes(tag="result", debug=debug_multi)

        # last iteration
        ext_precision = ML_DoubleDouble
        xn_2 = Multiplication(current_approx, current_approx, precision = ext_precision)
        xn_3 = Multiplication(current_approx, xn_2, precision = ext_precision)

        FourThird = Constant(4/SollyaObject(3), precision = ext_precision)

        # main scheme
        Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
        scheme = Statement(
                                Return(
                                        result
                                    ))

        return scheme

    def numeric_emulate(self, input_value):
        return cbrt(input_value)

    standard_test_cases =[
        (sollya_parse(x),) for x in    ["1.1", "1.5"]
    ] + [
        (sollya.parse("0x1.4fc9bcp-1"),),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Cbrt.get_default_args())
    # argument extraction 
    args = arg_template.arg_extraction()

    ml_cbrt          = ML_Cbrt(args)
    ml_cbrt.gen_implementation()
