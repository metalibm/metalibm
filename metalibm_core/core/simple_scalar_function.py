# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          Feb 22nd, 2020
# last-modified:    Feb 22nd, 2020
###############################################################################

from metalibm_core.core.ml_operations import Variable
from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.opt.p_function_inlining import inline_function


class ScalarUnaryFunction(ML_FunctionBasis):
    """ Basic class for function with a single input, single output """
    arity = 1

    def generate_scalar_scheme(self, vx):
        """ generate scheme assuming single input is vx """
        raise NotImplementedError

    def generate_inline_scheme(self, vx):
        """ generate a pair <variable, scheme>
            scheme is the operation graph to compute self function on vx
            and variable is the result variable """
        result_var = Variable("r", precision=self.get_precision(), var_type=Variable.Local)
        scalar_scheme = self.generate_scalar_scheme(vx)
        result_scheme = inline_function(scalar_scheme, result_var, {vx: vx})

        return result_var, result_scheme

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("x", self.get_input_precision(),
                                                    interval=self.input_intervals[0])

        scalar_scheme = self.generate_scalar_scheme(vx)
        return scalar_scheme


class ScalarBinaryFunction(ML_FunctionBasis):
    """ Basic class for function with two inputs, single output """
    arity = 2

    def __init__(self, args):
        super().__init__(args)
        self.arity = ScalarBinaryFunction.arity
        if len(self.auto_test_range) != self.arity:
            self.auto_test_range = [self.auto_test_range[0]] * self.arity
        if len(self.bench_test_range) != self.arity:
            self.bench_test_range = [self.bench_test_range[0]] * self.arity

    @staticmethod
    def get_default_args(**kw):
        default_args = {
            "arity": 2
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scalar_scheme(self, vx, vy):
        """ generate scheme assuming two inputs vx and vy """
        raise NotImplementedError

    def generate_inline_scheme(self, vx):
        """ generate a pair <variable, scheme>
            scheme is the operation graph to compute self function on vx
            and variable is the result variable """
        result_var = Variable("r", precision=self.get_precision(), var_type=Variable.Local)
        scalar_scheme = self.generate_scalar_scheme(vx)
        result_scheme = inline_function(scalar_scheme, result_var, {vx: vx})

        return result_var, result_scheme

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("x", self.get_input_precision(),
                                                    interval=self.input_intervals[0])
        vy = self.implementation.add_input_variable("y", self.get_input_precision(),
                                                    interval=self.input_intervals[1])

        scalar_scheme = self.generate_scalar_scheme(vx, vy)
        return scalar_scheme

