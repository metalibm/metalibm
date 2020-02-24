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
###############################################################################
import sys

import sollya

S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.legalizer import evaluate_graph

from metalibm_core.opt.p_function_inlining import generate_inline_fct_scheme


from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract, is_gappa_installed
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.debug_utils import debug_multi

from metalibm_functions.function_map import FUNCTION_MAP


LOG_VERBOSE_FUNCTION_EXPR = Log.LogLevel("FunctionExprVerbose")

def function_parser(str_desc, var_mapping):
    exp = FunctionObject("exp", [ML_Float], ML_Float, None)
    sqrt = FunctionObject("sqrt", [ML_Float], ML_Float, None)
    local_mapping = {"exp": exp, "sqrt": sqrt}
    local_mapping.update(var_mapping)
    print(local_mapping)
    graph = eval(str_desc, None, local_mapping)
    return graph


def instanciate_fct_call(node, precision):
    """ replace FunctionCall node by the actual function
        scheme """
    vx = node.get_input(0)
    func_name = node.get_function_object().name
    fct_ctor = FUNCTION_MAP[func_name]
    var_result = Variable("local_result", precision=precision, var_type=Variable.Local)
    fct_scheme = generate_inline_fct_scheme(fct_ctor, var_result, vx,
        {"precision": precision, "libm_compliant": False}
    )
    return var_result, fct_scheme


class FunctionExpression(ML_FunctionBasis):
    function_name = "generic_function"
    def __init__(self, args):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)
        self.function_expr_str = args.function_expr_str[0]


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for FunctionExpression,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_log = {
            "output_file": "func_expr.c",
            "function_name": "func_expr",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor(),
        }
        default_args_log.update(kw)
        return DefaultArgTemplate(**default_args_log)

    def instanciate_graph(self, op_graph, memoization_map=None):
        memoization_map = memoization_map or {}
        statement = Statement()
        def rec_instanciate(node):
            new_node = None
            if node in memoization_map:
                return memoization_map[node]
            elif isinstance(node, FunctionCall):
                # FIXME: manage only unary functions
                input_node = rec_instanciate(node.get_input(0))
                result_var, fct_scheme = instanciate_fct_call(node, self.precision)
                statement.add(result_var) # making sure result var is declared previously
                statement.add(fct_scheme)
                new_node = result_var
            elif isinstance(node, ML_LeafNode):
                # unmodified
                new_node = None
            else:
                for index, op in enumerate(node.get_inputs()):
                    new_op = rec_instanciate(op)
                    if not new_op is None:
                        node.set_input(index, new_op)
                statement.add(node)
            memoization_map[node] = new_node
            return new_node
        final_node = rec_instanciate(op_graph) or op_graph
        return final_node, statement


    def generate_scheme(self):
        self.var_mapping = {}
        for var_index in range(self.arity):
            # FIXME: maximal arity is 4
            var_tag = ["x", "y", "z", "t"][var_index]
            self.var_mapping[var_tag] = self.implementation.add_input_variable(
                var_tag, self.get_input_precision(var_index),
                interval=self.input_intervals[var_index])

        self.function_expr = function_parser(self.function_expr_str, self.var_mapping)

        function_expr_copy = self.function_expr.copy(dict((var, var) for var in self.var_mapping.items()))
        print(function_expr_copy)

        result, scheme = self.instanciate_graph(function_expr_copy)
        scheme.add(Return(result))
        print("scheme is: \n{}", scheme.get_str(depth=None))

        return scheme

    def numeric_emulate(self, input_value):
        value_mapping = {
            self.var_mapping["x"]: input_value
        }
        function_mapping = {
            "exp": (lambda v: sollya.exp(v)),
            "sqrt": (lambda v: sollya.sqrt(v)),
        }
        return evaluate_graph(self.function_expr, value_mapping, function_mapping)


if __name__ == "__main__":
    # auto-test
    ARG_TEMPLATE = ML_NewArgTemplate(default_arg=FunctionExpression.get_default_args())
    ARG_TEMPLATE.get_parser().add_argument(
        "function_expr_str", nargs=1, type=str,
        help="function expression to generate"
    )

    args = ARG_TEMPLATE.arg_extraction()

    ml_function_expr = FunctionExpression(args)
    ml_function_expr.gen_implementation()
