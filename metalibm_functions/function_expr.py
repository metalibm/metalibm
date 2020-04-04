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
import re
import sollya


from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.ml_operations import (
    Variable, FunctionObject, FunctionCall, Return, Statement,
    Division,
    is_leaf_node
)
from metalibm_core.core.ml_formats import ML_Binary32, ML_Float
from metalibm_core.core.legalizer import evaluate_graph
from metalibm_core.core.precisions import ML_Faithful

from metalibm_core.opt.p_function_inlining import generate_inline_fct_scheme
from metalibm_core.opt.opt_utils import evaluate_range


from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import DefaultArgTemplate, ML_NewArgTemplate
from metalibm_core.utility.log_report import Log


from metalibm_functions.function_map import FUNCTION_MAP


LOG_VERBOSE_FUNCTION_EXPR = Log.LogLevel("FunctionExprVerbose")

FUNCTION_OBJECT_MAPPING = {
    name: FunctionObject(name, [ML_Float] * FUNCTION_MAP[name][0].arity,
                         ML_Float, None, range_function=FUNCTION_MAP[name][2]) for name in FUNCTION_MAP
}

FCT_DESC_PATTERN = r"([-+/* ().,]|\d+|{}|[xyzt])*".format("|".join(FUNCTION_OBJECT_MAPPING.keys()))

def check_fct_expr(str_desc):
    """ check if function expression string is potentially valid """
    return not re.fullmatch(FCT_DESC_PATTERN, str_desc) is None

def function_parser(str_desc, var_mapping):
    """ parser of function expression, from str to ML_Operation graph

        :arg str_desc: expression string
        :type str_desc: str
        :arg var_mapping: pre-existing mapping str to Variable/ML_Operation
        :type var_mapping: dict
        :return: resulting operation graph
        :rtype: ML_Operation
    """
    var_mapping = var_mapping.copy()
    var_mapping.update(FUNCTION_OBJECT_MAPPING)
    graph = eval(str_desc, None, var_mapping)
    return graph

def count_expr_arity(str_desc):
    """ Determine the arity of an expression directly from its str
        descriptor """
    # we start by extracting all words in the string
    # and then count the unique occurence of words matching "x", "y", "z" or "t"
    return len(set(var for var in re.findall("\w+", str_desc) if re.fullmatch("[xyzt]", var)))


def instanciate_fct_call(node, precision):
    """ replace FunctionCall node by the actual function
        scheme """
    vx_list = [node.get_input(i) for i in range(node.get_function_object().arity)]
    func_name = node.get_function_object().name
    fct_ctor, fct_args, fct_range_function = FUNCTION_MAP[func_name]
    var_result = Variable("local_result", precision=precision, var_type=Variable.Local)
    local_args = {"precision": precision, "libm_compliant": False}
    local_args.update(fct_args)
    fct_scheme = generate_inline_fct_scheme(fct_ctor, var_result, vx_list,
                                            local_args)
    return var_result, fct_scheme


class FunctionExpression(ML_FunctionBasis):
    """ class for meta-function taking as argument a string expression
        and generating the correspond implementation """
    function_name = "generic_function"
    def __init__(self, args):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)
        self.function_expr_str = args.function_expr_str[0]
        self.arity = count_expr_arity(self.function_expr_str)
        self.function_expr = None
        self.var_mapping = None
        self.expand_div = args.expand_div

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for FunctionExpression,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_log = {
            "output_file": "func_expr.c",
            "function_name": "func_expr",
            "function_expr_str": "exp(x)",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "expand_div": False,
            "target": GenericProcessor.get_target_instance(),
        }
        default_args_log.update(kw)
        return DefaultArgTemplate(**default_args_log)

    def instanciate_graph(self, op_graph, memoization_map=None, expand_div=False):
        """ instanciate function graph, replacing FunctionCall node
            by expanded function implementation """
        memoization_map = memoization_map or {}
        statement = Statement()
        def rec_instanciate(node):
            """ recursive internal function for function graph instaciation """
            new_node = None
            if node in memoization_map:
                return memoization_map[node]
            elif isinstance(node, FunctionCall):
                # recursively going through the input graph of FunctionCall for
                # instanciation
                for arg_index in range(node.get_function_object().arity):
                    input_node = rec_instanciate(node.get_input(arg_index))
                    if not input_node is None:
                        node.set_input(arg_index, input_node)
                result_var, fct_scheme = instanciate_fct_call(node, self.precision)
                statement.add(result_var) # making sure result var is declared previously
                statement.add(fct_scheme)
                new_node = result_var
                new_node.set_interval(node.get_interval())
            elif isinstance(node, Division) and expand_div:
                new_node = FUNCTION_OBJECT_MAPPING["div"](node.get_input(0), node.get_input(1))
                new_node.set_attributes(precision=node.get_precision(), interval=node.get_interval())
                new_node = rec_instanciate(new_node)
            elif is_leaf_node(node):
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

        Log.report(Log.Info, "evaluating function range")
        evaluate_range(self.function_expr, update_interval=True)
        Log.report(LOG_VERBOSE_FUNCTION_EXPR, "scheme is: \n{}", self.function_expr.get_str(depth=None, display_interval=True))

        # defined copy map to avoid copying input Variables
        copy_map = dict((var, var) for var in self.var_mapping.items())

        function_expr_copy = self.function_expr.copy(copy_map)

        result, scheme = self.instanciate_graph(function_expr_copy, expand_div=self.expand_div)
        scheme.add(Return(result, precision=self.precision))

        return scheme

    def numeric_emulate(self, *input_values):
        """ exact numerical emulation of self's function """
        value_mapping = {
            self.var_mapping[var_tag]: input_values[arg_index] for arg_index, var_tag in enumerate(["x", "y", "z", "t"]) if var_tag in self.var_mapping
        }
        # TODO function evaluation graph could be pre-compiled during generate scheme
        # of function_expr_str parsing
        function_mapping = {
            tag: FUNCTION_MAP[tag][2] for tag in FUNCTION_MAP
        }
        return evaluate_graph(self.function_expr, value_mapping, function_mapping)


if __name__ == "__main__":
    # auto-test
    ARG_TEMPLATE = ML_NewArgTemplate(default_arg=FunctionExpression.get_default_args())
    ARG_TEMPLATE.get_parser().add_argument(
        "function_expr_str", nargs=1, type=str,
        help="function expression to generate"
    )
    ARG_TEMPLATE.get_parser().add_argument(
        "--expand-div", action="store_const", default=False, const=True,
        help="expand division into function scheme")

    ARGS = ARG_TEMPLATE.arg_extraction()

    ML_FUNCTION_EXPR = FunctionExpression(ARGS)
    ML_FUNCTION_EXPR.gen_implementation()
