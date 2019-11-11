# -*- coding: utf-8 -*-

## @package opt.runtime_error_eval
#  Metalibm runtime error evaluator

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

###############################################################################
# created:          Nov  9th, 2019
# last-modified:    Nov 10th, 2019
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.core.ml_operations import (
    is_leaf_node, FunctionObject,
    Conversion, is_conversion,
    TypeCast, is_typecast,
    Constant, Abs, Subtraction)
from metalibm_core.core.legalizer import is_constant
from metalibm_core.core.ml_formats import ML_Void

from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import TemplateOperatorFormat

from metalibm_core.utility.log_report import Log

LOG_RUNTIME_EVAL_ERROR = Log.LogLevel("LogRuntimeEvalError")


def evaluate_typecast_value(optree, value):
    assert isinstance(optree, TypeCast)
    input_format = optree.get_input(0).get_precision()
    output_format = optree.get_precision() 
    input_value = input_format.get_integer_coding(value)
    output_value = output_format.get_value_from_integer_coding(input_value, base=None)
    Log.report(LOG_RUNTIME_EVAL_ERROR, "value={}, input_value= {}, output_value={}", value, input_value, output_value) 
    return output_value

def evaluate_conversion_value(optree, value):
    return optree.get_precision().round_sollya_object(value)


def evaluate_graph_value(optree, input_mapping, memoization_map=None):
    """ Given the node -> value mapping input_mapping, evaluate
        optree numerical value
    """
    # initializing memoization_map
    memoization_map = {} if memoization_map is None else memoization_map
    # computing values
    if optree in memoization_map:
        return memoization_map[optree]
    elif optree in input_mapping:
        value = input_mapping[optree]
    elif is_constant(optree):
        value = optree.get_value()
    elif is_typecast(optree):
        input_value = evaluate_graph_value(optree.get_input(0), input_mapping, memoization_map)
        value = evaluate_typecast_value(optree, input_value)
    elif is_conversion(optree):
        input_value = evaluate_graph_value(optree.get_input(0), input_mapping, memoization_map)
        value = evaluate_conversion_value(optree, input_value)
    else:
        args_interval = tuple(
            evaluate_graph_value(op, input_mapping, memoization_map) for op in
            optree.get_inputs()
        )
        value = optree.apply_bare_range_function(args_interval)
    memoization_map[optree] = value
    Log.report(LOG_RUNTIME_EVAL_ERROR, "node {} value has been evaluated to: {}", optree.get_tag(), value)
    return value

def get_printf_value(optree, error_value, expected_value, language=C_Code):
    """ generate a printf call to display the local error value
        alongside the expected value and result
    """
    error_display_format = error_value.get_precision().get_display_format(language)
    expected_display_format = expected_value.get_precision().get_display_format(language)
    result_display_format = optree.get_precision().get_display_format(language)

    # generated function expects 3 arguments, optree value, error value and
    # expected value, in that order
    error_vars = error_display_format.pre_process_fct("{1}")
    expected_vars = expected_display_format.pre_process_fct("{2}")
    result_vars = result_display_format.pre_process_fct("{0}")

    template = ("printf(\"node {:35} error is {}, expected {} got {}\\n\", {}, {}, {})").format(
                    str(optree.get_tag()),
                    error_display_format.format_string,
                    expected_display_format.format_string,
                    result_display_format.format_string,
                    error_vars,
                    expected_vars,
                    result_vars
                )

    arg_format_list = [
        optree.get_precision(),
        error_value.get_precision(),
        expected_value.get_precision()
    ]
    printf_op = TemplateOperatorFormat(template, void_function=True, arity=3)
    printf_input_function = FunctionObject("printf", arg_format_list, ML_Void, printf_op)
    return printf_input_function(optree, error_value, expected_value)

def generate_node_eval_error(optree, input_mapping, node_error_map, node_value_map):
    if optree in node_error_map or optree in input_mapping:
        return
    # placeholder to avoid diplicate complication
    node_error_map[optree] = None

    # recursive on node inputs
    if not is_leaf_node(optree):
        for op in optree.get_inputs():
            generate_node_eval_error(op, input_mapping, node_error_map, node_value_map)

    expected_value = node_value_map[optree]
    assert expected_value != None
    expected_node = Constant(expected_value, precision=optree.get_precision())
    precision = optree.get_precision()
    #error_node = Abs(
    #    # FIXME/ may need to insert signed type if optree/expected_node are
    #    # unsigned
    #    Subtraction(
    #        optree,
    #        expected_node,
    #        precision=precision),
    #    precision=precision)
    error_node = Subtraction(
            optree,
            expected_node,
            precision=precision)
    error_display_statement = get_printf_value(optree, error_node, expected_node)
    node_error_map[optree] = error_display_statement

def generate_error_eval_graph(optree, input_mapping):
    """ given the mapping of input input_mapping compute the exact value
        for optree and compares it to the actual result

    """
    node_value_map = {}
    # starting at the root
    root_value = evaluate_graph_value(optree, input_mapping, node_value_map)

    node_error_map = {}

    # filling node_error_map
    generate_node_eval_error(optree, input_mapping, node_error_map, node_value_map)
    return node_error_map



