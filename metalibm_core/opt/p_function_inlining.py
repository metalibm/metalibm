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
# created:          Jul 28th, 2019
# last-modified:    Jul 28th, 2019
###############################################################################

from metalibm_core.core.ml_operations import (
    ReferenceAssign, Return, ML_LeafNode, Variable,
)

def inline_function(fct_scheme, dst_var, inputs_var2value):
    """ generate an operation graph which inline function scheme @p fct_scheme
        assuming @p inputs_var2value contains var -> value mapping for replacing
        and stores its result into dst_var """
    memoization_map = {}
    def recursive_inline(node):
        if node in memoization_map:
            return memoization_map[node]
        elif node in inputs_var2value:
            input_value = inputs_var2value[node]
            memoization_map[node] = input_value
            return input_value
        elif isinstance(node, Return):
            node_value = recursive_inline(node.get_input(0))
            if not node_value is dst_var:
                new_node = ReferenceAssign(dst_var, node_value)
                memoization_map[node] = new_node
                return new_node
            else:
                return node_value
        elif isinstance(node, ML_LeafNode):
            memoization_map[node] = node
            return node
        else:
            for i, op in enumerate(node.inputs):
                node.set_input(i, recursive_inline(op))
            memoization_map[node] = node
            return node
    return recursive_inline(fct_scheme)


def generate_inline_fct_scheme(FctClass, dst_var, input_arg_list, custom_class_params):
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
        { meta_fct_object.implementation.arg_list[i]: input_arg_list[i] for i in range(meta_fct_object.arity)}
    )
    return result_statement

