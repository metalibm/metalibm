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
# Author(s): Nicolas Brunie (nbrunie@kalray.eu)
# Created:          Aug  8th, 2017
# last-modified:    Mar  7th, 2018
###############################################################################

from functools import reduce

from metalibm_core.core.ml_formats import ML_Bool
from metalibm_core.core.ml_operations import (
    ML_LeafNode, Comparison, BooleanOperation,
    is_leaf_node,
    LogicalAnd, LogicalOr, Constant,
    BitLogicLeftShift, BitLogicRightShift,
    BitArithmeticRightShift,
)
from metalibm_core.core.advanced_operations import PlaceHolder
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.utility.log_report import Log


def evaluate_comparison_range(node):
    """ evaluate the numerical range of Comparison node, if any
        else returns None """
    return None

def is_comparison(node):
    """ test if node is a Comparison node or not """
    return isinstance(node, Comparison)

LOG_VERBOSE_EVALUATE_RANGE = Log.LogLevel("EvaluateRangeVerbose")

## Assuming @p optree has no pre-defined range, recursively compute a range
#  from the node inputs
def evaluate_range(optree, update_interval=False, memoization_map=None):
    """ evaluate the range of an Operation node

        Args:
            optree (ML_Operation): input Node

        Return:
            sollya Interval: evaluated range of optree or None if no range
                             could be determined
    """
    if memoization_map is None:
        memoization_map = {}
    init_interval = optree.get_interval()
    if not init_interval is None:
        return init_interval
    else:
        if optree in memoization_map:
            return memoization_map[optree]
        elif isinstance(optree, ML_LeafNode):
            op_range = optree.get_interval()
        elif is_comparison(optree):
            op_range = evaluate_comparison_range(optree)
            if update_interval:
                optree.set_interval(op_range)
        elif isinstance(optree, PlaceHolder):
            op_range = evaluate_range(optree.get_input(0),
                                      update_interval=update_interval,
                                      memoization_map=memoization_map)
            if update_interval:
                optree.set_interval(op_range)
        else:
            args_interval = tuple(
                evaluate_range(op, update_interval=update_interval,
                               memoization_map=memoization_map
                ) for op in optree.get_inputs())
            args_interval_map = {op: op_interval for op, op_interval in zip(optree.inputs, args_interval)}
            # evaluate_range cannot rely on bare_range_function only as some
            # operations (e.g. CountLeadingZeros) do not base interval computation
            # on their inputs' intervals but on other parameters
            ops_interval_get = lambda op: args_interval_map[op]
            op_range = optree.range_function(optree.inputs,
                                             ops_interval_getter=ops_interval_get)
            if update_interval:
                optree.set_interval(op_range)
        Log.report(LOG_VERBOSE_EVALUATE_RANGE, "range of {} is {}", optree, op_range)
        memoization_map[optree] = op_range
        return op_range


def forward_attributes(src, dst):
    """ forward compatible attributes from src node to dst node

        :param src: source source for attributes values
        :type src: ML_Operation
        :param dst: destination node for attributes copies
        :type dst: ML_Operation
    """
    dst.set_tag(src.get_tag())
    dst.set_debug(src.get_debug())
    dst.set_handle(src.get_handle())
    if hasattr(src.attributes, "init_stage"):
        forward_stage_attributes(src, dst)
    if isinstance(src, BooleanOperation) and isinstance(dst, BooleanOperation):
        dst.likely = src.likely


def forward_stage_attributes(src, dst):
    """ copy node's stage attributes from src node to dst node """
    dst.attributes.init_stage = src.attributes.init_stage


def depth_node_ordering(start_node, end_nodes):
    """ order the node between root start_node end end_nodes
        by depth (root first, starting with start_node)

        :param start_node: root of the sort (first node)
        :type start_node: ML_Operation
        :param end_nodes: nodes where the depth sort must end
        :type end_nodes: iterator over ML_Operation
        :return: depth ordered list of nodes
        :rtype: list(ML_Operation)
    """
    ordered_list = []
    ordered_set = set()
    working_list = [start_node]
    while working_list != []:
        node = working_list.pop(0)
        if not node in ordered_set:
            ordered_set.add(node)
            ordered_list.append(node)
        if not is_leaf_node(node) and not node in end_nodes:
            for node_op in node.get_inputs():
                working_list.append(node_op)
    return ordered_list

def logical_reduce(op_list, op_ctor=LogicalOr, precision=ML_Bool, **kw):
    """ Logical/Boolean operand list reduction """
    local_list = [node for node in op_list]
    while len(local_list) > 1:
        op0 = local_list.pop(0)
        op1 = local_list.pop(0)
        local_list.append(
            op_ctor(op0, op1, precision=precision)
        )
    # assigning attributes to the resulting node
    result = local_list[0]
    result.set_attributes(**kw)
    return result

## Specialization of logical reduce to OR operation
logical_or_reduce = lambda op_list, **kw: logical_reduce(op_list, LogicalOr, ML_Bool, **kw)
## Specialization of logical reduce to AND operation
logical_and_reduce = lambda op_list, **kw: logical_reduce(op_list, LogicalAnd, ML_Bool, **kw)



def uniform_list_check(value_list):
    """ Check that value_list is made of only a single value replicated in
        each element """
    return reduce((lambda acc, value: acc and value == value_list[0]), value_list, True)

def uniform_vector_constant_check(optree):
    """ check whether optree is a uniform vector constant """
    if isinstance(optree, Constant) and not optree.get_precision() is None \
            and optree.get_precision().is_vector_format():
        return uniform_list_check(optree.get_value())
    return False

def uniform_shift_check(optree):
    """ check whether optree is a bit shift by a uniform vector constant """
    if isinstance(optree, (BitLogicLeftShift, BitLogicRightShift, BitArithmeticRightShift)):
        return uniform_vector_constant_check(optree.get_input(1)) \
                or not optree.get_input(1).get_precision().is_vector_format()
    return False


def is_false(node):
    """ check if node is a Constant node whose value is equal to boolean False """
    return is_scalar_cst(node, False) or is_vector_uniform_cst(node, False)
def is_true(node):
    """ check if node is a Constant node whose value is equal to boolean True """
    return is_scalar_cst(node, True) or is_vector_uniform_cst(node, True)

def is_scalar_cst(node, value):
    """ check if node is a constant node with value equals to value """
    return isinstance(node, Constant) and not node.get_precision().is_vector_format() and node.get_value() == value
def is_vector_uniform_cst(node, scalar_value):
    """ check if node is a vector constant node with each value equals to
        scalar_value """
    return isinstance(node, Constant) and node.get_precision().is_vector_format() and node.get_value() == [scalar_value] * node.get_precision().get_vector_size()


def extract_tables(node):
    """ extract the set of all ML_Table nodes in the graph rooted at node """
    processed_set = set([node])
    table_set = set()
    working_set = [node]
    while working_set:
        elt = working_set.pop(0)
        if isinstance(elt, ML_NewTable):
            table_set.add(elt)
        elif not isinstance(elt, ML_LeafNode):
            for op_node in elt.inputs:
                if not op_node in processed_set:
                    processed_set.add(op_node)
                    working_set.append(op_node)
    return table_set
