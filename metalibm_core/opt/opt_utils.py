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

from metalibm_core.core.ml_operations import (
    ML_LeafNode, Comparison, BooleanOperation,
    is_leaf_node,
)
from metalibm_core.core.ml_hdl_operations import (
    PlaceHolder
)


def evaluate_comparison_range(optree):
    return None

def is_comparison(optree):
    return isinstance(optree, Comparison)

## Assuming @p optree has no pre-defined range, recursively compute a range
#  from the node inputs
def evaluate_range(optree):
    """ evaluate the range of an Operation node

        Args:
            optree (ML_Operation): input Node

        Return:
            sollya Interval: evaluated range of optree or None if no range
                             could be determined
    """
    init_interval =  optree.get_interval()
    if not init_interval is None:
        return init_interval
    else:
        if isinstance(optree, ML_LeafNode):
            return optree.get_interval()
        elif is_comparison(optree):
            return evaluate_comparison_range(optree)
        elif isinstance(optree, PlaceHolder):
            return evaluate_range(optree.get_input(0))
        else:
            args_interval = tuple(
                evaluate_range(op) for op in
                optree.get_inputs()
            )
            return optree.apply_bare_range_function(args_interval)


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
    w = [start_node]
    while w != []:
        node = w.pop(0)
        if not node in ordered_set:
            ordered_set.add(node)
            ordered_list.append(node)
        if not is_leaf_node(node) and not node in end_nodes:
            for op in node.get_inputs():
                w.append(op)
    return ordered_list

