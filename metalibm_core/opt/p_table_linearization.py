
# -*- coding: utf-8 -*-
###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2022 Nicolas Brunie
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
# Description: optimization pass to linearize tables
###############################################################################

from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.passes import METALIBM_PASS_REGISTER, OptreeOptimization, LOG_PASS_INFO
from metalibm_core.opt.node_transformation import Pass_NodeTransformation

from metalibm_core.core.ml_operations import Constant, TableLoad


def split2DTable(table):
    """ split a table[n][m] into m subtable[n] """
    n, m = table.dimensions
    subTables = [ML_NewTable([n], storage_precision = table.get_storage_precision(), empty=table.empty) for _ in range(m)]
    # filling tables
    for index in range(n):
        for subIndex in range(m):
            subTables[subIndex][index] = table[index][subIndex]
    return subTables 


class TableLinearizer(object):
    """ Basic table linearizer engine """
    def __init__(self, target):
        self.target = target
        self.memoization_map = {}
        self.linearizedTableMap = {}

    def expand_node(self, node):
        """ return the expansion of @p node when expandable
            else None """
        if node in self.memoization_map:
            return self.memoization_map[node]
        elif isinstance(node, TableLoad):
            table = node.get_input(0)
            if len(table.dimensions) == 2:
                # only 2D table are linearized
                if not table in self.linearizedTableMap:
                    self.linearizedTableMap[table] = split2DTable(table)
                mainIndex = node.get_input(1)
                subIndex = node.get_input(2)
                # subIndex should be a constant
                if isinstance(subIndex, Constant):
                    # TODO/OPT: could perform constant-propagation to create
                    # more constant index 
                    subTabIndex = subIndex.value 
                    result = TableLoad(self.linearizedTableMap[table][subTabIndex], mainIndex, precision=node.get_precision())
                else:
                    result = None
            else:
                result = None
        else:
            result = None

        self.memoization_map[node] = result
        return result


    def is_expandable(self, node):
        """ return True if @p node is expandable else False """
        return isinstance(node, TableLoad)


@METALIBM_PASS_REGISTER
class Pass_BasicLegalization(Pass_NodeTransformation):
    """ Basic legalization pass """
    pass_tag = "table_linearization"

    def __init__(self, target):
        OptreeOptimization.__init__(
            self, "table linearization pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        self.expander = TableLinearizer(target)

    def can_be_transformed(self, node, *args):
        """ Returns True if @p can be expanded from a multi-precision
            node to a list of scalar-precision fields,
            returns False otherwise """
        return self.expander.is_expandable(node)

    def transform_node(self, node, transformed_inputs, *args):
        """ If node can be transformed returns the transformed node
            else returns None """
        return self.expander.expand_node(node)

    def reconstruct_from_transformed(self, node, transformed_node):
        """return a node at the root of a transformation chain,
            compatible with untransformed nodes """
        return transformed_node

    ## standard Opt pass API
    def execute(self, optree):
        """ Implémentation of the standard optimization pass API """
        return self.transform_graph(optree)