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
# Created:         Oct  8th 2018
# Last-modified:   Oct  8th 2018
# Author(s):       Nicolas Brunie (nbrunie@kalray.eu)
#
# Desciprion:  Generic pass to perform node transformation and update the
#              operation graph based on pre-defined rules
###############################################################################

from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_operations import is_leaf_node

from metalibm_core.utility.log_report import Log


LOG_LEVEL_NODE_TRANSFORM_VERBOSE = Log.LogLevel("NodeTransformVerbose")


class Pass_NodeTransformation(FunctionPass):
    """ This pass can not be used as-is, it must be overloaded """
    pass_tag = "node_transformation"

    def __init__(self, target=None):
        FunctionPass.__init__(self, target=target)
        self.memoization_map = {}

    def get_memoization_key(self, node, *args):
        return node

    def has_memoization(self, node, *args):
        """ Return True if node as a corresponding entry in the memoization map """
        return self.get_memoization_key(node, *args) in self.memoization_map

    def get_memoization_value(self, node, *args):
        node_key = self.get_memoization_key(node, *args)
        try:
            return self.memoization_map[node_key]
        except KeyError:
            Log.report(
                Log.Error,
                "unable to found key {} in {} memoization map",
                node_key,
                self
            )
    def set_memoization_value(self, node, value, *args):
        """ define the value associated with @p node in the memoization table """
        self.memoization_map[self.get_memoization_key(node, *args)] = value

    def can_be_transformed(self, node, *args):
        """ return True if node can be transformed, False otherwise """
        raise NotImplementedError

    def transform_node(self, node, transformed_inputs, *args):
        """ If node can be transformed returns the transformed node
            else returns None """
        raise NotImplementedError

    def reconstruct_from_transformed(self, op_input, transformed_node):
        """return a node at the root of a transformation chain,
            compatible with untransformed nodes """
        raise NotImplementedError


    def transform_graph(self, node, *args, root=False):
        if self.has_memoization(node):
            return self.get_memoization_value(node)
        elif not self.can_be_transformed(node):
            # associate None to node in memoization_map
            Log.report(LOG_LEVEL_NODE_TRANSFORM_VERBOSE, "associating None to {}", node)
            self.set_memoization_value(node, None)
            # recursively process node's inputs
            if not is_leaf_node(node):
                for index, op_input in enumerate(node.get_inputs()):
                    new_input = self.transform_graph(op_input)
                    if not new_input is None:
                        reconstructed_input = self.reconstruct_from_transformed(op_input, new_input)
                        Log.report(LOG_LEVEL_NODE_TRANSFORM_VERBOSE, "new input id={} of {} is {}", index, node, reconstructed_input)
                        node.set_input(index, reconstructed_input)
                # if this call was made on the graph root then the
                # node is returned unmodified else None is returned to
                # indication no modification were made
                return node if root else None
            return node if root else None
        else:
            transformed_inputs = [self.transform_graph(op_input) for op_input in node.get_inputs()]
            new_node = self.transform_node(node, transformed_inputs)
            self.set_memoization_value(node, new_node)
            return new_node

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        # TODO/FIXME: not using memoization_map may lead to inefficiencies and duplicate
        #             node results
        return self.transform_graph(optree, root=True)




