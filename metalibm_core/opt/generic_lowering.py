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
# created:          May 29th, 2020
# last-modified:    May 29th, 2020
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################


from metalibm_core.core.ml_operations import is_leaf_node
from metalibm_core.core.passes import FunctionPass, METALIBM_PASS_REGISTER

from metalibm_core.code_generation.abstract_backend import GenericBackend

from metalibm_core.utility.log_report import Log

# Lowering mechanism:
#   The basic concept behind a LoweringAction is a function which transforms
#   a node into another node more suitable for implementation on the intended
#   target.
#
#   If a lowering action is found for a given node then the node is lowered
#   and the process is recursively called on the lowered node.
#   If no lowering action is found than the lowering considered the node inputs.
#
#   No lowering action cycle must exist.



class LoweringAction:
    """ base class for lowering a node """

    def lower_node(self, node):
        return node

class LoweringEngine:
    def __init__(self, target):
        self.target = target
        self.lowered_nodes = {}

    def lower_node(self, node):
        if node in self.lowered_nodes:
            return self.lowered_nodes[node]
        else:
            if not is_leaf_node(node):
                lowering_action = self.target.get_lowering_action(node, node.inputs)
            else:
                empty_inputs = (),
                lowering_action = self.target.get_lowering_action(node, empty_inputs)
            if lowering_action is None:
                self.lowered_nodes[node] = node
                if not is_leaf_node(node):
                    for op in node.inputs:
                        node.set_input(index, self.lower_node(op))
                    return node
            else:
                new_node = lowering_action(node)
                self.lowered_nodes[node] = new_node
                # recursive call
                # TODO/FIXME: implemented for simplicity, should be implemented
                #             flat, with no recursion
                return self.lower_node(new_node)

class GenericLoweringBackend(GenericBackend):
    def get_lowering_action(self, node, lowered_tuple_inputs, first_level_descriptor=None):
        """ processor generate expression """
        lowering_action = self.get_recursive_implementation(node, first_level_descriptor)
        return lowering_action

    @property
    def action_selection_table(self):
        """ semantic indirection between action_selection_table and
            code_generation_table """
        return self.lowering_table

LOG_LOWERING_INFO = Log.LogLevel("LoweringInfo")

@METALIBM_PASS_REGISTER
class Pass_GenericLowering(FunctionPass):
    """ implement generic lowering """
    pass_tag = "generic_lowering"
    def __init__(self, target, description="lower operation for target"):
        FunctionPass.__init__(self, description, target)

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        raise NotImplementedError

    def execute_on_graph(self, op_graph):
        lowering_engine = LoweringEngine(self.target)
        lowered_graph = lowering_engine.lower_node(op_graph)
        return lowered_graph

    def execute_on_function(self, fct, fct_group):
        """ execute generic lowering on function <fct> from group <fct_group>
        """
        Log.report(LOG_LOWERING_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        fct_scheme = fct.get_scheme()
        lowered_scheme = self.execute_on_graph(fct_scheme)
        fct.set_scheme(lowered_scheme)
