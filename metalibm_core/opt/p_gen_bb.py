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
# Desciption: set of optimization passes to:
#             - generate a Basic Block program representation from MDL's
#               dataflow graph
#             - perform SSA translation of the basic block CFG to single
#               static assignment form: (phi-node insertion, variable renaming)
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################

from metalibm_core.core.ml_operations import (
    Statement, ConditionBlock, Return, Loop,
    Variable, ReferenceAssign, ML_LeafNode,
    ControlFlowOperation, EmptyOperand,
    Constant
)

from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO

from metalibm_core.core.bb_operations import (
    BasicBlockList,
    ConditionalBranch, UnconditionalBranch, BasicBlock,
    PhiNode
)

from metalibm_core.utility.log_report import Log

# specific debug log levels for this file: (very) verbose and
# info(rmative)
LOG_LEVEL_GEN_BB_VERBOSE = Log.LogLevel("GenBBVerbose")
LOG_LEVEL_GEN_BB_INFO = Log.LogLevel("GenBBInfo")


def transform_cb_to_bb(bb_translator, optree, fct=None, fct_group=None, memoization_map=None):
    """ Transform a ConditionBlock node to a list of BasicBlock
        returning the entry block as result.

        @param bb_translator generic basic-block translator object
                             which can be called recursively on CB sub basic
                             blocks
        @param optree input operation tree
        @return entry BasicBlock to the ConditionBlock translation """
    # get current unfinished basic block (CB:) from top of stack
    #
    # When encoutering
    # if <cond>:
    #       if_branch
    # else:
    #       else_branch
    #
    # generates
    #  CB:
    #    ....
    #    br BB0
    #   BB0:
    #      cb <cond>, if_block
    #   if_block:
    #       if_branch
    #       br next
    #   else_block:     (if any)
    #       else_branch
    #       br next
    #   next:
    #
    #  next is pushed on top of stack
    entry_bb = bb_translator.get_current_bb()
    # at this point we can force addition to bb_list
    # as we known entry_bb will not be empty because of
    # ConditionalBranch appended at the end of this function
    bb_translator.pop_current_bb(force_add=True)

    cond = optree.get_input(0)
    if_branch = optree.get_input(1)
    if optree.get_input_num() > 2:
        else_branch = optree.get_input(2)
    else:
        else_branch = None
    # create new basic block to generate if branch
    if_entry = bb_translator.push_new_bb()
    if_bb = bb_translator.execute_on_optree(
        if_branch, fct, fct_group, memoization_map)
    assert if_bb == if_entry
    # pop bb created for if-branch
    if_end = bb_translator.pop_current_bb(force_add=True)

    # push new bb for else-branch or next
    if not else_branch is None:
        else_entry = bb_translator.push_new_bb()
        else_bb = bb_translator.execute_on_optree(
            else_branch, fct, fct_group, memoization_map)
        # end of bb for else branch
        else_end = bb_translator.pop_current_bb(force_add=True)
        # new bb for next block (only if else block not empty)
        next_bb = bb_translator.push_new_bb()
        add_to_bb(else_end, UnconditionalBranch(next_bb))
    else:
        else_entry = bb_translator.push_new_bb()

    next_bb = bb_translator.get_current_bb()
    # adding end of block instructions
    cb = ConditionalBranch(cond, if_entry, else_entry)
    add_to_bb(entry_bb, cb)
    # adding end of if block
    add_to_bb(if_end, UnconditionalBranch(next_bb))
    return entry_bb


def transform_loop_to_bb(
        bb_translator, optree, fct=None, fct_group=None, memoization_map=None):
    """ Transform a Loop node to a list of BasicBlock
        returning the entry block as result.

        @param bb_translator generic basic-block translator object
                             which can be called recursively on Loop's sub basic
                             blocks
        @param optree input operation tree
        @return entry BasicBlock to the Loop translation """
    init_statement = optree.get_input(0)
    loop_cond = optree.get_input(1)
    loop_body = optree.get_input(2)

    # loop pre-header
    init_block = bb_translator.execute_on_optree(
        init_statement, fct, fct_group, memoization_map)

    # building loop header
    bb_translator.push_new_bb("loop_header")
    loop_header_entry = bb_translator.get_current_bb()
    bb_translator.execute_on_optree(loop_cond, fct, fct_group, memoization_map)
    bb_translator.pop_current_bb()

    # building loop body
    bb_translator.push_new_bb("loop_body")
    body_bb = bb_translator.execute_on_optree(
        loop_body, fct, fct_group, memoization_map)

    add_to_bb(body_bb, UnconditionalBranch(loop_header_entry))
    bb_translator.pop_current_bb()
    bb_translator.push_new_bb("loop_exit")
    next_bb = bb_translator.get_current_bb()

    # loop header branch generation
    loop_branch = ConditionalBranch(loop_cond, body_bb, next_bb)
    add_to_bb(loop_header_entry, loop_branch)

    # lopp pre-header to header branch generation
    add_to_bb(init_block, UnconditionalBranch(loop_header_entry))

    # returning entry block
    return init_block

def get_recursive_op_graph_vars(node, memoization_map=None):
    """ return the list of unique variables appearing in the sub-grapj
        @p node """
    memoization_map = {} if memoization_map is None else memoization_map
    if node in memoization_map:
        return memoization_map[node]
    elif isinstance(node, Variable):
        result = set([node])
        memoization_map[node] = result
        return result
    elif isinstance(node, ML_LeafNode):
        result = set()
        memoization_map[node] = result
        return result
    result = set()
    memoization_map[node] = result
    result = result.union(
        sum([list(
            get_recursive_op_graph_vars(op, memoization_map)
        ) for op in node.get_inputs()], []))
    memoization_map[node] = result
    return result

def copy_up_to_variables(node):
    """ Create a copy of nodes without copying variables """
    var_list = get_recursive_op_graph_vars(node)
    return node.copy(dict((var, var) for var in var_list))

def is_leaf_or_cf(node):
    """ testing if @p node is an object of type ML_LeafNode or
        ControlFlowOperation """
    return isinstance(node, ML_LeafNode) or isinstance(node, ControlFlowOperation)

def sort_node_by_bb(bb_list):
    """ build a dictionnary mapping each node
        to a basic-block, also checks that each node
        only appears in a basic block """
    bb_map = {}
    for bb in bb_list.inputs:
        working_set = set()
        processed_nodes = set()
        for node in bb.get_inputs():
            working_set.add(node)
        while working_set:
            node = working_set.pop()
            if not node in processed_nodes:
                processed_nodes.add(node)
                if node in bb_map and bb_map[node] != bb:
                    if isinstance(node, Variable):
                        pass
                    elif isinstance(node, Constant):
                        # a constant may be copied
                        node = node.copy()
                    else:
                        # conflicting bb, copying until variable
                        node = copy_up_to_variables(node)
                bb_map[node] = bb
                if not is_leaf_or_cf(node):
                    for op in node.get_inputs():
                        working_set.add(op)
                elif isinstance(node, ConditionalBranch):
                    working_set.add(node.get_input(0))

    return bb_map

class CFGEdge(object):
    """ Control-Flow Graph edge (connecting the source
        and the destination of a branch instruction) """
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class BasicBlockGraph:
    """ Encapsulate several structure related
        to a basic block control flow graphs """
    def __init__(self, root, bb_list):
        # list of basic blocks
        self.root = root
        self.bb_list = bb_list
        self._bb_map = None
        self._variable_list = None
        self._variable_defs = None
        self._dominance_frontier_map = None
        self._dominator_map = None
        self._immediate_dominator_map = None
        self._cfg_edges = None
        self._dominator_tree = None

    @property
    def dominator_tree(self):
        """ return the dominator tree """
        if self._dominator_tree is None:
            self._dominator_tree = build_dominator_tree(
                self.immediate_dominator_map, self.root)
        return self._dominator_tree

    @property
    def bb_map(self):
        """ Basic Block map with lazy evaluation """
        if self._bb_map is None:
            self._bb_map = sort_node_by_bb(self.bb_list)
        return self._bb_map

    @property
    def variable_list(self):
        """ return the list of Variable nodes which appear in
            the basic block graph """
        if self._variable_list is None:
            var_list, var_defs = build_variable_list_and_defs(self.bb_list, self.bb_map)
            self._variable_list = var_list
            self._variable_defs = var_defs
        return self._variable_list

    @property
    def variable_defs(self):
        """ return a dict which associates to each variable the list of
            instructions where it is defined """
        if self._variable_defs is None:
            var_list, var_defs = build_variable_list_and_defs(self.bb_list, self.bb_map)
            self._variable_list = var_list
            self._variable_defs = var_defs
        return self._variable_defs

    @property
    def dominance_frontier_map(self):
        """ return a dict which associate to each node the list of nodes which
            constitute its dominance frontier """
        if self._dominance_frontier_map is None:
            self._dominance_frontier_map = get_dominance_frontiers(self)
        return self._dominance_frontier_map

    @property
    def cfg_edges(self):
        """ set of control flow edges of the basic block graph """
        if self._cfg_edges is None:
            self._cfg_edges = build_cfg_edges_set(self.bb_list)
        return self._cfg_edges

    @property
    def dominator_map(self):
        """ dict associating to each node the least of nodes which dominate it """
        if self._dominator_map is None:
            self._dominator_map = build_dominator_map(self)
            for bb in self._dominator_map:
                Log.report(
                    LOG_LEVEL_GEN_BB_VERBOSE,
                    "bb {}'s dominator: {}",
                    bb.get_tag(),
                    [dom.get_tag() for dom in self._dominator_map[bb]])
        return self._dominator_map

    @property
    def immediate_dominator_map(self):
        """ dict associating to each node its unique immediate dominator """
        if self._immediate_dominator_map is None:
            self._immediate_dominator_map = build_immediate_dominator_map(
                self.dominator_map, self.bb_list)
        return self._immediate_dominator_map

    def op_dominates(self, op0, op1):
        """ test if op0 dominates op1 """
        return self.bb_map[op0] in self.dominator_map[self.bb_map[op1]]



def build_cfg_edges_set(bb_list):
    cfg_edges = set()
    for bb in bb_list.inputs:
        if bb.empty:
            Log.report(Log.Error, "basic block {} is empty and does not have a last op", bb)
        last_op = bb.get_inputs()[-1]
        if isinstance(last_op, UnconditionalBranch):
            dst = last_op.get_input(0)
            cfg_edges.add(CFGEdge(bb, dst))
        elif isinstance(last_op, ConditionalBranch):
            true_dst = last_op.get_input(1)
            false_dst = last_op.get_input(2)
            cfg_edges.add(CFGEdge(bb, true_dst))
            cfg_edges.add(CFGEdge(bb, false_dst))
    return cfg_edges

def build_predominance_map_list(cfg_edges):
    """ build a dictionnary associating to each node the list of its
        predecessors """
    predominance_map_list = {}
    for edge in cfg_edges:
        if not edge.dst in predominance_map_list:
            predominance_map_list[edge.dst] = []
        predominance_map_list[edge.dst].append(edge.src)
    for bb in predominance_map_list:
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "bb {}'s predominance_map_list: {}".format(
                bb.get_tag(),
                [predom.get_tag() for predom in predominance_map_list[bb]]))
    return predominance_map_list

def build_dominator_map(bbg):
    """ Build a dict which associates to each node N
        a list of nodes which dominate N.
        (by definition this list contains at least N, since each node
        dominates itself) """
    predominance_map_list = build_predominance_map_list(bbg.cfg_edges)
    # building dominator map for each node
    dominator_map = {}
    for bb in bbg.bb_list.inputs:
        dominator_map[bb] = set(bbg.bb_list.get_inputs())
    dominator_map[bbg.root] = set([bbg.root])
    while True:
        change = False
        # dom(n) = fix point of intersect(dom(x), x in predecessor(n)) U {n}
        for bb in predominance_map_list:
            dom = set.union(
                set([bb]),
                set.intersection(
                    *tuple(dominator_map[pred] for pred in predominance_map_list[bb])
                )
            )
            if not bb in dominator_map:
                Log.report(Log.Error, "following bb was not in dominator_map: {}", bb)
            if dom != dominator_map[bb]:
                change = True
            dominator_map[bb] = dom
            Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "bb {}'s dominator list is {}",
                       bb.get_tag(), [dom.get_tag() for dom in dominator_map[bb]])
        if not change:
            break
    return dominator_map

def build_immediate_dominator_map(dominator_map, bb_list):
    """ Build a dictionnary associating the immediate dominator of each
        basic-block to this BB """
    # looking for immediate dominator
    immediate_dominator_map = {}
    for bb in bb_list.inputs:
        for dom in dominator_map[bb]:
            # skip self node
            if dom is bb: continue
            is_imm = True
            for other_dom in dominator_map[bb]:
                if other_dom is dom or other_dom is bb: continue
                if dom in dominator_map[other_dom]:
                    is_imm = False
                    break
            if is_imm:
                immediate_dominator_map[bb] = dom
                break
        if not bb in immediate_dominator_map:
            Log.report(LOG_LEVEL_GEN_BB_INFO,
                       "could not find immediate dominator for: \n{}", bb)
    return immediate_dominator_map


def does_not_strictly_dominate(dominator_map, x, y):
    """ predicate testing if @p x does not strictly dominate @p y using
        @p dominator_map """
    if x == y:
        return True
    elif x in dominator_map[y]:
        # y is dominated by x, and x != y so x strictly dominates y
        return False
    return True


def get_dominance_frontiers(bbg):
    """ compute the dominance frontier for each node in the basic-block group
        @p bbg """
    # build BB's conftrol flow graph
    dominance_frontier = {}
    for edge in bbg.cfg_edges:
        x = edge.src
        while does_not_strictly_dominate(bbg.dominator_map, x, edge.dst):
            if not x in dominance_frontier:
                dominance_frontier[x] = set()
            dominance_frontier[x].add(edge.dst)
            if not x in bbg.immediate_dominator_map:
                Log.report(Log.Error, "could not find {} in bbg.immediate_dominator_map", x)
            x = bbg.immediate_dominator_map[x]
    for x in dominance_frontier:
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "dominance_frontier of {} is {}",
            x.get_tag(),
            [bb.get_tag() for bb in dominance_frontier[x]])
    return dominance_frontier

def build_variable_list_and_defs(bb_list, bb_map):
    """ Build the list of Variable node appearing in the
        basic block list @p bb_list, also build a dict
        listing all definition of each variable """
    # listing variable and their definition
    variable_list = set()
    variable_defs = {}
    processed_nodes = set()
    working_set = set()
    for bb in bb_list.inputs:
        for node in bb.get_inputs():
            working_set.add(node)
    while working_set:
        node = working_set.pop()
        if not node in processed_nodes:
            processed_nodes.add(node)
            if isinstance(node, Variable):
                variable_list.add(node)
                if not node in variable_defs:
                    variable_defs[node] = set()
            elif isinstance(node, ReferenceAssign):
                var = node.get_input(0)
                if not var in variable_defs:
                    variable_defs[var] = set()
                variable_defs[var].add(bb_map[node])
            if not isinstance(node, ML_LeafNode):
                for op_input in node.get_inputs():
                    working_set.add(op_input)
    return variable_list, variable_defs


def phi_node_insertion(bbg):
    """ Perform first phase of SSA translation: insert phi-node
        where required """

    for v in bbg.variable_list:
        F = set() # set of bb where phi-node were added
        W = set() # set of bb which contains definition of v
        try:
            def_list = bbg.variable_defs[v]
        except KeyError as e:
            Log.report(Log.Error, "variable {} has no defs", v, error=e)
        for bb in def_list:
            #bb = bb_map[op]
            W.add(bb)
        while W: # non-empty set are evaluated to True
            x = W.pop()
            try:
                df = bbg.dominance_frontier_map[x]
            except KeyError:
                Log.report(
                    LOG_LEVEL_GEN_BB_VERBOSE,
                    "could not find dominance frontier for {}",
                    x.get_tag())
                df = []
            for y in df:
                # Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "y: {}", y)
                if not y in F:
                    # add phi-node x <- at entry of y
                    # TODO: manage more than 2 predecessor for v
                    phi_fct = PhiNode(v, v, None, v, None)
                    y.push(phi_fct)
                    # adding  phi funciton to bb map
                    bbg.bb_map[phi_fct] = y
                    F.add(y)
                    if not y in def_list:
                        W.add(y)


class DominatorTree(dict):
    """ DominatorTree object (dict + root) """
    def __init__(self, root=None):
        dict.__init__(self)
        self.root = root


def build_dominator_tree(immediate_dominator_map, root):
    """ generate the dominator tree map which associates to each node
        the list of nodes it immediately dominates.
        The tree is generated from the @p immediate_dominator_map which
        associate to each node its unique immediate dominator """
    # immediate_dominator_map Node -> immediate dominator (unique)
    dom_tree = DominatorTree(root)
    for imm_dominated in immediate_dominator_map:
        imm_dominator = immediate_dominator_map[imm_dominated]
        if not imm_dominator in dom_tree:
            dom_tree[imm_dominator] = []
        dom_tree[imm_dominator].append(imm_dominated)
    return dom_tree


def get_var_used_by_non_phi(node):
    """ return the list of unique variables used by operation @p node """
    if isinstance(node, BasicBlock):
        # does not traverse basick block
        return []
    elif isinstance(node, PhiNode):
        return []
    elif isinstance(node, Variable):
        return [node]
    elif isinstance(node, ML_LeafNode):
        return []
    elif isinstance(node, ReferenceAssign):
        return sum([get_var_used_by_non_phi(op_input) for op_input in node.get_inputs()[1:]], [])
    return sum([get_var_used_by_non_phi(op_input) for op_input in node.get_inputs()], [])

def update_used_var(op, old_var, new_var, memoization_map=None):
    """ Change occurence of @p old_var by @p new_var in Operation @p op """
    assert not op is None
    if memoization_map is None:
        memoization_map = {}

    Log.report(
        LOG_LEVEL_GEN_BB_VERBOSE,
        "update_var of op {} from {} to {}",
        op, old_var, new_var)

    if new_var is None:
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "skipping update_used_var as new_var is {}",
            new_var)
        return

    if op in memoization_map:
        return memoization_map[op]
    elif isinstance(op, BasicBlock):
        Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "skipping bb")
        return None
    elif op == old_var:
        if new_var is None:
            Log.report(
                Log.Error,
                "trying to update var {} to {}, maybe variable was used before def",
                old_var, new_var)
        memoization_map[op] = new_var
        return new_var
    elif isinstance(op, ML_LeafNode):
        memoization_map[op] = op
        return op
    else:
        # in-place swap
        memoization_map[op] = op
        for index, op_input in enumerate(op.get_inputs()):
            if isinstance(op, ReferenceAssign) and index == 0:
                # first input is in fact an output
                continue
            if op_input == old_var:
                if new_var is None:
                    Log.report(Log.Error, "trying to update var {} to {}, maybe variable was used before def", old_var, new_var)
                op.set_input(index, new_var)
            else:
                # recursion
                update_used_var(op_input, old_var, new_var, memoization_map)
        return op

def update_def_var(node, var, new_var):
    """ Update @p var which should be the variable defined by @p
        and replace it by @p vp """
    # TODO: manage sub-assignation cases
    assert isinstance(node, (ReferenceAssign, PhiNode))
    assert node.get_input(0) is var
    assert not new_var is None
    Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "updating var def in {} from {} to {}", node, var, new_var)
    node.set_input(0, new_var)

def updating_reaching_def(bbg, reaching_def, var, op):
    """ updating the @p reaching_def structure for Variable @p var
        assuming the reaching-definition must be the one valid
        at the location of operation @p op """
    # search through chain of definitions for var until it find
    # the closest definition that dominates op, then update
    # reaching_def[var] in-place with this definition
    current = reaching_def[var]
    while not (current == None or bbg.op_dominates(bbg.variable_defs[current], op)):
        current = reaching_def[current]
    if current is None:
        Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "skipping updating_reaching_def from {} to {}", reaching_def[var], current)
        return False
    Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "reaching_def of var {} from op {} updated from {} to {}", var, op, reaching_def[var], current)
    change = (current != reaching_def[var])
    reaching_def[var] = current
    return change


def get_var_def_by_op(op):
    """ return the list of variable defined by op (if any) """
    if isinstance(op, PhiNode):
        return [op.get_input(0)]
    elif isinstance(op, ReferenceAssign):
        return [op.get_input(0)]
    return []

def get_phi_list_in_bb_successor(bb):
    """ return the list of PhiNode which appears in the successors of bb """
    phi_list = []
    for successor in bb.successors:
        for op in successor.get_inputs():
            if isinstance(op, PhiNode):
                phi_list.append(op)
    return phi_list

def update_indexed_used_var_in_phi(op, index, old_var, new_var, pred_bb):
    """ Update the Variable @p old_var used by PhiNode @p op at operand index
        @p index by new_var. as op is a phi-node, we also need to update
        the basic-block @p pred_bb associated to @p new_var """
    assert isinstance(op, PhiNode)
    if new_var is None:
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "skipping update_indexed_used_var_in_phi from {} to {}",
            old_var, new_var)
    else:
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "updating indexed used var in phi from {} to {}",
            old_var, new_var)
        op.set_input(index, new_var)
        op.set_input(index + 1, pred_bb)

def get_indexed_var_used_by_phi(phinode):
    """ return a list of (index, node) corresponding to the occurences
        of Variable nodes in the list of used-input of @p phinode """
    assert isinstance(phinode, PhiNode)
    return [(2 * index + 1, op, phinode.get_input(2 * index + 2)) for index, op in enumerate(phinode.get_inputs()[1::2]) if isinstance(op, Variable)]


def variable_renaming(bbg):
    """ Perform second stage of SSA transformation: variable renaming """
    # dict Variable -> definition
    reaching_def = dict((var, None) for var in bbg.variable_list)

    var_index = {}
    def new_var_index(var):
        """ return a free numerical index to associate to a new instance
            of @p var """
        if not var in var_index:
            var_index[var] = 0
        new_index = var_index[var]
        var_index[var] = new_index + 1
        return new_index

    memoization_map = {}

    def rec_bb_processing(bb):
        """ perform variable renaming in the basic block @p bb
            and recursivly in bb's children in the dominator tree """
        Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "processing bb {}", bb)
        # because a node can be duplicated between
        # its declaration and its use in a subsequent operation in the same
        # basic block, we must make sure it is processed only once
        # by update_used_var for a given <var>. Thus for each
        # <var> we store a memoization_map of processed nodes
        updated_used_var_memoization_map = {}
        def get_mem_map(var):
            """ return the updated_used_var memoization_map associated
                to @p var """
            if not var in updated_used_var_memoization_map:
                updated_used_var_memoization_map[var] = {}
            return updated_used_var_memoization_map[var]
        for op in bb.get_inputs():
            if op in memoization_map:
                continue
            else:
                memoization_map[op] = None
            Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "processing op {}", op)
            if not isinstance(op, PhiNode):
                for var in get_var_used_by_non_phi(op):
                    Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "processing var {} used by non-phi node", var)
                    updating_reaching_def(bbg, reaching_def, var, op)
                    Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "updating var from {} to {} used by non-phi node", var, reaching_def[var])
                    local_mem_map = get_mem_map(var)
                    update_used_var(op, var, reaching_def[var], memoization_map=local_mem_map)
                    # to avoid multiple update we add the output memoization_table
                    # to the table of the destination variable
                    # so the last time the destination variable is considered for update
                    # it will discard all update made during this BB processing
                    get_mem_map(reaching_def[var]).update(local_mem_map)
            for var in get_var_def_by_op(op):
                updating_reaching_def(bbg, reaching_def, var, op)
                vp = Variable("%s_%d" % (var.get_tag(), new_var_index(var)), precision=var.get_precision()) # TODO: tag
                update_def_var(op, var, vp)
                reaching_def[vp] = reaching_def[var]
                reaching_def[var] = vp
                bbg.variable_defs[vp] = op
        Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "processing phi in successor")
        for phi in get_phi_list_in_bb_successor(bb):
            for index, var, var_bb in get_indexed_var_used_by_phi(phi):
                Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "processing operand #{} of phi: {}, var_bb is {}", index, var, var_bb)
                if not isinstance(var_bb, EmptyOperand):
                    continue
                # updating_reaching_def(bbg, reaching_def, var, phi)
                update_indexed_used_var_in_phi(phi, index, var, reaching_def[var], bb)
                break
        # finally traverse sub-tree
        if bb in bbg.dominator_tree:
            for child in bbg.dominator_tree[bb]:
                rec_bb_processing(child)

    # processing dominator tree in depth-first search pre-order
    rec_bb_processing(bbg.dominator_tree.root)

def add_to_bb(bb, node):
    """ add operation node @p node to the end of the basic-block @p bb """
    if not bb.final and (bb.empty or not isinstance(bb.get_input(-1), ControlFlowOperation)):
        bb.add(node)

class Pass_GenerateBasicBlock(FunctionPass):
    """ pre-Linearize operation tree to basic blocks
        Control flow construction are transformed into linked basic blocks
        Dataflow structure of the operation graph is kept for unambiguous
        construct (basically non-control flow nodes) """
    pass_tag = "gen_basic_block"
    def __init__(self, target, description = "generate basic-blocks pass"):
        FunctionPass.__init__(self, description, target)
        self.memoization_map = {}
        self.top_bb_list = None
        self.current_bb_stack = []
        self.bb_tag_index = 0

    def set_top_bb_list(self, bb_list):
        """ define the top basic block and reset the basic block stack so it
            only contains this block """
        self.top_bb_list = bb_list
        self.current_bb_stack = [self.top_bb_list.entry_bb]
        return self.top_bb_list

    def push_to_current_bb(self, node):
        """ add a new node at the end of the current basic block
            which is the topmost on the BB stack """
        assert self.current_bb_stack
        self.current_bb_stack[-1].push(node)

    def get_new_bb_tag(self, tag):
        if tag is None:
            tag = "bb_%d" % self.bb_tag_index
            self.bb_tag_index +=1
        return tag

    @property
    def top_bb(self):
        return self.current_bb_stack[-1]

    def push_new_bb(self, tag=None):
        # create the new basic-block
        tag = self.get_new_bb_tag(tag)
        new_bb = BasicBlock(tag=tag)
        self.top_bb_list.add(new_bb)

        # register new basic-block at the end of the current list
        # self.top_bb_list.add(new_bb)

        # set the new basic block at the top of the bb stack
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "appending new bb to stack: ", new_bb.get_tag())
        self.current_bb_stack.append(new_bb)
        return new_bb


    def pop_current_bb(self, force_add=False):
        """ remove the topmost basic block from the BB stack
            and add it to the list of basic blocks """
        if len(self.current_bb_stack) >= 1:
            top_bb = self.current_bb_stack[-1]
            Log.report(
                LOG_LEVEL_GEN_BB_VERBOSE,
                "poping top_bb {} from stack ",
                top_bb.get_tag())
            # TODO/FIXME: fix bb regsiterting in top_bb_list testing
            top_bb = self.current_bb_stack.pop(-1)
            return top_bb
        Log.report(LOG_LEVEL_GEN_BB_VERBOSE, "   top_bb was empty")
        return None

    def flush_bb_stack(self):
        """ Pop all basic-blocks from the stack except the last (function
            enttry point) """
        # flush all but last bb (which is root/main)
        while len(self.current_bb_stack) > 1:
            assert self.pop_current_bb()
        Log.report(
            LOG_LEVEL_GEN_BB_VERBOSE,
            "bb_stack after flush: ",
            [bb.get_tag() for bb in self.current_bb_stack])

    def get_current_bb(self):
        """ return the basic-block at the top of the stack """
        return self.current_bb_stack[-1]

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        """ return the head basic-block, i.e. the entry bb for the current node
            implementation """
        assert not isinstance(optree, BasicBlock)
        entry_bb =  self.get_current_bb()
        if isinstance(optree, ConditionBlock):
            entry_bb = transform_cb_to_bb(self, optree)
        elif isinstance(optree, Loop):
            entry_bb = transform_loop_to_bb(self, optree)
        elif isinstance(optree, Return):
            # return must be processed separately as it finishes a basic block
            self.push_to_current_bb(optree)
            self.get_current_bb().final = True

        elif isinstance(optree, Statement):
            for op in optree.get_inputs():
                self.execute_on_optree(op, fct, fct_group, memoization_map)
        else:
            self.push_to_current_bb(optree)
        return entry_bb


    def execute_on_function(self, fct, fct_group):
        """ execute basic-block generation pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_LEVEL_GEN_BB_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        optree = fct.get_scheme()
        memoization_map = {}
        new_bb = BasicBlock(tag="main")
        bb_list = BasicBlockList(tag="main")
        bb_list.entry_bb = new_bb
        top_bb_list = self.set_top_bb_list(bb_list)
        last_bb = self.execute_on_optree(optree, fct, fct_group, memoization_map)
        # pop last basic-block (to add it to the list)
        self.flush_bb_stack()
        fct.set_scheme(top_bb_list)


class Pass_BBSimplification(FunctionPass):
    """ Simplify BB graph """
    pass_tag = "basic_block_simplification"

    def __init__(self, target, description = "simplify basic-blocks pass"):
        FunctionPass.__init__(self, description, target)
        self.memoization_map = {}

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        """ eliminate every basic-block from BasicBlockList @p optree which is
            not accessible from @p optree's entry point """
        assert isinstance(optree, BasicBlockList)
        memoization_map = {}
        bb_root = optree.entry_bb

        # determining the list of bb which are not accessible from bb_root
        all_bbs = set(optree.get_inputs())
        accessible_bbs = []

        processed_list = []
        work_list = [bb_root]
        accessible_bbs = [bb_root]
        while work_list:
            current_bb = work_list.pop(0)

            processed_list.append(current_bb)
            for next_bb in current_bb.successors:
                if not next_bb in processed_list:
                    work_list.append(next_bb)
                    accessible_bbs.append(next_bb)

        new_bb_list = accessible_bbs
        optree.inputs = new_bb_list


    def execute_on_function(self, fct, fct_group):
        """ execute basic-block simplification pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_LEVEL_GEN_BB_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        bb_list = fct.get_scheme()
        self.execute_on_optree(bb_list, fct, fct_group)



class Pass_SSATranslate(FunctionPass):
    """ Translate basic-block into  Single Static Assignement form """
    pass_tag = "ssa_translation"
    def __init__(self, target, description="translate basic-blocks into ssa form pass"):
        FunctionPass.__init__(self, description, target)
        self.memoization_map = {}
        self.top_bb_list = None


    def execute_on_function(self, fct, fct_group):
        """ Execute SSA translation pass on function @p fct from
            function-group @p fct_group """
        Log.report(LOG_LEVEL_GEN_BB_INFO, "executing pass {} on fct {}".format(
            self.pass_tag, fct.get_name()))
        optree = fct.get_scheme()
        assert isinstance(optree, BasicBlockList)
        bb_root = optree.get_input(0)
        bbg = BasicBlockGraph(bb_root, optree)
        phi_node_insertion(bbg)
        variable_renaming(bbg)

# registering basic-block generation pass
Log.report(LOG_PASS_INFO, "Registering generate Basic-Blocks pass")
Pass.register(Pass_GenerateBasicBlock)
# registering ssa translation pass
Log.report(LOG_PASS_INFO, "Registering ssa translation pass")
Pass.register(Pass_SSATranslate)
# registering basic-block simplification pass
Log.report(LOG_PASS_INFO, "Registering basic-block simplification pass")
Pass.register(Pass_BBSimplification)

if __name__ == "__main__":
    bb_root = BasicBlock(tag="bb_root")
    bb_1 = BasicBlock(tag="bb_1")
    bb_2 = BasicBlock(tag="bb_2")
    bb_3 = BasicBlock(tag="bb_3")

    var_x = Variable("x", precision=None)
    var_y = Variable("y", precision=None)

    bb_root.add(ReferenceAssign(var_x, 1))
    bb_root.add(ReferenceAssign(var_y, 2))
    bb_root.add(ConditionalBranch(var_x > var_y, bb_1, bb_2))

    bb_1.add(ReferenceAssign(var_x, 2))
    bb_1.add(UnconditionalBranch(bb_3))

    bb_2.add(ReferenceAssign(var_y, 3))
    bb_2.add(UnconditionalBranch(bb_3))

    bb_3.add(ReferenceAssign(var_y, var_x))


    program_bb_list = BasicBlockList(tag="main")
    for bb in [bb_root, bb_1, bb_2, bb_3]:
        program_bb_list.add(bb)

    print(program_bb_list.get_str(depth=None))

    BBG = BasicBlockGraph(bb_root, program_bb_list)

    phi_node_insertion(BBG)

    print("after phi node insertion")
    print(program_bb_list.get_str(depth=None))

    variable_renaming(BBG)
    print("after variable renaming")
    print(program_bb_list.get_str(depth=None))
