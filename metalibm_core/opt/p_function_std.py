# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

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

import collections
from sollya import inf, sup

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import (
    ML_LeafNode, Constant, Variable,
    Statement, ConditionBlock, ReferenceAssign,
    ConditionBlock, SwitchBlock, Negation,
    Addition, FusedMultiplyAdd, Subtraction,
    Multiplication,
    Return, TableLoad,
    Conversion, DivisionSeed,
    ControlFlowOperation,
    is_leaf_node,
)
from metalibm_core.core.ml_hdl_operations import (
    Process, Loop, ComponentInstance, Assert, Wait, PlaceHolder
)
from metalibm_core.core.passes import (
    Pass, LOG_PASS_INFO, FunctionPass
)
from metalibm_core.core.bb_operations import (
    UnconditionalBranch, ConditionalBranch, BasicBlock, PhiNode
)



def simplify_inverse(optree, processor):
    dummy_var = Variable("dummy_var_seed", precision = optree.get_precision())
    dummy_div_seed = DivisionSeed(dummy_var, precision = optree.get_precision())
    inv_approx_table = processor.get_recursive_implementation(dummy_div_seed, language = None, table_getter = lambda self: self.approx_table_map)

    seed_input = optree.inputs[0]
    c0 = Constant(0, precision = ML_Int32)

    if optree.get_precision() == inv_approx_table.get_storage_precision():
        return TableLoad(inv_approx_table, inv_approx_table.get_index_function()(seed_input), c0, precision = optree.get_precision()) 
    else:
        return Conversion(TableLoad(inv_approx_table, inv_approx_table.get_index_function()(seed_input), c0, precision = inv_approx_table.get_storage_precision()), precision = optree.get_precision()) 


support_simplification = {
    #FusedMultiplyAdd: {
    #    FusedMultiplyAdd.Standard: {
    #        lambda optree: True: 
    #            lambda optree, processor: Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()),
    #    },
    #    FusedMultiplyAdd.Subtract: {
    #        lambda optree: True: 
    #            lambda optree, processor: Subtraction(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()),
    #    },
    #    FusedMultiplyAdd.Negate: {
    #        lambda optree: True: 
    #            lambda optree, processor: Negate(Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), optree.inputs[2], precision = optree.get_precision()), precision = optree.get_precision()),
    #    },
    #    FusedMultiplyAdd.SubtractNegate: {
    #        lambda optree: True: 
    #            lambda optree, processor: Subtraction(optree.inputs[2], Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), precision = optree.get_precision()),
    #    },
    #    FusedMultiplyAdd.DotProduct: {
    #        lambda optree: True:
    #            lambda optree, processor: Addition(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), Multiplication(optree.inputs[2], optree.inputs[3], precision = optree.get_precision()), precision = optree.get_precision()), 
    #    },
    #    FusedMultiplyAdd.DotProductNegate: {
    #        lambda optree: True: 
    #            lambda optree, processor: Subtraction(Multiplication(optree.inputs[0], optree.inputs[1], precision = optree.get_precision()), Multiplication(optree.inputs[2], optree.inputs[3], precision = optree.get_precision()), precision = optree.get_precision()), 
    #    },
    #},
    Subtraction: {
      None: {
        lambda optree: True: 
          lambda optree, processor: Addition(optree.inputs[0], Negation(optree.inputs[1], precision = optree.inputs[1].get_precision()), precision = optree.get_precision())
      },
    },
    DivisionSeed: {
        None: {
            lambda optree: True:
                simplify_inverse,
        },
    },
}


def silence_fp_operations(optree, force=False, memoization_map=None):
    """ ensure that all floating-point operations from optree root
        have the silent attribute set to True """
    memoization_map = {} if memoization_map is None else memoization_map
    if optree in memoization_map:
        return
    else:
        memoization_map[optree] = optree
        if isinstance(optree, ML_LeafNode):
            pass
        else:
            for op in optree.inputs:
                silence_fp_operations(op, force=force, memoization_map=memoization_map)
            for op in optree.get_extra_inputs():
                silence_fp_operations(op, force = force, memoization_map = memoization_map)
            if isinstance(optree, Multiplication) or isinstance(optree, Addition) or isinstance(optree, FusedMultiplyAdd) or isinstance(optree, Subtraction):
                # FIXME no check on optree precision
                if optree.get_silent() == None: optree.set_silent(True)

def fuse_multiply_add(optree, silence=False, memoization=None, change_handle=True, dot_product_enabled=True):
    """ whenever possible fuse a multiply and add/sub into a FMA/FMS """
    memoization = memoization if not memoization is None else {}
    def local_fuse_fma(op):
        return fuse_multiply_add(op, silence, memoization, change_handle, dot_product_enabled)
    if (isinstance(optree, Addition) or isinstance(optree, Subtraction)) and not optree.get_unbreakable():
        if len(optree.inputs) != 2:
            # more than 2-operand addition are not supported yet
            optree.inputs = tuple(local_fuse_fma(op) for op in optree.inputs)
            return optree

        else:
            if optree in memoization: 
                return memoization[optree]

            elif optree.get_unbreakable():
                optree.inputs = tuple(local_fuse_fma(op) for op in optree.inputs)
                memoization[optree] = optree
                return optree

            elif True in [(op.get_debug() != None and isinstance(op, Multiplication)) for op in optree.inputs]:
                # exclude node with debug operands
                optree.inputs = tuple(local_fuse_fma(op) for op in optree.inputs)
                memoization[optree] = optree
                return optree

            elif dot_product_enabled and isinstance(optree.inputs[0], Multiplication) and isinstance(optree.inputs[1], Multiplication) and not optree.inputs[0].get_prevent_optimization() and not optree.inputs[1].get_prevent_optimization():
                specifier = FusedMultiplyAdd.DotProductNegate if isinstance(optree, Subtraction) else FusedMultiplyAdd.DotProduct 
                mult0 = local_fuse_fma(optree.inputs[0].inputs[0])
                mult1 = local_fuse_fma(optree.inputs[0].inputs[1])
                mult2 = local_fuse_fma(optree.inputs[1].inputs[0])
                mult3 = local_fuse_fma(optree.inputs[1].inputs[1])
                new_op = FusedMultiplyAdd(mult0, mult1, mult2, mult3, specifier = specifier)
                new_op.attributes = optree.attributes.get_light_copy()
                new_op.set_silent(silence)
                new_op.set_index(optree.get_index())
                # propagating exact attribute
                if optree.inputs[0].get_exact() and optree.inputs[1].get_exact() and optree.get_exact():
                    new_op.set_exact(True)
                # modifying handle
                if change_handle: optree.get_handle().set_node(new_op)
                memoization[optree] = new_op
                return new_op

            elif isinstance(optree.inputs[0], Multiplication) and not optree.inputs[0].get_prevent_optimization():
                specifier = FusedMultiplyAdd.Subtract if isinstance(optree, Subtraction) else FusedMultiplyAdd.Standard 
                mult0 = local_fuse_fma(optree.inputs[0].inputs[0])
                mult1 = local_fuse_fma(optree.inputs[0].inputs[1])
                addend = local_fuse_fma(optree.inputs[1])

                new_op = FusedMultiplyAdd(mult0, mult1, addend, specifier = specifier)
                new_op.attributes = optree.attributes.get_light_copy()
                new_op.set_silent(silence)
                new_op.set_index(optree.get_index())

                # propagating exact attribute
                if optree.inputs[0].get_exact() and optree.get_exact():
                    new_op.set_exact(True)

                # modifying handle
                if change_handle: optree.get_handle().set_node(new_op)

                memoization[optree] = new_op
                return new_op

            elif isinstance(optree.inputs[1], Multiplication) and not optree.inputs[1].get_prevent_optimization():
                specifier = FusedMultiplyAdd.SubtractNegate if isinstance(optree, Subtraction) else FusedMultiplyAdd.Standard 
                mult0 = local_fuse_fma(optree.inputs[1].inputs[0])
                mult1 = local_fuse_fma(optree.inputs[1].inputs[1])
                addend = local_fuse_fma(optree.inputs[0])
                new_op = FusedMultiplyAdd(mult0, mult1, addend, specifier = specifier)
                new_op.attributes = optree.attributes.get_light_copy()
                new_op.set_silent(silence)
                new_op.set_commutated(True)
                memoization[optree] = new_op

                new_op.set_index(optree.get_index())
                # propagating exact attribute
                if optree.inputs[1].get_exact() and optree.get_exact():
                    new_op.set_exact(True)

                # modifying handle
                if change_handle: optree.get_handle().set_node(new_op)

                return new_op
            else:
                optree.inputs = tuple(local_fuse_fma(op) for op in optree.inputs)
                memoization[optree] = optree
                return optree
    else:
        if optree.get_extra_inputs() != []: 
            optree.set_extra_inputs([local_fuse_fma(op) for op in optree.get_extra_inputs()])

        if isinstance(optree, ML_LeafNode):
            memoization[optree] = optree
            return optree
        else:
            optree.inputs = tuple(local_fuse_fma(op) for op in optree.inputs)
            memoization[optree] = optree
            return optree


def subexpression_sharing(root_node):
    """ reseach and factorize sub-graphs between ConditionBlock, Loop,
        SwitchBlock branches """
    def search_level_map(optree):
        """ search if optree has been defined among the active node """
        for level in level_sharing_map:
            if optree in level: return True
        return False

    def common_ancestor(parent_list_0, parent_list_1):
        """ search the closest node of parent_list_0,
            also registered in parent_list_1 """
        for b in parent_list_0[::-1]:
            if b in parent_list_1:
                return b
        return None

    def contains_non_input_variable(node, memoization_map=None):
        """ Test if the graph whose root is node contains any non-input variable
            which may indicate that those variables may be modified
            unexpectedly and makes the node factorizing hazardous """
        memoization_map = {} if memoization_map is None else memoization_map
        if node in memoization_map:
            return memoization_map[node]
        if isinstance(node, Variable) and node.var_type != Variable.Input:
            memoization_map[node] = True
            return True
        elif is_leaf_node(node):
            memoization_map[node] = False
            return False
        else:
            result = any(contains_non_input_variable(op, memoization_map) for op in node.inputs)
            memoization_map[node] = result
            return result

    non_input_variable_memoization = {}


    def recursive_ancestor_lookup(optree, sharing_map=None, level_sharing_map=None,
                                  current_parent_list=None, ancestor_map=None):
        if isinstance(optree, ConditionBlock):
            # condition
            recursive_ancestor_lookup(optree.inputs[0], sharing_map, level_sharing_map, current_parent_list + [optree], ancestor_map=ancestor_map)
            # branches
            for op in optree.inputs[1:]:
                recursive_ancestor_lookup(op, sharing_map, [{}] + level_sharing_map, current_parent_list + [optree], ancestor_map=ancestor_map)

        elif isinstance(optree, SwitchBlock):
            # switch value
            recursive_ancestor_lookup(optree.inputs[0], sharing_map, level_sharing_map, current_parent_list + [optree], ancestor_map=ancestor_map)
            # case_statement
            case_map = optree.get_case_map()
            for case in case_map:
                op = case_map[case]
                recursive_ancestor_lookup(op, sharing_map, [{}] + level_sharing_map, current_parent_list + [optree], ancestor_map=ancestor_map)

        elif isinstance(optree, Statement):
            if not optree.get_prevent_optimization():
              for op in optree.inputs:
                  recursive_ancestor_lookup(op, sharing_map, [{}] + level_sharing_map, current_parent_list, ancestor_map=ancestor_map)

        elif isinstance(optree, Loop):
            pass

        elif isinstance(optree, ML_LeafNode):
            pass
        else:
            if optree in sharing_map:
                if not search_level_map(optree):
                    # parallel branch sharing possibility
                    ancestor = common_ancestor(sharing_map[optree], current_parent_list)
                    if ancestor != None:
                        ancestor_map[ancestor].append(optree)
            elif not contains_non_input_variable(optree, non_input_variable_memoization):
                # only considers node which does not depend on non-input variables
                sharing_map[optree] = current_parent_list
                level_sharing_map[0][optree] = current_parent_list
                for op in optree.inputs:
                    recursive_ancestor_lookup(op, sharing_map, level_sharing_map, current_parent_list, ancestor_map=ancestor_map)
            else:
                pass

    # init
    sharing_map = {} 
    level_sharing_map = [{}] 
    current_parent_list = []
    ancestor_map = collections.defaultdict(list)
    recursive_ancestor_lookup(root_node, sharing_map, level_sharing_map,
                              current_parent_list, ancestor_map)
    insert_pre_statement(root_node, ancestor_map)

## Generic vector promotion pass
class PassSilenceFPOperation(FunctionPass):
    """ Silence all floating-point operations """
    pass_tag = "silence_fp_ops"
    def __init__(self, target, force=False):
        FunctionPass.__init__(self, "silence_fp_ops")
        self.memoization_map = {}
        self.force = force

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        """ Execute silence fp op pass on optree """
        memoization_map = memoization_map if not memoization_map is None else {}
        silence_fp_operations(scheme, self.force, memoization_map)
        return optree


class PassFuseFMA(FunctionPass):
    """ Fuse floating-point MAC into FMA"""
    pass_tag = "fuse_fma"
    def __init__(self, target, change_handle=True, dot_product_enabled=False, silence=False):
        FunctionPass.__init__(self, "fuse_fma")
        self.memoization_map = {}
        self.change_handle = change_handle
        self.dot_product_enabled = dot_product_enabled
        self.silence = silence


    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        memoization_map = memoization_map if not memoization_map is None else {}
        return fuse_multiply_add(
            optree, self.silence, memoization_map, self.change_handle,
            self.dot_product_enabled)

def insert_pre_statement(node, ancestor_map):
    """ ancestor_map contains a list of node associated to each key (ancesstor)
        this function, starting from root node, insert in the statement
        surrounding ancestor a decleration for each node in the list """
    if isinstance(node, Statement):
        result_input_list = []
        for op in node.inputs:
            if op in ancestor_map:
                result_input_list = result_input_list + [pre_node for pre_node in ancestor_map[op]]
            insert_pre_statement(op, ancestor_map)
            result_input_list.append(op)
        node.inputs = result_input_list
    elif isinstance(node, ControlFlowOperation):
        for op in node.inputs:
            insert_pre_statement(op, ancestor_map)
    else:
        # non control flow operation
        pass


class PassSubExpressionSharing(FunctionPass):
    """ Factorize sub-expression shared between control flow branches """
    pass_tag = "sub_expr_sharing"
    def __init__(self, target):
        FunctionPass.__init__(self, "sub_expr_sharing")
        self.memoization_map = {}

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        subexpression_sharing(optree)
        return optree

def has_support_simplification(optree):
    """ check if node optree can be simplified """
    if optree.__class__ in support_simplification:
        code_gen_key = optree.get_codegen_key()
        if code_gen_key in support_simplification[optree.__class__]:
            for cond in support_simplification[optree.__class__][code_gen_key]:
              if cond(optree): return True
    return False

def get_support_simplification(optree, processor):
    """ retrieve support simplified version of optree"""
    code_gen_key = optree.get_codegen_key()
    for cond in support_simplification[optree.__class__][code_gen_key]:
        if cond(optree):
            return support_simplification[optree.__class__][code_gen_key][cond](optree, processor)
    Log.report(Log.Error, "support simplification mapping not found")


class PassCheckProcessorSupport(FunctionPass):
    """ Check if the intended target supports every operation appearing in the graph """
    pass_tag = "check_processor_support"
    def __init__(self, target, language=C_Code, debug=False):
        FunctionPass.__init__(self, "check_processor_support")
        self.processor = target
        self.language = language
        self.debug = debug

    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None): 
        memoization_map = memoization_map if not memoization_map is None else {}
        Log.report(Log.Info, "executing check-processor with target {}".format(self.processor))
        check_result = self.check_processor_support(
            self.processor,
            optree, memoization_map, self.debug, self.language
        )
        return optree

    @staticmethod
    def check_processor_support(processor, root, memoization_map=None, debug=False, language=C_Code):
        """ check if all precision-instantiated operation are supported by the processor """
        memoization_map = memoization_map if not memoization_map is None else {}

        Log.report(Log.Info, "checking processor support: {}", processor.__class__)

        def recursive_support_check(optree):
            if  optree in memoization_map:
                return True

            elif not isinstance(optree, ML_LeafNode):
                # memoization
                memoization_map[optree] = True
                for inp in optree.inputs:
                    recursive_support_check(inp)

                if isinstance(optree, ConditionBlock):
                    pass
                elif isinstance(optree, Statement):
                    pass
                elif isinstance(optree, ConditionalBranch):
                    pass
                elif isinstance(optree, UnconditionalBranch):
                    pass
                elif isinstance(optree, BasicBlock):
                    pass
                elif isinstance(optree, PhiNode):
                    pass
                elif isinstance(optree, Loop):
                    pass
                elif isinstance(optree, Return):
                    pass
                elif isinstance(optree, ReferenceAssign):
                    pass
                elif isinstance(optree, PlaceHolder):
                    pass
                elif isinstance(optree, SwitchBlock):

                    for op in optree.get_extra_inputs():
                      # TODO: assert case is integer constant
                      recursive_support_check(op)
                elif not processor.is_supported_operation(optree, debug = debug, language = language):
                    # trying operand format escalation
                    init_optree = optree
                    old_list = optree.inputs
                    while False: #optree.__class__ in type_escalation:
                        match_found = False
                        for result_type_cond in type_escalation[optree.__class__]:
                            if result_type_cond(optree.get_precision()): 
                                for op_index in range(len(optree.inputs)):
                                    op = optree.inputs[op_index]
                                    for op_type_cond in type_escalation[optree.__class__][result_type_cond]:
                                        if op_type_cond(op.get_precision()): 
                                            new_type = type_escalation[optree.__class__][result_type_cond][op_type_cond](optree) 
                                            if op.get_precision() != new_type:
                                                # conversion insertion
                                                input_list = list(optree.inputs)
                                                input_list[op_index] = Conversion(op, precision = new_type)
                                                optree.inputs = tuple(input_list)
                                                match_found = True
                                                break
                                break
                        if not match_found:
                            break
                    # checking final processor support
                    if not processor.is_supported_operation(optree, language=language):
                        # look for possible simplification
                        if has_support_simplification(optree):
                            simplified_tree = get_support_simplification(optree, processor)
                            Log.report(Log.Verbose, "simplifying %s" % optree.get_str(depth = 2, display_precision = True))
                            Log.report(Log.Verbose, "into %s" % simplified_tree.get_str(depth = 2, display_precision = True))
                            optree.change_to(simplified_tree)
                            if processor.is_supported_operation(optree, language=language):
                                memoization_map[optree] = True
                                return True
                        print("pre escalation node is: ", old_list) # Error print
                        print("languages is {}".format(language))
                        print("Operation' keys are: {}".format(processor.get_operation_keys(optree))) # Error print
                        print("Operation tree is: \n", optree.get_str(display_precision=True, depth=1, display_id=True, memoization_map=None)) # Error print
                        Log.report(Log.Error, "unsupported operation in PassCheckProcessorSupport's check_processor_support {}:\n{}", processor, optree)
            else:
                # memoization
                memoization_map[optree] = True
            return True

        return recursive_support_check(root)


# register pass
Log.report(LOG_PASS_INFO, "Registering silence_fp_operations pass")
Pass.register(PassSilenceFPOperation)

Log.report(LOG_PASS_INFO, "Registering fuse_fma pass")
Pass.register(PassFuseFMA)

Log.report(LOG_PASS_INFO, "Registering sub_expr_sharing pass")
Pass.register(PassSubExpressionSharing)

Log.report(LOG_PASS_INFO, "Registering check_processor_support pass")
Pass.register(PassCheckProcessorSupport)
