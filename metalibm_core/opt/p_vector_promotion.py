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
# Desciprion: optimization pass to promote a scalar/vector DAG into vector
#                         registers
###############################################################################

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import FunctionPass, Pass, LOG_PASS_INFO
from metalibm_core.core.ml_table import ML_NewTable, ML_TableFormat
from metalibm_core.core.ml_operations import (
        ML_LeafNode, VectorElementSelection, FunctionCall, Conversion, Constant,
        ControlFlowOperation, ReferenceAssign,
)

from metalibm_core.opt.p_check_support import Pass_CheckSupport

LOG_LEVEL_VPROMO_VERBOSE = Log.LogLevel("VPromoVerbose")
LOG_LEVEL_VPROMO_INFO = Log.LogLevel("VPromoInfo")

## Test if @p optree is a non-constant leaf node
#    @param optree operation node to be tested
#    @return boolean result of predicate evaluation
def is_leaf_no_Constant(optree):
	return isinstance(optree, ML_LeafNode) and not isinstance(optree, Constant)

def insert_conversion_when_required(op_input, final_precision):
    # assert not final_precision is None
    if op_input.get_precision() != final_precision:
        return Conversion(op_input, precision = final_precision)
    else:
        return op_input


## Generic vector promotion pass
class Pass_Vector_Promotion(FunctionPass):
    """ Generic pass to promote node format to better format for the final
        target. This pass is specially dedicated to promote generic vector
        format to target-specific format (e.g. v4float to ML_SSE_v4float)

        This pass works by looking up in the translation table if a
        promoted format is available for all the I/Os format of a node.
        If such formats exist and if the node with promoted format is
        supported by the target the promotion is performed.
            Conversion are inserted at the begining and the end of a
        promoted subgraph to ensure compatible connection with the full
        graph """
    pass_tag = "vector_promotion"
    ## Return the translation table of formats
    #    to be used for promotion
    def get_translation_table(self):
        raise NotImplementedError

    def __init__(self, target):
        FunctionPass.__init__(self, "vector promotion pass", target)
        ## memoization map for promoted optree
        self.memoization_map = {}
        ## memoization map for converted promoted optree
        self.conversion_map = {}
        # memoization map for copy
        self.copy_map = {}

        self.support_checker = Pass_CheckSupport(target)

    ## Evaluate the latency of a converted operation
    #    graph to determine whether the conversion
    #    is worth it
    def evaluate_converted_graph_cost(self, optree):
        pass

    def get_conv_format(self, precision):
        # table precision are left unchanged
        if isinstance(precision, ML_TableFormat):
            return precision
        else:
            return self.get_translation_table()[precision]

    def is_convertible_format(self, precision):
        # Table format is always convertible
        if isinstance(precision, ML_TableFormat):
            return True
        else:
            return precision in self.get_translation_table()

    ## test wether optree's operation is supported on
    #    promoted formats
    def does_target_support_promoted_op(self, optree):
        # check that the output format is supported
        if not self.is_convertible_format(optree.get_precision()):
            return False
        if is_leaf_no_Constant(optree) \
             or isinstance(optree, VectorElementSelection) \
             or isinstance(optree, FunctionCall):
            return False
        # check that the format of every input is supported
        for arg in optree.get_inputs():
            if not self.is_convertible_format(arg.get_precision()):
                return False
        ## This local key getter modifies on the fly
        # the optree precision to promotion-based formats
        # to determine if the converted node is supported
        def key_getter(target_obj, optree):
            op_class = optree.__class__
            result_type = (self.get_conv_format(optree.get_precision().get_match_format()),)
            arg_type = tuple((self.get_conv_format(arg.get_precision().get_match_format()) if not arg.get_precision() is None else None) for arg in optree.get_inputs())
            interface = result_type + arg_type
            codegen_key = optree.get_codegen_key()
            return op_class, interface, codegen_key

        support_status = self.target.is_supported_operation(optree, key_getter=key_getter)
        if not support_status:
            Log.report(LOG_LEVEL_VPROMO_INFO, "NOT supported in vector_promotion: {}".format(optree.get_str(depth = 2, display_precision = True, memoization_map = {})))
            op, formats, specifier = key_getter(None, optree)
            Log.report(LOG_LEVEL_VPROMO_INFO, "with key: {}, {}, {}".format(str(op), [str(f) for f in formats], str(specifier)))
        else:
            Log.report(LOG_LEVEL_VPROMO_VERBOSE, "supported in vector_promotion: {}".format(optree.get_str(depth = 2, display_precision = True, memoization_map = {})))
            op, formats, specifier = key_getter(None, optree)
            Log.report(LOG_LEVEL_VPROMO_VERBOSE, "with key: {}, {}, {}".format(str(op), [str(f) for f in formats], str(specifier)))
        return support_status

    def memoize(self, expected_format, optree, new_optree):
        """ Memoization @p new_optree which is the promoted version of @p optree
            with precision @p expected_format
        """
        self.memoization_map[(expected_format, optree)] = new_optree
        return new_optree


    def get_converted_node(self, optree):
        if optree in self.conversion_map:
            return self.conversion_map[optree]
        else:
            if self.does_target_support_promoted_op(optree):
                new_optree = optree.copy(copy_map=self.copy_map)
                new_inputs = [self.get_converted_node(op) for op in new_optree.get_inputs()]
                new_optree.inputs = new_inputs
                new_optree.set_precision(self.get_conv_format(optree.get_precision()))
                self.conversion_map[optree] = new_optree


    def promote_node(self, optree, expected_format):
        """
           Convert a graph of operation to exploit the promoted registers
           @param optree operation node to be promoted
           @param expected_format is the format expected for the result
                  This may trigger Conversion insertion when required
           @return promoted node with precision equals to expected_format
        """
        if (expected_format, optree) in self.memoization_map:
                return self.memoization_map[(expected_format, optree)]
        if 1:
            if self.does_target_support_promoted_op(optree):
                new_optree = optree.copy(copy_map=self.copy_map)

                new_inputs = [self.promote_node(op, self.get_conv_format(op.precision)) for op in optree.get_inputs()]
                new_optree.inputs = new_inputs
                new_optree.set_precision(self.get_conv_format(optree.get_precision()))

                # must be converted back to initial format
                # before being returned
                new_optree = insert_conversion_when_required(new_optree, expected_format)
                return self.memoize(expected_format, optree, new_optree)
            else:
                new_optree = optree
                if isinstance(optree, ML_NewTable):
                    # Table are not promoted (we assume vector access are gather
                    # which are compatible between vector format): TO BE IMPROVED
                    return self.memoize(expected_format, optree, optree)
                elif is_leaf_no_Constant(optree):
                    if expected_format is optree.precision:
                        return optree
                    elif optree.get_precision() in self.get_translation_table():
                        new_optree = insert_conversion_when_required(new_optree, expected_format)
                        return self.memoize(expected_format, optree, new_optree)
                    else:
                        Log.report(Log.Error, "following leaf's optree format is not supported for conversion (though required): {}", optree)
                        raise NotImplementedError
                else:
                    # new_optree = optree.copy(copy_map = self.copy_map)
                    # promote node operands
                    new_inputs = [self.promote_node(op, op.precision) for op in optree.get_inputs()]
                    # register modified inputs
                    new_optree.inputs = new_inputs

                    if not isinstance(optree, ControlFlowOperation) and not isinstance(optree, ReferenceAssign):
                        new_optree = insert_conversion_when_required(new_optree, expected_format)
                    return self.memoize(expected_format, optree, new_optree)


    # standard Opt pass API
    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        return self.promote_node(optree, optree.precision)


Log.report(LOG_PASS_INFO, "Registering vector_conversion pass")
# register pass
Pass.register(Pass_Vector_Promotion)
