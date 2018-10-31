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
        ML_LeafNode, VectorElementSelection, FunctionCall, Conversion, Constant
)

from metalibm_core.opt.p_check_support import Pass_CheckSupport

## Test if @p optree is a non-constant leaf node
#    @param optree operation node to be tested
#    @return boolean result of predicate evaluation
def is_leaf_no_Constant(optree):
	return isinstance(optree, ML_LeafNode) and not isinstance(optree, Constant)


def insert_conversion_when_required(op_input, final_precision):
        if op_input.get_precision() != final_precision:
                return Conversion(op_input, precision = final_precision)
        else:
                return op_input


## Generic vector promotion pass
class Pass_Vector_Promotion(FunctionPass):
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
            Log.report(Log.Verbose, "NOT supported in vector_promotion: {}".format(optree.get_str(depth = 2, display_precision = True, memoization_map = {})))
            op, formats, specifier = key_getter(None, optree)
            Log.report(Log.Verbose, "with key: {}, {}, {}".format(str(op), [str(f) for f in formats], str(specifier)))
        else:
            Log.report(Log.Verbose, "supported in vector_promotion: {}".format(optree.get_str(depth = 2, display_precision = True, memoization_map = {})))
            op, formats, specifier = key_getter(None, optree)
            Log.report(Log.Verbose, "with key: {}, {}, {}".format(str(op), [str(f) for f in formats], str(specifier)))
        return support_status

    ## memoize converted
    def memoize(self, force, optree, new_optree):
        self.memoization_map[(force, optree)] = new_optree
        #if not isinstance(optree, ML_LeafNode) and not self.target.is_supported_operation(optree):
        #    print optree.get_str(display_precision = True, memoization_map = {})
        #    print new_optree.get_str(display_precision = True, memoization_map = {})
        #    #raise Exception()
        return new_optree

    ## memoize conversion
    def memoize_conv(self, force, optree, new_optree):
        self.conversion_map[(force, optree)] = new_optree
        #if not isinstance(optree, ML_LeafNode) and not self.target.is_supported_operation(optree):
        #    raise Exception()
        return new_optree

    def get_converted_node(self, optree):
        if optree in self.conversion_map:
            return self.conversion_map[optree]
        else:
            if self.does_target_support_promoted_op(optree):
                new_optree = optree.copy(copy_map = self.copy_map)
                new_inputs = [self.get_converted_node(op) for op in new_optree.get_inputs()]
                new_optree.inputs = new_inputs
                new_optree.set_precision(self.get_conv_format(optree.get_precision()))
                self.conversion_map[optree] = new_optree


    ## Convert a graph of operation to exploit the promoted registers
    #    @param parent_converted indicates that the result must
    #                 be in promoted formats else it must be in input format
    #                 In case of promoted-format, the return value may need to be
    #                 a conversion if the operation is not supported
    def promote_node(self, optree, parent_converted = False):
        if (parent_converted, optree) in self.memoization_map:
                return self.memoization_map[(parent_converted, optree)]
        if 1:
            new_optree = optree.copy(copy_map = self.copy_map)
            if self.does_target_support_promoted_op(optree):

                new_inputs = [self.promote_node(op, parent_converted = True) for op in optree.get_inputs()]
                new_optree.inputs = new_inputs
                new_optree.set_precision(self.get_conv_format(optree.get_precision()))

                # must be converted back to initial format
                # before being returned
                if not parent_converted:

                    new_optree = insert_conversion_when_required(new_optree, optree.get_precision()) # Conversion(new_optree, precision = optree.get_precision())

                return self.memoize(parent_converted, optree, new_optree)
            elif isinstance(optree, ML_NewTable):
                return self.memoize(parent_converted, optree, optree)
            elif is_leaf_no_Constant(optree):
                if parent_converted and optree.get_precision() in self.get_translation_table():
                    new_optree = insert_conversion_when_required(new_optree, self.get_conv_format(optree.get_precision()))#Conversion(optree, precision = self.get_conv_format(optree.get_precision()))
                    return self.memoize(parent_converted, optree, new_optree)
                elif parent_converted:
                    raise NotImplementedError
                else:
                    return self.memoize(parent_converted, optree, optree)
            else:
                # new_optree = optree.copy(copy_map = self.copy_map)

                # propagate conversion to inputs
                new_inputs = [self.promote_node(op) for op in optree.get_inputs()]


                # register modified inputs
                new_optree.inputs = new_inputs

                if parent_converted and optree.get_precision() in self.get_translation_table():
                    new_optree = insert_conversion_when_required(new_optree, self.get_conv_format(optree.get_precision()))#Conversion(new_optree, precision = self.get_conv_format(optree.get_precision()))
                    return new_optree
                elif parent_converted:
                    print(optree.get_precision())
                    raise NotImplementedError
                return self.memoize(parent_converted, optree, new_optree)


    # standard Opt pass API
    def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
        return self.promote_node(optree)


Log.report(LOG_PASS_INFO, "Registering vector_conversion pass")
# register pass
Pass.register(Pass_Vector_Promotion)
