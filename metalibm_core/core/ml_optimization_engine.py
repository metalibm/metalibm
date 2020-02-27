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
# created:          Mar 20th, 2014
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import sys

from sollya import inf, sup

from ..utility.log_report import Log
from .ml_operations import *
from .ml_hdl_operations import *
from .ml_formats import *

import metalibm_core.opt.p_function_std as p_function_std
import metalibm_core.opt.p_function_typing as p_function_typing


# high verbosity log-level for optimization engine
LOG_LEVEL_OPT_ENG_VERBOSE = Log.LogLevel("OptEngVerbose")


type_escalation = {
    Addition: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
        },
    },
    Multiplication: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
            lambda op_type: isinstance(op_type, ML_FP_Format):
                lambda op: op.get_precision(),
        },
    },
    FusedMultiplyAdd: {
        lambda result_type: isinstance(result_type, ML_FP_Format): {
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: op.get_precision(),
            lambda op_type: isinstance(op_type, ML_FP_Format):
                lambda op: op.get_precision(),
        },
    },
    ExponentInsertion: {
        lambda result_type: not isinstance(result_type, ML_VectorFormat) : {
            lambda op_type: isinstance(op_type, ML_FP_Format): 
                lambda op: {32: ML_Int32, 64: ML_Int64}[op.get_precision().get_bit_size()],
            lambda op_type: isinstance(op_type, ML_Fixed_Format):
                lambda op: ML_Int32,
        },
    },
}


# Table of transformation rule to translate an operation into its exact (no rounding error) counterpart
exactify_rule = {
    Constant: {
      None: {
        lambda optree, exact_format: optree.get_precision() is None: 
          lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
      },
    },
    Division: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    Addition: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    Multiplication: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    Subtraction: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
    FusedMultiplyAdd: { 
        None: {
            lambda optree, exact_format: True: 
                lambda opt_eng, optree, exact_format: opt_eng.swap_format(optree, exact_format),
        },
    },
}






class OptimizationEngine(object):
    """ backend (precision instanciation and optimization passes) class """
    def __init__(self, processor, default_integer_format = ML_Int32, default_fp_precision = ML_Binary32, change_handle = True, dot_product_enabled = False, default_boolean_precision = ML_Int32):
        self.processor = processor
        self.default_integer_format = default_integer_format
        self.default_fp_precision = default_fp_precision
        self.change_handle = change_handle
        self.dot_product_enabled = dot_product_enabled
        self.default_boolean_precision = default_boolean_precision

    def set_dot_product_enabled(self, dot_product_enabled):
        self.dot_product_enabled = dot_product_enabled

    def get_dot_product_enabled(self):
        return self.dot_product_enabled

    def copy_optree(self, optree, copy_map = None):
        copy_map = {} if copy_map is None else copy_map
        return optree.copy(copy_map)


    def get_default_fp_precision(self, optree):
        return self.default_fp_precision


    def get_boolean_format(self, optree):
        """ return boolean format to use for optree """
        return self.default_boolean_precision
    def set_boolean_format(self, new_boolean_format):
        self.default_boolean_precision = new_boolean_format


    def merge_abstract_format(self, optree, args, default_precision = None):
        """ merging input format in multi-ary operation to determined result format """
        return p_function_typing.merge_ops_abstract_format(optree, args, default_precision)


    def instantiate_abstract_precision(self, optree, default_precision=None,
                                       memoization_map=None):
        """ recursively determine an abstract precision for each node """
        return p_function_typing.instantiate_abstract_precision(
            optree, default_precision, memoization_map)
        

    def simplify_fp_context(optree):
        """ factorize exception clearing and rounding mode changes accross
            connected DAG of floating-point operations """

        def is_fp_op(_optree):
            return isinstance(_optree.get_precision(), ML_FP_Format)

        if is_fp_op(optree):
            pass
        


    def instantiate_precision(self, optree, default_precision=None, memoization_map=None):
        """ instantiate final precisions and insert required conversions
            if the operation is not supported """
        memoization_map = memoization_map or {}
        return p_function_typing.instantiate_precision(
            optree, default_precision, memoization_map, backend=self
        )


    def cb_parent_tagging(self, optree, parent_block = None):
        """ tries to factorize subexpression sharing between branch of ConditionBlock """
        if isinstance(optree, ConditionBlock):
            optree.parent = parent_block
            for op in optree.inputs: 
                self.cb_parent_tagging(op, parent_block = optree)
        elif not isinstance(optree, ML_LeafNode):
            for op in optree.inputs:
                self.cb_parent_tagging(op, parent_block = parent_block)


    def subexpression_sharing(self, optree):
        return p_function_std.subexpression_sharing(optree)


    def extract_fast_path(self, optree):
        """ extracting fast path (most likely execution path leading
            to a Return operation) from <optree> """
        if isinstance(optree, ConditionBlock):
            cond = optree.inputs[0]
            likely = cond.get_likely()
            if likely:
                return self.extract_fast_path(optree.inputs[1])
            elif likely == False and len(optree.inputs) >= 3:
                return self.extract_fast_path(optree.inputs[2])
            else:
                return None
        elif isinstance(optree, Statement):
            for sub_stat in optree.inputs:
                ss_fast_path = self.extract_fast_path(sub_stat)
                if ss_fast_path != None: return ss_fast_path
            return None
        elif isinstance(optree, Return):
            return optree.inputs[0]
        else:
            return None


    def factorize_fast_path(self, optree):
        """ extract <optree>'s fast path and add it to be pre-computed at
            the start of <optree> computation """
        fast_path = self.extract_fast_path(optree)
        if fast_path == None:
            return
        elif isinstance(optree, Statement):
            optree.push(fast_path)
        else:
            Log.report(Log.Error, "unsupported root for fast path factorization")


    def fuse_multiply_add(self, optree, silence = False, memoization = None):
        return p_function_std.fuse_multiply_add(
            optree, silence, memoization, self.change_handle,
            self.dot_product_enabled)

    def silence_fp_operations(self, optree, force = False, memoization_map = None):
        return p_function_std.silence_fp_operations(optree, force, memoization_map)


    def register_nodes_by_tag(self, optree, node_map = {}):
        """ build a map tag->optree """
        # registering node if tag is defined
        if optree.get_tag() != None:
            node_map[optree.get_tag()] = optree

        # processing extra_inputs list
        for op in optree.get_extra_inputs():
            self.register_nodes_by_tag(op, node_map)

        # processing inputs list for non ML_LeafNode optree
        if not isinstance(optree, ML_LeafNode):
            for op in optree.inputs:
                self.register_nodes_by_tag(op, node_map)


    def recursive_swap_format(self, optree, old_format, new_format, memoization_map = None):
      memoization_map = {} if memoization_map is None else memoization_map
      if optree in memoization_map:
        return
      else:
        if optree.get_precision() is old_format:
          optree.set_precision(new_format)
        memoization_map[optree] = optree
        for node in optree.get_inputs() + optree.get_extra_inputs():
          self.recursive_swap_format(node, old_format, new_format)


    def check_processor_support(self, optree, memoization_map = {}, debug = False, language = C_Code):
        return p_function_std.PassCheckProcessorSupport.check_processor_support(self.processor, optree, memoization_map, debug, langage) 

    def swap_format(self, optree, new_format):
        optree.set_precision(new_format)
        return optree

    def exactify(self, optree, exact_format = ML_Exact, memoization_map = {}):
        """ recursively process <optree> according to table exactify_rule 
            to translete each node into is exact counterpart (no rounding error)
            , generally by setting its precision to <exact_format> """
        if optree in memoization_map:
            return memoization_map[optree]
        if not isinstance(optree, ML_LeafNode):
            for inp in optree.inputs:
                self.exactify(inp, exact_format, memoization_map)
            for inp in optree.get_extra_inputs():
                self.exactify(inp, exact_format, memoization_map)

        if optree.__class__ in exactify_rule:
            for cond in exactify_rule[optree.__class__][None]:
                if cond(optree, exact_format):
                    new_optree = exactify_rule[optree.__class__][None][cond](self, optree, exact_format)
                    memoization_map[optree] = new_optree
                    return new_optree

        memoization_map[optree] = optree
        return optree

    def static_vectorization(self, optree):
      pass


    def optimization_process(self, pre_scheme, default_precision, copy = False, fuse_fma = True, subexpression_sharing = True, silence_fp_operations = True, factorize_fast_path = True, language = C_Code):
        # copying when required
        scheme = pre_scheme if not copy else pre_scheme.copy({})

        if fuse_fma:
            Log.report(Log.Info, "Fusing FMA")
        scheme_post_fma = scheme if not fuse_fma else self.fuse_multiply_add(scheme, silence = silence_fp_operations)

        Log.report(Log.Info, "Infering types")
        self.instantiate_abstract_precision(scheme_post_fma, None)
        Log.report(Log.Info, "Instantiating precisions")
        self.instantiate_precision(scheme_post_fma, default_precision)

        if subexpression_sharing:
            Log.report(Log.Info, "Sharing sub-expressions")
            self.subexpression_sharing(scheme_post_fma)

        if silence_fp_operations:
            Log.report(Log.Info, "Silencing exceptions in internal fp operations")
            self.silence_fp_operations(scheme_post_fma)

        Log.report(Log.Info, "Checking processor support")
        self.check_processor_support(scheme_post_fma, memoization_map = {}, language = language)

        if factorize_fast_path:
            Log.report(Log.Info, "Factorizing fast path")
            self.factorize_fast_path(scheme_post_fma)

        return scheme_post_fma
        


            


        
