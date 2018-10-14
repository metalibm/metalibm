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
# created:          Dec 24th, 2013
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import sys

from .code_element import CodeVariable, CodeExpression
from ..core.ml_operations import Variable, Constant, ConditionBlock, Return, TableLoad, Statement, SpecificOperation, Conversion
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from .generator_utility import C_Code, Gappa_Code, RoundOperator
from ..core.attributes import ML_Debug
from .code_object import Gappa_Unknown, GappaCodeObject

from ..utility.gappa_utils import execute_gappa_script_extract
from ..utility.log_report import Log


class GappaCodeGenerator(object):
    language = Gappa_Code

    """ C language code generator """
    def __init__(self, processor, declare_cst = True, disable_debug = False, libm_compliant = False, exact_mode = False):
        # on level for each of exact_mode possible values
        self.memoization_map = [{}]
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.exact_mode = exact_mode

        self.exact_hint_map = {False: {}, True: {}}

    def get_unknown_precision(self):
        """ return a default format when compound operator encounter
            an undefined precision """
        return ML_Exact

    def set_exact_mode(self, value = True):
        self.exact_mode = value

    def get_exact_mode(self):
        return self.exact_mode

    def open_memoization_level(self):
        self.memoization_map.insert(0, {})

    def close_memoization_level(self):
        self.memoization_map.pop(0)

    def clear_memoization_map(self):
        self.exact_hint_map = {False: {}, True: {}}
        self.memoization_map = [{}]

    def has_memoization(self, optree):
        """ test if a optree has already been generated and memoized """
        for memoization_level in self.memoization_map:
            if optree in memoization_level: return True
        return False

    def get_memoization(self, optree):
        """ retrieve pre-existing memoization entry """
        for memoization_level in self.memoization_map:
            if optree in memoization_level: return memoization_level[optree]
        return None

    def add_memoization(self, optree, code_value):
        """ register memoization value <code_value> for entry <optree> """
        self.memoization_map[0][optree] = code_value


    def add_hypothesis(self, code_object, hypoth_optree, hypoth_value):
        hypothesis_code = self.generate_expr(code_object, hypoth_optree, initial = True, language = Gappa_Code)
        code_object.add_hypothesis(hypothesis_code, hypoth_value)

    def add_goal(self, code_object, goal_optree, goal_value = Gappa_Unknown):
        goal_code = self.generate_expr(code_object, goal_optree, initial = True, language = Gappa_Code)
        code_object.add_goal(goal_code, goal_value)

    def add_hint(self, code_object, hint_hypoth, hint_goal, hint_annotation = None, isApprox = False):
        hypoth_code = self.generate_expr(code_object, hint_hypoth, initial = False, folded = False, language = Gappa_Code)
        goal_code = self.generate_expr(code_object, hint_goal, initial = False, folded = False, language = Gappa_Code)
        if hint_annotation is not None:
          declare_cst = self.declare_cst
          self.declare_cst = False
          annotation_code = self.generate_expr(code_object, hint_annotation, initial = False, folded = False, language = Gappa_Code, strip_outer_parenthesis = True)
          self.declare_cst = declare_cst
        else:
          annotation_code = None
        code_object.add_hint(hypoth_code, goal_code, annotation_code, isApprox)


    # force_variable_storing is not supported
    def generate_expr(self, code_object, optree, folded = True, result_var = None, initial = False, __exact = None, language = None, strip_outer_parenthesis = False, force_variable_storing = False):
        """ code generation function """
        #exact_value = exact or self.get_exact_mode()

        # search if <optree> has already been processed
        if self.has_memoization(optree):
            return self.get_memoization(optree)

        result = None
        # implementation generation
        if isinstance(optree, CodeVariable):
            result = optree

        elif isinstance(optree, Variable):
            #if optree.get_max_abs_error() != None:
            #    max_abs_error = optree.get_max_abs_error()
            #    var_name = code_object.get_free_var_name(optree.get_precision(), prefix = "%s_" % optree.get_tag(), declare = True)
            #    result = CodeVariable(var_name, optree.get_precision())

            #   error_var = Variable(tag = var_name, precision = optree.get_precision())
            #    optree.set_max_abs_error(None)
            #    hypothesis = error_var - optree
            #    hypothesis.set_precision(ML_Exact)
            #    self.add_hypothesis(code_object, hypothesis, Interval(-max_abs_error, max_abs_error))
            #    optree.set_max_abs_error(max_abs_error)
            #    self.add_memoization(error_var, result)
            #else:
            #    result = CodeVariable(optree.get_tag(), optree.get_precision())
            result = CodeVariable(optree.get_tag(), optree.get_precision())

        elif isinstance(optree, Constant):
            precision = optree.get_precision()
            if self.declare_cst:
                cst_prefix = "cst" if optree.get_tag() is None else optree.get_tag()
                cst_varname = code_object.declare_cst(optree, prefix = cst_prefix)
                result = CodeVariable(cst_varname, precision)
            else:
                result = CodeExpression(precision.get_gappa_cst(optree.get_value()), precision)

        elif isinstance(optree, Conversion):
            if optree.get_rounding_mode() is not None:
              local_implementation = RoundOperator(optree.get_precision(), direction = optree.get_rounding_mode())
            else:
              local_implementation = RoundOperator(optree.get_precision())
            return local_implementation.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var)
            

        elif isinstance(optree, TableLoad):
            # declaring table
            table = optree.inputs[0]
            tag = table.get_tag()
            table_name = code_object.declare_table(table, prefix = tag if tag != None else "table") 

            index_code = [self.generate_expr(code_object, index_op, folded = folded).get() for index_op in optree.inputs[1:]]

            result = CodeExpression("%s[%s]" % (table_name, "][".join(index_code)), optree.inputs[0].get_storage_precision())

        elif isinstance(optree, ConditionBlock):
            condition = optree.inputs[0]
            if_branch = optree.inputs[1]
            else_branch = optree.inputs[2] if len(optree.inputs) > 2 else None

            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), folded = folded)

            cond_code = self.generate_expr(code_object, condition, folded = folded)
            if condition.get_likely() in [True, False]:
                code_object << "\nif (__builtin_expect(%s, %d)) " % (cond_code.get(), {True: 1, False: 0}[condition.get_likely()])
            else:
                code_object << "\nif (%s) " % cond_code.get()
            self.open_memoization_level()
            code_object.open_level()
            #if_branch_code = self.processor.generate_expr(self, code_object, if_branch, if_branch.inputs, folded)
            if_branch_code = self.generate_expr(code_object, if_branch, folded = folded)
            code_object.close_level(cr = "")
            self.close_memoization_level()
            if else_branch:
                code_object << " else "
                code_object.open_level()
                self.open_memoization_level()
                else_branch_code = self.generate_expr(code_object, else_branch, folded = folded)
                code_object.close_level()
                self.close_memoization_level()
            else:
                code_object << "\n"

            return None

        elif isinstance(optree, Return):
            return None

        elif isinstance(optree, SpecificOperation):
            result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = False, result_var = result_var, language = self.language)
            code_object << "%s;\n" % result_code.get()
            return None

        elif isinstance(optree, Statement):
            for op in optree.inputs:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, folded = folded, initial = True)

            return None

        else:
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var, language = self.language)

            if optree.get_exact():
                key = optree.get_handle()
                exact_flag = (optree.get_precision() == ML_Exact or self.get_exact_mode() == True)
                if key in self.exact_hint_map[True] and key in self.exact_hint_map[False]:
                    # already processed, skip
                    pass
                else:
                    self.exact_hint_map[exact_flag][key] = optree
                    if key in self.exact_hint_map[not exact_flag]:
                        self.add_hint(code_object, self.exact_hint_map[False][key], self.exact_hint_map[True][key])
            


        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            code_object << self.generate_debug_msg(optree, result)
            

        if initial and not isinstance(result, CodeVariable):
            final_var = result_var if result_var else code_object.get_free_var_name(optree.get_precision(), prefix = "result", declare = True)
            code_object << self.generate_assignation(final_var, result.get())
            return CodeVariable(final_var, optree.get_precision())

        if strip_outer_parenthesis and isinstance(result, CodeExpression):
          result.strip_outer_parenthesis()
        return result

    def generate_code_assignation(self, code_object, result_var, expr_code, final=True):
        return self.generate_assignation(result_var, expr_code, final=final)

    def generate_assignation(self, result_var, expression_code, final = True):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = ";\n" if final else ""
        return "%s = %s%s" % (result_var, expression_code, final_symbol) 


    def generate_declaration(self, symbol, symbol_object, initial = True, final = True):
        if isinstance(symbol_object, Constant):
            initial_symbol = ""#(symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s = %s%s" % (initial_symbol, symbol, symbol_object.get_precision().get_gappa_cst(symbol_object.get_value()), final_symbol) 
        elif isinstance(symbol_object, Variable):
            initial_symbol = ""#(symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s%s" % (initial_symbol, symbol, final_symbol) 
        elif isinstance(symbol_object, ML_Table):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
        return ""


    def generate_debug_msg(self, optree, result):
        debug_object = optree.get_debug()
        precision = optree.get_precision()
        display_format = debug_object.get_display_format(precision.get_c_display_format()) if isinstance(debug_object, ML_Debug) else precision.get_c_display_format()
        debug_msg = "#ifdef ML_DEBUG\n"
        debug_msg += """printf("%s: %s\\n", %s);\n""" % (optree.get_tag(), display_format, result.get())
        debug_msg += "#endif\n"
        return debug_msg


    def get_eval_error(self, pre_optree, variable_copy_map = {}, goal_precision = ML_Binary32, gappa_filename = "gappa_tmp.g"):
        """ helper to compute the evaluation error of <pre_optree> bounded by tagged-node in variable_map, 
            assuming variable_map[v] is the liverange of node v """
        # registering initial bounds
        bound_list = [op for op in variable_copy_map]
        # copying pre-operation tree
        optree = pre_optree.copy(variable_copy_map)
        gappa_code = GappaCodeObject()

        gappa_result_approx = self.generate_expr(gappa_code, optree, initial = False)
        gappa_result_exact  = self.generate_expr(gappa_code, optree, initial = False)
        goal = gappa_result_approx.get_variable(gappa_code) - gappa_result_exact.get_variable(gappa_code)
        goal.set_attributes(precision = goal_precision, tag = "goal")
        self.add_goal(gappa_code, goal)
        for v in bound_list:
            self.add_hypothesis(gappa_code, variable_copy_map[v], variable_copy_map[v].get_interval())
        self.clear_memoization_map()
        return execute_gappa_script_extract(gappa_code.get(self), gappa_filename = gappa_filename)["goal"]


    def get_eval_error_v2(self, opt_engine, pre_optree, variable_copy_map = {}, goal_precision = ML_Exact, gappa_filename = "gappa_tmp.g", relative_error = False):
        """ helper to compute the evaluation error of <pre_optree> bounded by tagged-node in variable_map, 
            assuming variable_map[v] is the liverange of node v """
        # registering initial bounds
        bound_list = []
        bound_unique_list = []
        bound_targets = []
        for op in variable_copy_map:
          bound_list.append(op)
          if not variable_copy_map[op] in bound_targets:
            bound_unique_list.append(op)
            bound_targets.append(variable_copy_map[op])
        # copying pre-operation tree
        optree = pre_optree.copy(variable_copy_map)
        gappa_code = GappaCodeObject()

        # quantization error variable map
        var_error_copy_map = {}
        for v in bound_list:
            max_abs_error = v.get_max_abs_error()
            if max_abs_error == None:
                var_error_copy_map[v] = variable_copy_map[v]
                if v in bound_unique_list: 
                  self.add_hypothesis(gappa_code, variable_copy_map[v], variable_copy_map[v].get_interval())
            else:
                var_error_interval = Interval(-max_abs_error, max_abs_error)
                var = variable_copy_map[v]
                exact_var = Variable(var.get_tag() + "_", precision = var.get_precision(), interval = var.get_interval())
                var_error_copy_map[v] = exact_var 

                if v in bound_unique_list: 
                  self.add_hypothesis(gappa_code, exact_var, variable_copy_map[v].get_interval())
                sub_var = var - exact_var
                sub_var.set_precision(ML_Exact)

                if v in bound_unique_list: 
                  self.add_hypothesis(gappa_code, sub_var, var_error_interval)

        pre_exact_optree = pre_optree.copy(var_error_copy_map)
        exact_optree = opt_engine.exactify(pre_exact_optree.copy())

        gappa_result_exact  = self.generate_expr(gappa_code, exact_optree, initial = True)
        #print "gappa_code: ", gappa_code.get(self)
        gappa_result_approx = self.generate_expr(gappa_code, optree, initial = False)
        #print "gappa_code: ", gappa_code.get(self)
        # Gappa Result Approx variable
        gra_var = gappa_result_approx.get_variable(gappa_code)
        # Gappa Result Exact variable
        gre_var = gappa_result_exact.get_variable(gappa_code)
        goal_diff = gra_var - gre_var 
        goal_diff.set_attributes(precision = goal_precision, tag = "goal_diff")
        if relative_error:
          goal = goal_diff / gre_var
        else:
          goal = goal_diff
        goal.set_attributes(precision = goal_precision, tag = "goal")
        self.add_goal(gappa_code, goal)

        self.clear_memoization_map()
        try:
          eval_error = execute_gappa_script_extract(gappa_code.get(self), gappa_filename = gappa_filename)["goal"]
          return eval_error
        except ValueError:
          Log.report(Log.Error, "Unable to compute evaluation error with gappa")
          

    def get_eval_error_v3(self, opt_engine, pre_optree, variable_copy_map = {}, goal_precision = ML_Exact, gappa_filename = "gappa_tmp.g", dichotomy = [], relative_error = False):
        # storing initial interval values
        init_interval = {}
        for op in variable_copy_map:
            init_interval[op] = variable_copy_map[op].get_interval()

        eval_error_list = []
        case_id = 0

        # performing dichotomised search
        for case in dichotomy: 
            clean_copy_map = {}
            for op in variable_copy_map:
                clean_copy_map[op] = variable_copy_map[op]
                if op in case:
                    # if op interval is set in case, transmist interval information to copy map
                    clean_copy_map[op].set_interval(case[op])
                else:
                    # else making sure initial interval is set
                    clean_copy_map[op].set_interval(init_interval[op])
                    
            # computing evaluation error in local conditions
            eval_error = self.get_eval_error_v2(opt_engine, pre_optree, clean_copy_map, goal_precision, ("c%d_" % case_id) + gappa_filename, relative_error = relative_error)
            eval_error_list.append(eval_error)
            case_id += 1

        return eval_error_list


    def get_interval_code(self, pre_goal, variable_copy_map = {}, goal_precision = ML_Exact, update_handle = True):
        # registering initial bounds
        bound_list = [op for op in variable_copy_map]

        # copying pre-operation tree
        goal = pre_goal.copy(variable_copy_map)
        goal.set_attributes(precision = goal_precision, tag = "goal")
        Log.report(Log.Info, "goal: ", goal)

        # updating handle
        if update_handle:
            for v in variable_copy_map:
                new_v = variable_copy_map[v]
                v.get_handle().set_node(new_v)

        gappa_code = GappaCodeObject()

        #gappa_result_approx = self.generate_expr(gappa_code, goal, initial = False, exact = False)
        self.add_goal(gappa_code, goal)
        for v in bound_list:
            self.add_hypothesis(gappa_code, variable_copy_map[v], variable_copy_map[v].get_interval())

        return gappa_code


    def get_interval_code_no_copy(self, goal, goal_precision = ML_Exact, update_handle = True, bound_list = []):

        # copying pre-operation tree
        goal.set_attributes(precision = goal_precision, tag = "goal")

        gappa_code = GappaCodeObject()

        #gappa_result_approx = self.generate_expr(gappa_code, goal, initial = False, exact = False)
        self.add_goal(gappa_code, goal)
        for v in bound_list:
            self.add_hypothesis(gappa_code, v, v.get_interval())

        return gappa_code



