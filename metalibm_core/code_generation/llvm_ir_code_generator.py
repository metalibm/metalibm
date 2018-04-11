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

from ..core.ml_operations import (
    Variable, Constant, ConditionBlock, Return, TableLoad, Statement,
    SpecificOperation, Conversion, FunctionObject
)
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from ..core.attributes import ML_Debug
from .generator_utility import C_Code, Gappa_Code, RoundOperator
from .code_element import CodeVariable, CodeExpression
from .code_object import MultiSymbolTable

from ..utility.log_report import Log

from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.llvm_utils import llvm_ir_format

# TODO factorize outside this file
class Label(object):
    """ Label (tag for code position) object """
    def __init__(self, name):
        self.name = name

def get_free_label_name(code_object, prefix):
    """ generate a new label name (previously unused) """
    return code_object.get_free_symbol_name(
        MultiSymbolTable.LabelSymbol, None, Label,
        prefix=prefix,
        declare=True)

def llvm_ir_generate_condition_block(generator, optree, code_object, language, folded=False, next_block=None):
    condition = optree.inputs[0]
    if_branch = optree.inputs[1]
    else_branch = optree.inputs[2] if len(optree.inputs) > 2 else None
    # generating pre_statement
    generator.generate_expr(
        code_object, optree.get_pre_statement(),
        folded=folded, language=language)
    # generate code to evaluate if-then-else condition
    cond_code = generator.generate_expr(
        code_object, condition, folded=folded, language=language)

    def get_end_label():
        if next_block is None:
            return get_free_label_name(code_object, "end")
        else:
            return next_block

    if_label = get_free_label_name(code_object, "true_label")

    is_fallback_if = is_fallback_statement(if_branch) 

    if else_branch:
        # if there is an else branch then else label must be specific
        else_label = get_free_label_name(code_object, "false_label")
        # we need a end label if one (or more) of if/else is a fallback
        is_fallback = is_fallback_if or is_fallback_statement(else_branch)
        if is_fallback:
            end_label = get_end_label() 
        else:
            end_label = None
    else:
        # there is no else so false-cond requires a fallback blocks
        is_fallback = True 
        else_label = get_end_label()
        end_label = else_label

    code_object << "br i1 {cond} , label %{if_label}, label %{else_label}\n".format(
        cond=cond_code.get(),
        if_label=if_label,
        else_label=else_label
    ) 
    def append_label(label):
        print "adding label : ", label
        code_object.close_level(footer="", cr="")
        code_object << label << ":"
        code_object.open_level(header="") #, extra_shared_tables=[MultiSymbolTable.VariableSymbol])

    append_label(if_label)

    # generating code for if-branch
    if_branch_code = generator.generate_expr(
        code_object, if_branch, folded=folded,
        language=language, next_block=end_label)

    if is_fallback_if:
        code_object << "br label %" << end_label << "\n"

    if else_branch:
        append_label(else_label)
        else_branch_code = generator.generate_expr(
            code_object, else_branch, folded=folded,
            language=language, next_block=end_label)

    if is_fallback and next_block is None:
        append_label(end_label)

    return None

def is_fallback_statement(optree):
    """ Determinate if <optree> may fallback to the next block (True)
        or if it will never reach next block (False)"""
    if isinstance(optree, Return):
        return False
    elif isinstance(optree, ConditionBlock):
        branch_if = optree.get_input(1)
        branch_else = optree.get_input(2) if len(optree.inputs) > 2 else None
        if branch_else is None:
            # if there is no else-branch than false condition will
            # trigger fallback
            return True
        else:
            return is_fallback_statement(branch_if) or is_fallback_statement(branch_else)
    elif isinstance(optree, Statement):
        # if any of the sequential sub-statement is not a 
        # fallback then the overall statement is not one too
        for op in optree.get_inputs():
            if not is_fallback_statement(optree):
                return False
        return True
    else:
        return True
                            

class LLVMIRCodeGenerator(object):
    """ LLVM-IR language code generator """
    language = LLVM_IR_Code

    def __init__(self, processor, declare_cst=True, disable_debug=False, libm_compliant=False, language=LLVM_IR_Code):
        # on level for each of exact_mode possible values
        self.generated_map = [{}]
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.language = language
        self.end_label = None

    def open_memoization_level(self):
        """ Create a new memoization level on top of the stack """
        self.generated_map.insert(0, {})
    def close_memoization_level(self):
        """ Close the highest memoization level """
        self.generated_map.pop(0)

    def has_memoization(self, optree):
        """ test if a optree has already been generated and memoized """
        for memoization_level in self.generated_map:
            if optree in memoization_level: return True
        return False
    def get_memoization(self, optree):
        """ retrieve pre-existing memoization entry """
        for memoization_level in self.generated_map:
            if optree in memoization_level: return memoization_level[optree]
        return None

    def add_memoization(self, optree, code_value):
        """ register memoization value <code_value> for entry <optree> """
        self.generated_map[0][optree] = code_value
        

    # force_variable_storing is not supported
    def generate_expr(self, code_object, optree, folded=True, result_var=None, initial=False, __exact=None, language=None, strip_outer_parenthesis=False, force_variable_storing=False, next_block=None):
        """ code generation function """

        # search if <optree> has already been processed
        if self.has_memoization(optree):
            return self.get_memoization(optree)

        result = None
        # implementation generation
        if isinstance(optree, CodeVariable):
            # adding LLVM variable "%" prefix
            if optree.name[0] != "%":
                optree.name = "%" + optree.name
            result = optree

        elif isinstance(optree, Variable):
            result = CodeVariable("%" + optree.get_tag(), optree.get_precision())

        elif isinstance(optree, Constant):
            precision = optree.get_precision()
            result = CodeExpression(precision.get_gappa_cst(optree.get_value()), precision)

        elif isinstance(optree, Statement):
            # all but last generation
            for op in optree.inputs[:-1]:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, folded = folded, initial = True)
            # last sub-statement generation, with next_block forwarding
            if len(optree.inputs) > 0:
                op = optree.inputs[-1]
                if not self.has_memoization(op):
                    self.generate_expr(
                        code_object, op, folded=folded,
                        initial=True, next_block=next_block
                    )

            return None

        elif isinstance(optree, ConditionBlock):
            return llvm_ir_generate_condition_block(
                self, optree, code_object, folded=folded, language=language,
                next_block=next_block)

        else:
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var, language = self.language)
            # each operation is generated on a separate line

        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            code_object << self.generate_debug_msg(optree, result)


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

    def get_llvm_varname(self, tag):
        return "%" + tag

    def get_function_declaration(self, function_name, output_format, arg_list, final=True, language=C_Code):
        """ generate function declaration code """
        arg_format_list = ", ".join("%s %s" % (llvm_ir_format(inp.get_precision()), self.get_llvm_varname(inp.get_tag())) for inp in arg_list)
        return "define %s @%s(%s)\n" % (llvm_ir_format(output_format), function_name, arg_format_list)

    def generate_untied_statement(self, expression_code, final=True):
        """ generate code for a statement which is not tied (void) """
        final_symbol = ";\n" if final else ""
        return "%s%s" % (expression_code, final_symbol) 

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
        elif isinstance(symbol_object, CodeFunction):
            return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            return "%s\n" % symbol_object.get_declaration()
        elif isinstance(symbol_object, Label):
            return "ERROR<%s:>\n" % symbol_object.name
        else:
            Log.report(Log.Error, "{} decl generation not-implemented".format(symbol_object), error=NotImplementedError)

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
        return ""


    def generate_debug_msg(self, optree, result):
        raise NotImplementedError


