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

from metalibm_core.code_generation.llvm_utils import llvm_ir_format


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
    def generate_expr(self, code_object, optree, folded = True, result_var = None, initial = False, __exact = None, language = None, strip_outer_parenthesis = False, force_variable_storing = False):
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
            for op in optree.inputs:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, folded = folded, initial = True)

            return None

        else:
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var, language = self.language)

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
        final_symbol = "" if final else ""
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
        else:
            raise NotImplementedError

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
        return ""


    def generate_debug_msg(self, optree, result):
        raise NotImplementedError


