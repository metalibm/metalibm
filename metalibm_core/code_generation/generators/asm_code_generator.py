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


from metalibm_core.core.ml_operations import (
    Addition,
    Variable, Constant, ConditionBlock, Return, TableLoad, Statement,
    SpecificOperation, Conversion, FunctionObject,
    ReferenceAssign, Loop,
    is_leaf_node,
    TableStore,
)
from metalibm_core.core.bb_operations import (
    BasicBlockList,
    BasicBlock, ConditionalBranch, UnconditionalBranch,
    PhiNode,
)
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.code_element import CodeVariable, CodeExpression
from metalibm_core.code_generation.code_object import MultiSymbolTable

from metalibm_core.utility.log_report import Log

from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.llvm_utils import llvm_ir_format
from metalibm_core.code_generation.code_generator import CodeGenerator
from metalibm_core.code_generation.code_constant import ASM_Code

from metalibm_core.code_generation.code_generator import (
    CodeGenerator, RegisterCodeGenerator)

from metalibm_core.core.bb_operations import SequentialBlock
from metalibm_core.core.machine_operations import (
    MachineRegister, RegisterAssign)
from metalibm_core.code_generation.asm_utility import (
    Label, get_free_label_name)


@RegisterCodeGenerator([ASM_Code])
class AsmCodeGenerator(CodeGenerator):
    """ ASM language code generator """
    language = LLVM_IR_Code

    def __init__(self, processor, declare_cst=True, disable_debug=False,
                 libm_compliant=False, language=ASM_Code,
                 decorate_code=False):
        # on level for each of exact_mode possible values
        self.generated_map = self.get_empty_memoization_map()
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.language = language
        self.end_label = None
        # map of basic blocks (bb node -> label)
        self.bb_map = {}
        if decorate_code: Log.report(Log.Error, "decorate_code option is not supported in AsmCodeGenerator")

    def get_empty_memoization_map(self):
        """ build an initialized structure for the memoization map """
        return [{}]
    def clear_memoization_map(self):
        """ Clear the content of the meoization map """
        self.generated_map = self.get_empty_memoization_map()


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

    def get_bb_label(self, code_object, bb):
        if bb in self.bb_map:
            return self.bb_map[bb]
        else:
            new_label = get_free_label_name(code_object, bb.get_tag() or "BB")
            self.bb_map[bb] = new_label
            return new_label

    def generate_expr(self, code_object, node, folded=True, result_var=None,
                      initial=False, __exact=None, language=None,
                      strip_outer_parenthesis=False,
                      force_variable_storing=False, next_block=None):
        """ code generation function,
            Notes: force variable storing is not supported """
        assert not force_variable_storing

        # search if <optree> has already been processed
        if self.has_memoization(node):
            return self.get_memoization(node)

        result = None
        # implementation generation
        if isinstance(node, MachineRegister):
            result = self.processor.generate_register(node)

        elif isinstance(node, Constant):
            precision = node.get_precision()
            result = CodeExpression(self.processor.generate_constant_expr(node), precision)

        elif isinstance(node, BasicBlock):
            if isinstance(node, SequentialBlock):
                # not really a BasicBlock, just a sequence of instruction
                pass
            else:
                bb_label = self.get_bb_label(code_object, node)
                code_object.close_level(footer="", cr="")
                code_object << bb_label << ":"
                code_object.open_level(header="")
            for op in node.inputs:
                self.generate_expr(code_object, op, folded=folded,
                    initial=True, language=language)
            return None

        elif isinstance(node, ConditionalBranch):
            assert len(node.inputs) == 2
            self.processor.generate_conditional_branch(self, code_object, node)
            return None

        elif isinstance(node, UnconditionalBranch):
            assert len(node.inputs) == 1
            self.processor.generate_unconditional_branch(self, code_object, node)
            return None

        elif isinstance(node, BasicBlockList):
            for bb in node.inputs:
                self.generate_expr(code_object, bb, folded=folded, language=language)
            return None

        elif isinstance(node, (Statement, ConditionBlock, Loop)):
            Log.report(Log.Error, "{} class nodes are not supported in LLVM-IR codegen"
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", node.__class__, node)

        elif isinstance(node, PhiNode):
            raise NotImplementedError

        elif isinstance(node, RegisterAssign):
            output_reg = self.generate_expr(code_object, node.get_input(0))
            result_value = node.get_input(1)

            #result_value_code = self.generate_expr(
            #    code_object, result_value, folded=folded, result_var=output_reg,
            #    language=language)
            if is_leaf_node(result_value):
                value_inputs = None
            else:
                value_inputs = result_value.inputs
            result_value_code = self.processor.generate_expr(self, code_object, result_value,
                                                  value_inputs, folded=folded,
                                                  result_var=output_reg,
                                                  language=self.language)

            return None
        elif isinstance(node, (Return, TableStore)):
            result = self.processor.generate_expr(self, code_object, node,
                                                  node.inputs, folded=folded,
                                                  result_var=None,
                                                  language=self.language)

        else:
            Log.report(Log.Error, "following node is not support in asm_code_generator: {}", node)
            # each operation is generated on a separate line

        # registering result into memoization table
        self.add_memoization(node, result)

        # debug management
        if node.get_debug() and not self.disable_debug:
            Log.report(Log.Warning, "debug is not support in {}", self.__class__)

        return result

    def generate_code_assignation(self, code_object, result_var, expr_code, final=True, original_node=None):
        return self.generate_assignation(result_var, expr_code, final=final)

    def generate_assignation(self, result_var, expression_code, final=True, precision=None):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = "\n" if final else ""
        format_symbol = llvm_ir_format(precision) if precision else ""
        return "{result} = {format_str} {expr}{final}".format(
            result=result_var,
            expr=expression_code,
            format_str=format_symbol,
            final=final_symbol
        )

    def get_llvm_varname(self, tag):
        return "%" + tag

    def get_function_definition(self, fct_type, final=True, language=LLVM_IR_Code, arg_list=None):
        """ generate function definition prolog code """
        return self.get_function_common(fct_type, "define", final, language, arg_list)
    def get_function_declaration(self, fct_type, final=True, language=LLVM_IR_Code, arg_list=None):
        """ generate function declaration code """
        return self.get_function_common(fct_type, "declare", final, language, arg_list)

    def get_function_common(self, fct_type, common_keyword, final=True, language=LLVM_IR_Code, arg_list=None):
        """ :param common_keyword: llvm-ir specialization function declaration or definition """
        if arg_list:
            arg_format_list = ", ".join("%s %s" % (llvm_ir_format(inp.get_precision()), self.get_llvm_varname(inp.get_tag())) for inp in arg_list)
        else:
            arg_format_list = ", ".join(input_format.get_name(language=language) for input_format in fct_type.arg_list_precision)
        function_name = fct_type.name
        output_format = fct_type.output_format
        return "{name}: // {out_format} ({arg_list})\n".format(
            keyword=common_keyword,
            out_format=llvm_ir_format(output_format),
            name=function_name,
            arg_list=arg_format_list)


    def generate_untied_statement(self, expression_code, final=True):
        """ generate code for a statement which is not tied (void) """
        final_symbol = "\n" if final else ""
        return "%s%s" % (expression_code, final_symbol) 

    def generate_declaration(self, symbol, symbol_object, initial = True, final = True):
        if isinstance(symbol_object, Constant):
            initial_symbol = ""#(symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = "\n" if final else ""
            return "%s%s = %s%s" % (initial_symbol, symbol, symbol_object.get_precision().get_gappa_cst(symbol_object.get_value()), final_symbol) 
        elif isinstance(symbol_object, Variable):
            initial_symbol = ""#(symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = "\n" if final else ""
            return "%s%s%s" % (initial_symbol, symbol, final_symbol) 
        elif isinstance(symbol_object, ML_Table):
            raise NotImplementedError
        elif isinstance(symbol_object, CodeFunction):
            return "%s\n" % symbol_object.get_LLVM_definition()
            #return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            # declare an external function object
            return "{}\n".format(symbol_object.get_declaration(self, language=LLVM_IR_Code))
        elif isinstance(symbol_object, Label):
            return "ERROR<%s:>\n" % symbol_object.name
        else:
            Log.report(Log.Error, "{} decl generation not-implemented".format(symbol_object), error=NotImplementedError)

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
        return ""


    def generate_debug_msg(self, optree, result):
        Log.report(Log.Error, "[unimplemented LLVM-IR DBG msg] trying to generate debug msg for node {}", optree, error=NotImplementedError)


