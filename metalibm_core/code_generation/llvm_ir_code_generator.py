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
import copy

from ..core.ml_operations import (
    Addition,
    Variable, Constant, ConditionBlock, Return, TableLoad, Statement,
    SpecificOperation, Conversion, FunctionObject,
    ReferenceAssign, Loop,
)
from ..core.bb_operations import (
    BasicBlockList,
    BasicBlock, ConditionalBranch, UnconditionalBranch,
    PhiNode,
)
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
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

def append_label(code_object, label):
    """ append a new label location at the end of code_object """
    code_object.close_level(footer="", cr="")
    code_object << label << ":"
    code_object.open_level(header="") #, extra_shared_tables=[MultiSymbolTable.VariableSymbol])

def llvm_ir_generate_condition_block(generator, optree, code_object, language, folded=False, next_block=None, initial=False):
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
        is_fallback_else = is_fallback_statement(else_branch)
        is_fallback = is_fallback_if or is_fallback_else
        if is_fallback:
            end_label = get_end_label()
        else:
            end_label = None
    else:
        # there is no else so false-cond requires a fallback blocks
        is_fallback = True
        else_label = get_end_label()
        end_label = else_label

    code_object << "br i1 {cond} , label %{if_label}, label %{else_label}".format(
        cond=cond_code.get(),
        if_label=if_label,
        else_label=else_label
    )

    append_label(code_object, if_label)

    # generating code for if-branch
    if_branch_code = generator.generate_expr(
        code_object, if_branch, folded=folded,
        language=language, next_block=end_label)

    if is_fallback_if:
        code_object << "br label %" << end_label << "\n"

    if else_branch:
        append_label(code_object, else_label)
        else_branch_code = generator.generate_expr(
            code_object, else_branch, folded=folded,
            language=language, next_block=end_label)

    if is_fallback and next_block is None:
        append_label(code_object, end_label)

    return None


def llvm_ir_generate_loop(generator, optree, code_object, language, folded=False, next_block=None, initial=False):
    init_block = optree.get_input(0)
    loop_test = optree.get_input(1)
    loop_body = optree.get_input(2)

    def get_end_label():
        if next_block is None:
            return get_free_label_name(code_object, "loop_end")
        else:
            return next_block

    header_label = get_free_label_name(code_object, "loop_header")
    loop_test_label = get_free_label_name(code_object, "loop_test")
    loop_body_label = get_free_label_name(code_object, "loop_body")
    loop_end_label = get_end_label()

    # generate loop initialization block 
    append_label(code_object, header_label)
    generator.generate_expr(code_object, init_block, folded=folded, language=language)
    append_label(code_object, loop_test_label)
    cond_code = generator.generate_expr(code_object, loop_test, folded=folded, language=language)
    code_object << "br i1 {cond} , label %{loop_body}, label %{loop_end}".format(
        cond=cond_code.get(),
        loop_body=loop_body_label,
        loop_end=loop_end_label,
    )
    append_label(code_object, loop_body_label)
    generator.generate_expr(code_object, loop_body, next_block=loop_body_label, folded=folded, language=language)
    code_object << "br label %" << loop_test_label << "\n"

    if next_block is None:
        append_label(code_object, loop_end_label)
    

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
            if not is_fallback_statement(op):
                return False
        return True
    else:
        return True


def generate_llvm_cst(value, precision, precision_header=True):
    """ Generate LLVM-IR code string to encode numerical value <value> """
    if ML_FP_Format.is_fp_format(precision):
        if FP_SpecialValue.is_special_value(value):
            value = copy.copy(value)
            value.precision = ML_Binary64
            mask = ~(2**(ML_Binary64.get_field_size() - precision.get_field_size()) - 1) 
            return "0x{value:x}".format(
                # special constant must be 64-bit encoded in hexadecimal
                value=(ML_Binary64.get_integer_coding(value) & mask)
            )
        else:
            sollya.settings.display = sollya.decimal
            value_str = str(precision.round_sollya_object(value))
            if not "." in value_str:
                # adding suffix ".0" if numeric value is an integer
                value_str += ".0"
            return value_str
    elif is_std_integer_format(precision):
        return "{value}".format(
            # prec="" if not precision_header else llvm_ir_format(precision),
            value=int(value)
        )
    else:
        Log.report(
            Log.Error,
            "format {} not supported in LLVM-IR generate_llvm_cst",
            precision
        )

def generate_Constant_expr(optree):
    """ generate LLVM-IR code to materialize Constant node """
    assert isinstance(optree, Constant)
    if optree.precision.is_vector_format():
        cst_value = optree.get_value()
        return CodeExpression(
            "<{}>".format(
                ", ".join(generate_llvm_cst(
                        elt_value, optree.precision.get_scalar_format()
                    ) for elt_value in cst_value
                    )
            ),
            optree.precision
        )
    else:
        return CodeExpression(
            generate_llvm_cst(optree.get_value(), optree.precision),
            optree.precision
        )

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
        # map of basic blocks (bb node -> label)
        self.bb_map = {}

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
            result = generate_Constant_expr(optree)
            #result = CodeExpression(precision.get_gappa_cst(optree.get_value()), precision)

        elif isinstance(optree, BasicBlock):
            bb_label = self.get_bb_label(code_object, optree)
            code_object.close_level(footer="", cr="")
            code_object << bb_label << ":"
            code_object.open_level(header="")
            for op in optree.inputs:
                self.generate_expr(code_object, op, folded=folded,
                    initial=True, language=language)
            return None

        elif isinstance(optree, ConditionalBranch):
            cond = optree.get_input(0)
            if_bb = optree.get_input(1)
            else_bb = optree.get_input(2)
            if_label = self.get_bb_label(code_object, if_bb)
            else_label = self.get_bb_label(code_object, else_bb)

            cond_code = self.generate_expr(
                code_object, cond, folded=folded, language=language)

            code_object << "br i1 {cond} , label %{if_label}, label %{else_label}\n".format(
                cond=cond_code.get(),
                if_label=if_label,
                else_label=else_label
            )
            # generating destination bb
            # self.generate_expr(code_object, if_bb, folded=folded, language=language)
            # self.generate_expr(code_object, else_bb, folded=folded, language=language)
            return None

        elif isinstance(optree, UnconditionalBranch):
            dest_bb = optree.get_input(0)
            code_object << "br label %{}\n".format(self.get_bb_label(code_object, dest_bb))
            # generating destination bb
            # self.generate_expr(code_object, dest_bb, folded=folded, language=language)
            return None

        elif isinstance(optree, BasicBlockList):
            for bb in optree.inputs:
                self.generate_expr(code_object, bb, folded=folded, language=language)
            return None

        elif isinstance(optree, Statement):
            Log.report(Log.Error, "Statement are not supported in LLVM-IR codegen"
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", optree)

        elif isinstance(optree, ConditionBlock):
            Log.report(Log.Error, "ConditionBlock are not supported in LLVM-IR codegen"
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", optree)

        elif isinstance(optree, Loop):
            Log.report(Log.Error, "Loop are not supported in LLVM-IR codegen"
                "They must be translated to BB (e.g. through gen_basic_block pass)"
                "faulty node: {}", optree)

        elif isinstance(optree, PhiNode):
            output_var = optree.get_input(0)
            output_var_code = self.generate_expr(
                code_object, output_var, folded=folded, language=language)

            value_list = []
            for input_var, bb_var in zip(optree.get_inputs()[1::2], optree.get_inputs()[2::2]):
                assert isinstance(input_var, Variable)
                assert isinstance(bb_var, BasicBlock)
                input_var = self.generate_expr(
                    code_object, input_var, folded=folded, language=language
                )
                bb_label = self.get_bb_label(code_object, bb_var)
                value_list.append("[{var}, %{bb}]".format(var=input_var.get(), bb=bb_label))

            code_object << "{output_var} = phi {precision} {value_list}\n".format(
                output_var=output_var_code.get(),
                precision=llvm_ir_format(precision=output_var.get_precision()),
                value_list=(", ".join(value_list))
            )

            return None

        elif isinstance(optree, ReferenceAssign):
            output_var = optree.get_input(0)
            result_value = optree.get_input(1)

            # In LLVM it is illegal to assign a constant value, directly to a
            # variable so with insert a dummy add with 0
            if isinstance(result_value, Constant):
                cst_precision = result_value.get_precision()
                result_value = Addition(
                    result_value,
                    Constant(0, precision=cst_precision),
                    precision=cst_precision)

            # TODO/FIXME: fix single static assignation enforcement
            #output_var_code = self.generate_expr(
            #    code_object, output_var, folded=False, language=language
            #)

            result_value_code = self.generate_expr(
                code_object, result_value, folded=folded, result_var="%"+output_var.get_tag(), language=language
            )
            assert isinstance(result_value_code, CodeVariable)
            # code_object << self.generate_assignation(output_var_code.get(), result_value_code.get(), precision=output_var_code.precision)
            # debug msg generation is not supported in LLVM code genrator

            return None

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

    def generate_code_assignation(self, code_object, result_var, expr_code, final=True, original_node=None):
        return self.generate_assignation(result_var, expr_code, final=final)

    def generate_assignation(self, result_var, expression_code, final=True, precision=None):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = ";\n" if final else ""
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
        return "{keyword} {out_format} @{name}({arg_list})\n".format(
            keyword=common_keyword,
            out_format=llvm_ir_format(output_format),
            name=function_name,
            arg_list=arg_format_list)


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
            return "%s\n" % symbol_object.get_LLVM_definition()
            #return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            # declare an external function object
            return "{};\n".format(symbol_object.get_declaration(self, language=LLVM_IR_Code))
        elif isinstance(symbol_object, Label):
            return "ERROR<%s:>\n" % symbol_object.name
        else:
            Log.report(Log.Error, "{} decl generation not-implemented".format(symbol_object), error=NotImplementedError)

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
        return ""


    def generate_debug_msg(self, optree, result):
        Log.report(Log.Error, "[unimplemented LLVM-IR DBG msg] trying to generate debug msg for node {}", optree, error=NotImplementedError)


