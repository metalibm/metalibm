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


from ..utility.log_report import Log

# TODO clean long import list
from ..core.ml_operations import (
    Variable, Constant, ConditionBlock, Return, TableLoad, Statement,\
    Loop, SpecificOperation, ExceptionOperation, ClearException, \
    NoResultOperation, SwitchBlock, FunctionObject, ReferenceAssign, \
    BooleanOperation,
    FunctionType
)
from ..core.ml_table import ML_Table, ML_NewTable
from ..core.ml_formats import *
from ..core.attributes import ML_Debug
from .code_constant import C_Code
from .code_element import CodeVariable, CodeExpression
from .code_function import CodeFunction


class CCodeGenerator(object):
    language = C_Code

    """ C language code generator """
    def __init__(self, processor, declare_cst = True, disable_debug = False, libm_compliant = False, default_rounding_mode = ML_GlobalRoundMode, default_silent = None, language = C_Code):
        self.memoization_map = [{}]
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.fp_context = FP_Context(rounding_mode = default_rounding_mode, silent = default_silent)
        self.language = language
        Log.report(Log.Info, "CCodeGenerator initialized with language: %s" % self.language)


    def check_fp_context(self, fp_context, rounding_mode, silent):
        """ check required fp_context compatibility with
            current fp context """
        return rounding_mode == fp_context.get_rounding_mode() and silent == fp_context.get_silent()

    def adapt_fp_context(self, code_object, old_fp_context, new_fp_context):
        if old_fp_context.get_rounding_mode() != new_fp_context.get_rounding_mode():
            if old_fp_context.get_rounding_mode() == ML_GlobalRoundMode:
                # TODO
                pass


    def get_unknown_precision(self):
        return None


    def open_memoization_level(self):
        self.memoization_map.insert(0, {})
    def close_memoization_level(self):
        self.memoization_map.pop(0)


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

    # force_variable_storing is not supported yet
    def generate_expr(self, code_object, optree, folded = True, result_var = None, initial = False, language = None, force_variable_storing = False):
        """ code generation function """
        language = self.language if language is None else language

        # search if <optree> has already been processed
        if self.has_memoization(optree):
            return self.get_memoization(optree)

        result = None
        # implementation generation
        if isinstance(optree, CodeVariable):
            result = optree

        elif isinstance(optree, Variable):
            if optree.get_var_type() is Variable.Local:
              final_var =  code_object.get_free_var_name(optree.get_precision(), prefix = optree.get_tag(), declare = True)
              result = CodeVariable(final_var, optree.get_precision())
            else:
              result = CodeVariable(optree.get_tag(), optree.get_precision())

        elif isinstance(optree, ML_NewTable):
            # Implementing LeafNode ML_NewTable generation support
            table = optree
            tag = table.get_tag()
            table_name = code_object.declare_table(
                table, prefix = tag if tag != None else "table"
            )
            result = CodeVariable(table_name, table.get_precision())

        elif isinstance(optree, SwitchBlock):
            switch_value = optree.inputs[0]
            # generating pre_statement
            self.generate_expr(
                code_object, optree.get_pre_statement(),
                folded = folded, language = language
            )

            switch_value_code = self.generate_expr(
                code_object, switch_value, folded = folded, language = language
            )
            case_map = optree.get_case_map()

            code_object << "\nswitch(%s) {\n" % switch_value_code.get()
            for case in case_map:
              case_value = case
              case_statement = case_map[case]
              if isinstance(case_value, tuple):
                for sub_case in case:
                  code_object << "case %s:\n" % sub_case
              else:
                code_object << "case %s:\n" % case
              code_object.open_level()
              self.generate_expr(
                code_object, case_statement,
                folded = folded, language = language
              )
              code_object.close_level()
            code_object << "}\n"

            return None

        elif isinstance(optree, ReferenceAssign):
            output_var = optree.inputs[0]
            result_value = optree.inputs[1]

            output_var_code   = self.generate_expr(
                code_object, output_var, folded = False, language = language
            )

            if isinstance(result_value, Constant):
              # generate assignation
              result_value_code = self.generate_expr(
                code_object, result_value, folded = folded, language = language
              )
              code_object << self.generate_assignation(
                output_var_code.get(), result_value_code.get()
              )
            else:
              result_value_code = self.generate_expr(
                code_object, result_value, folded = folded, language = language
              )
              code_object << self.generate_assignation(output_var_code.get(), result_value_code.get())
              if optree.get_debug() and not self.disable_debug:
                code_object << self.generate_debug_msg(result_value, result_value_code, code_object, debug_object = optree.get_debug())

            #code_object << self.generate_assignation(output_var_code.get(), result_value_code.get())
            #code_object << output_var.get_precision().generate_c_assignation(output_var_code, result_value_code)
            
            return None

        elif isinstance(optree, Loop):
            init_statement = optree.inputs[0]
            exit_condition = optree.inputs[1]
            loop_body      = optree.inputs[2]

            self.generate_expr(code_object, init_statement, folded = folded, language = language)
            code_object << "\nfor (;%s;)" % self.generate_expr(code_object, exit_condition, folded = False, language = language).get()
            code_object.open_level()
            self.generate_expr(code_object, loop_body, folded = folded, language = language)
            code_object.close_level()

            return None

        elif isinstance(optree, ConditionBlock):
            condition = optree.inputs[0]
            if_branch = optree.inputs[1]
            else_branch = optree.inputs[2] if len(optree.inputs) > 2 else None

            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), folded = folded, language = language)

            cond_code = self.generate_expr(code_object, condition, folded = folded, language = language)
            if isinstance(condition, BooleanOperation):
              cond_likely = condition.get_likely()
            else:
              # TODO To be refined (for example Constant(True)
              #      should be associated with likely True
              cond_likely = None
              Log.report(
                Log.Warning,
                " The following condition has no (usable) likely attribute: {}",
                condition,
              )
            if cond_likely in [True, False]:
                code_object << "\nif (__builtin_expect(%s, %d)) " % (cond_code.get(), {True: 1, False: 0}[condition.get_likely()])
            else:
                code_object << "\nif (%s) " % cond_code.get()
            self.open_memoization_level()
            code_object.open_level()
            #if_branch_code = self.processor.generate_expr(self, code_object, if_branch, if_branch.inputs, folded)
            if_branch_code = self.generate_expr(code_object, if_branch, folded = folded, language = language)
            code_object.close_level(cr = "")
            self.close_memoization_level()
            if else_branch:
                code_object << " else "
                code_object.open_level()
                self.open_memoization_level()
                else_branch_code = self.generate_expr(code_object, else_branch, folded = folded, language = language)
                code_object.close_level()
                self.close_memoization_level()
            else:
                code_object << "\n"

            return None

        elif isinstance(optree, Return):
            return_result = optree.inputs[0]
            return_code = self.generate_expr(code_object, return_result, folded = folded, language = language)
            code_object << "return %s;\n" % return_code.get()
            return None #return_code

        elif isinstance(optree, ExceptionOperation):
            if optree.get_specifier() in [ExceptionOperation.RaiseException, ExceptionOperation.ClearException, ExceptionOperation.RaiseReturn]:
                result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = False, result_var = result_var, language = language)
                code_object << "%s;\n" % result_code.get()
                if optree.get_specifier() == ExceptionOperation.RaiseReturn:
                    if self.libm_compliant:
                        # libm compliant exception management
                        code_object.add_header("support_lib/ml_libm_compatibility.h")
                        return_value = self.generate_expr(code_object, optree.get_return_value(), folded = folded, language = language)
                        arg_value = self.generate_expr(code_object, optree.get_arg_value(), folded = folded, language = language)
                        function_name = optree.function_name
                        exception_list = [op.get_value() for op in optree.inputs]
                        if ML_FPE_Inexact in exception_list:
                            exception_list.remove(ML_FPE_Inexact)

                        if len(exception_list) > 1:
                            raise NotImplementedError
                        if ML_FPE_Overflow in exception_list:
                            code_object << "return ml_raise_libm_overflowf(%s, %s, \"%s\");\n" % (return_value.get(), arg_value.get(), function_name)
                        elif ML_FPE_Underflow in exception_list:
                            code_object << "return ml_raise_libm_underflowf(%s, %s, \"%s\");\n" % (return_value.get(), arg_value.get(), function_name)
                        elif ML_FPE_Invalid in exception_list:
                            code_object << "return %s;\n" % return_value.get() 
                    else:
                        return_precision = optree.get_return_value().get_precision()
                        self.generate_expr(code_object, Return(optree.get_return_value(), precision = return_precision), folded = folded, language = language)
                return None
            else:
                result = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var, language = language)

        elif isinstance(optree, NoResultOperation):
            result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = False, result_var = result_var, language = language)
            code_object << "%s;\n" % result_code.get() 
            return None

        elif isinstance(optree, Statement):
            for op in optree.inputs:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, folded = folded, initial = True, language = language)

            return None
        elif isinstance(optree, Constant):
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, [], generate_pre_process = generate_pre_process, folded = folded, result_var = result_var, language = language)

        else:
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, generate_pre_process = generate_pre_process, folded = folded, result_var = result_var, language = language)

        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            code_object << self.generate_debug_msg(optree, result, code_object)
            

        if initial and not isinstance(result, CodeVariable) and not result is None:
            final_var = result_var if result_var else code_object.get_free_var_name(optree.get_precision(), prefix = "result", declare = True)
            code_object << self.generate_assignation(final_var, result.get())
            return CodeVariable(final_var, optree.get_precision())

        return result

    def generate_clear_exception(self, code_generator, code_object, optree, var_arg_list, language = None, **kwords): 
        #generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
        self.generate_expr(code_object, ClearException(), language = language)

    def generate_code_assignation(self, code_object, result_var, expr_code, final=True):
        return self.generate_assignation(result_var, expr_code, final=final)


    def generate_assignation(self, result_var, expression_code, final = True):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = ";\n" if final else ""
        return "%s = %s%s" % (result_var, expression_code, final_symbol) 

    def generate_untied_statement(self, expression_code, final = True):
      final_symbol = ";\n" if final else ""
      return "%s%s" % (expression_code, final_symbol) 


    def generate_declaration(self, symbol, symbol_object, initial = True, final = True):
        if isinstance(symbol_object, Constant):
            initial_symbol = (symbol_object.get_precision().get_code_name(language = self.language) + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s = %s%s" % (initial_symbol, symbol, symbol_object.get_precision().get_cst(symbol_object.get_value(), language = self.language), final_symbol) 

        elif isinstance(symbol_object, Variable):
            initial_symbol = (symbol_object.get_precision().get_code_name(language = self.language) + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s%s" % (initial_symbol, symbol, final_symbol) 

        elif isinstance(symbol_object, ML_Table):
            # TODO: check @p initial effect
            if symbol_object.is_empty():
              initial_symbol = (symbol_object.get_definition(symbol, final = "", language = self.language)) if initial else ""
              return "{};\n".format(initial_symbol)
            else:
              initial_symbol = (symbol_object.get_definition(symbol, final = "", language = self.language) + " ") if initial else ""
              table_content_init = symbol_object.get_content_init(language = self.language)
              return "%s = %s;\n" % (initial_symbol, table_content_init)

        elif isinstance(symbol_object, CodeFunction):
            return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            return "%s\n" % symbol_object.get_declaration()
        else:
            Log.report(Log.Error, "{} decl generation not-implemented".format(symbol_object), error=NotImplementedError)

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
      if isinstance(symbol_object, Constant) or isinstance(symbol_object, Variable):
        final_symbol = ";\n" if final else ""
        init_code = symbol_object.get_precision().generate_initialization(symbol, symbol_object, language = self.language)
        if init_code != None:
          return "%s%s" % (init_code, final_symbol)
        else:
          return ""
      else:
        return ""


    def get_fct_arg_decl(self, arg_type, arg_tag, language=C_Code):
        """ Generate function argument declaration code """
        if isinstance(arg_type, FunctionType):
            return "{return_format} ({arg_tag})({arg_format_list})".format(
                return_format=arg_type.output_precision.get_name(language=language),
                arg_tag=arg_tag,
                arg_format_list=",".join([self.get_fct_arg_decl(sub_arg_prec, "", language) for sub_arg_prec in arg_type.arg_list_precision])
            )

        else:   
            return "{} {}".format(arg_type.get_name(language=language), arg_tag)



    def get_function_declaration(self, function_name, output_format, arg_list, final=True, language=C_Code):
        """ generate function declaration code """
        arg_format_list = ", ".join(self.get_fct_arg_decl(inp.get_precision(), inp.get_tag()) for inp in arg_list)
        final_symbol = ";" if final else ""
        return "%s %s(%s)%s" % (output_format.get_name(language=language), function_name, arg_format_list, final_symbol)

    def generate_debug_msg(self, optree, result, code_object, debug_object = None):
        debug_object = optree.get_debug() if debug_object is None else debug_object
        debug_object = debug_object.select_object(optree) if isinstance(debug_object, ML_Debug) else debug_object
        # adding required headers
        if isinstance(debug_object, ML_Debug):
            for header in debug_object.get_require_header():
              code_object.add_header(header)
        precision = optree.get_precision()
        display_format = debug_object.get_display_format(precision.get_display_format(language = self.language)) if isinstance(debug_object, ML_Debug) else precision.get_display_format(language = self.language)
        display_result = debug_object.get_pre_process(result.get(), optree) if isinstance(debug_object, ML_Debug) else result.get()
        debug_msg = "#ifdef ML_DEBUG\n"
        debug_msg += """printf("%s: %s\\n", %s);\n""" % (optree.get_tag(), display_format, display_result)
        debug_msg += "#endif\n"
        return debug_msg
