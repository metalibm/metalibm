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


from metalibm_core.utility.log_report import Log

# TODO clean long import list
from metalibm_core.core.ml_operations import (
    RaiseException, Variable, Constant, ConditionBlock, Return, Statement,\
    Loop, ExceptionOperation, ClearException, \
    SwitchBlock, FunctionObject, ReferenceAssign, \
    BooleanOperation,
    FunctionType
)
from metalibm_core.core.advanced_operations import PlaceHolder
from metalibm_core.core.ml_table import ML_Table, ML_NewTable
from metalibm_core.core.ml_formats import *

from metalibm_core.utility.debug_utils import ML_Debug

from metalibm_core.code_generation.code_constant import C_Code, OpenCL_Code
from metalibm_core.code_generation.code_element import CodeVariable, CodeExpression
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.generator_utility import DummyTree
from metalibm_core.code_generation.code_generator import RegisterCodeGenerator, CodeGenerator


#  generate_expr's argument
#       code_object: destination code_object to receive resulting source code
#       optree/node: operation graph node to generate
#       inlined: if True <optree/node> expression must be generated as inlined CodeExpression
#                not stored in a CodeVariable
#       force_variable_storing: force the expression result to be assigned to a
#                               CodeVariable, the CodeVariable is returned as
#                               generated_expr results
#       result_var: if force_variable_storing is set, may indicate which
#                   variable must be used to store the expression result
#       lvalue: indicates that the expression is used a lvalue (left-value)



@RegisterCodeGenerator([C_Code, OpenCL_Code])
class CCodeGenerator(CodeGenerator):
    language = C_Code

    """ C language code generator """
    def __init__(self, processor, declare_cst = True, disable_debug = False, libm_compliant = False, default_rounding_mode = ML_GlobalRoundMode, default_silent = None, language = C_Code, decorate_code=False):
        self.memoization_map = self.get_empty_memoization_map()
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.fp_context = FP_Context(rounding_mode = default_rounding_mode, silent = default_silent)
        self.language = language
        self.decorate_code = decorate_code
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

    def get_empty_memoization_map(self):
        """ build an initialized structure for the memoization map """
        return [{}]
    def clear_memoization_map(self):
        """ Clear the content of the meoization map """
        self.memoization_map = self.get_empty_memoization_map()


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
    def generate_expr(self, code_object,
                            optree,
                            result_var=None,
                            language=None,
                            force_variable_storing=False,
                            lvalue=False,
                            inlined=None):
        """ code generation function
            Args:
                :param optree: Operation node to generate code for
                :type optree: ML_Operation
                :param result_var: destination variable (if specified)
                :type result_var: CodeVariable
                :param language: source code language
                :param force_variable_storing: TBD
                :param lvalue: indicate expression is a lvalue
                :type lvalue: bool
                :param inlined: indicate that the result must be an inlined CodeExpression
                :type inlined: bool
        """
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
                language=language,
            )

            switch_value_code = self.generate_expr(
                code_object, switch_value, language=language
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
                language=language
              )
              code_object.close_level()
            code_object << "}\n"

            return None

        elif isinstance(optree, ReferenceAssign):
            output_var = optree.inputs[0]
            result_value = optree.inputs[1]

            output_var_code   = self.generate_expr(
                code_object, output_var, lvalue=True, language=language
            )

            if isinstance(result_value, Constant):
              # generate assignation
              result_value_code = self.generate_expr(
                code_object, result_value, inlined=True, language = language
              )
              code_object << self.generate_assignation(
                output_var_code.get(), result_value_code.get()
              )
            else:
              result_value_code = self.generate_expr(
                code_object, result_value, language=language
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

            self.generate_expr(code_object, init_statement, language = language)
            code_object << "\nfor (;%s;)" % self.generate_expr(code_object, exit_condition, inlined=True, language=language).get()
            code_object.open_level()
            self.generate_expr(code_object, loop_body, language=language)
            code_object.close_level()

            return None

        elif isinstance(optree, ConditionBlock):
            condition = optree.inputs[0]
            if_branch = optree.inputs[1]
            else_branch = optree.inputs[2] if len(optree.inputs) > 2 else None

            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), language=language)

            cond_code = self.generate_expr(code_object, condition, language=language)
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
            if_branch_code = self.generate_expr(code_object, if_branch, language=language)
            code_object.close_level(cr = "")
            self.close_memoization_level()
            if else_branch:
                code_object << " else "
                code_object.open_level()
                self.open_memoization_level()
                else_branch_code = self.generate_expr(code_object, else_branch, language=language)
                code_object.close_level()
                self.close_memoization_level()
            else:
                code_object << "\n"

            return None

        elif isinstance(optree, Return):
            if len(optree.inputs) == 0:
                # void return
                code_object << "return;\n"

            else:
                return_result = optree.inputs[0]
                return_code = self.generate_expr(code_object, return_result, language=language)
                code_object << "return %s;\n" % return_code.get()
                return None #return_code

        elif isinstance(optree, ExceptionOperation):
            if isinstance(optree, (RaiseException, ClearException)):
                result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, result_var=result_var, language=language)
                # TODO/FIXME: need cleanup
                if result_code != None:
                    code_object << "%s;\n" % result_code.get()
                if isinstance(optree, RaiseException):
                    # todo: exception mode specilization
                    if self.libm_compliant:
                        # libm compliant exception management
                        code_object.add_header("support_lib/ml_libm_compatibility.h")
                        return_value = self.generate_expr(code_object, optree.get_return_value(), language=language)
                        arg_value = self.generate_expr(code_object, optree.get_arg_value(), language=language)
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
                return None
            else:
                result = self.processor.generate_expr(self, code_object, optree, optree.inputs, result_var=result_var, language=language)

        elif isinstance(optree, PlaceHolder):
            head = optree.get_input(0)
            for tail_node in optree.inputs[1:]:
                if not self.has_memoization(tail_node):
                    self.generate_expr(code_object, tail_node, language=language)

            # generate PlaceHolder's main_value
            head_code = self.generate_expr(code_object, head, language=language)
            return head_code

        elif isinstance(optree, Statement):
            for op in optree.inputs:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, language = language)

            return None
        elif isinstance(optree, Constant):
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, [], generate_pre_process = generate_pre_process, result_var=result_var, language=language)

        else:
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, generate_pre_process = generate_pre_process, result_var=result_var, language=language, inlined=inlined, lvalue=lvalue)

        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            code_object << self.generate_debug_msg(optree, result, code_object)


        #if initial and not isinstance(result, CodeVariable) and not result is None:
        #    final_var = result_var if result_var else code_object.get_free_var_name(optree.get_precision(), prefix = "result", declare = True)
        #    code_object << self.generate_assignation(final_var, result.get())
        #    return CodeVariable(final_var, optree.get_precision())

        return result

    def generate_clear_exception(self, code_generator, code_object, optree, var_arg_list, language = None, **kwords): 
        #generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
        self.generate_expr(code_object, ClearException(), language = language)

    def generate_code_assignation(self, code_object, result_var, expr_code, final=True, original_node=None):
        if self.decorate_code and not original_node is None and not isinstance(original_node, DummyTree):
            code_decoration = original_node.get_str(depth=2, display_precision=True)
            code_object.add_multiline_comment(code_decoration)
        return self.generate_assignation(result_var, expr_code, final=final)


    def generate_assignation(self, result_var, expression_code, final = True):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = ";\n" if final else ""
        return "%s = %s%s" % (result_var, expression_code, final_symbol) 

    def generate_untied_statement(self, expression_code, final = True):
      final_symbol = ";\n" if final else ""
      return "%s%s" % (expression_code, final_symbol)


    def generate_declaration(self, symbol, symbol_object, initial=True, final=True, static_const=True):
        # TODO extract attributes from symbol_object
        if isinstance(symbol_object, Constant):
            format_str = symbol_object.get_precision().get_code_name(language=self.language)
            attributes = ["static const"] if static_const else []
            prefix = (attributes + [format_str]) if initial else []
            final_symbol = ";\n" if final else ""
            return "{prefix} = {value}{final}".format(
                prefix=" ".join(prefix + [symbol]),
                value=symbol_object.get_precision().get_cst(symbol_object.get_value(), language=self.language),
                final=final_symbol,
            )

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
            return "%s\n" % symbol_object.get_declaration(self)
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
                return_format=arg_type.output_format.get_name(language=language),
                arg_tag=arg_tag,
                arg_format_list=",".join([self.get_fct_arg_decl(sub_arg_prec, "", language) for sub_arg_prec in arg_type.arg_list_precision])
            )

        else:
            return "{} {}".format(arg_type.get_name(language=language), arg_tag)

    def get_function_definition(self, fct_type, final=True, language=C_Code, arg_list=None):
        """ C function definition prolog is the same as the function declaration """
        return self.get_function_declaration(fct_type, final, language, arg_list)

    def get_function_declaration(self, fct_type, final=True, language=C_Code, arg_list=None):
        """ generate a C code function declaration, if @p arg_list is set
            generated named parameters, else unamed ones """
        language = self.language if language is None else language
        attributes = " ".join(fct_type.attributes)
        if arg_list:
            arg_format_list = ", ".join("%s %s" % (inp.get_precision().get_name(language=language), inp.get_tag()) for inp in arg_list)
        else:
            arg_format_list = ", ".join(input_format.get_name(language=language) for input_format in fct_type.arg_list_precision)
        final_symbol = ";" if final else ""
        return "{attributes}{attr_del}{fct_ret_format} {fct_name}({arg_format_list}){final}".format(
            attributes=attributes,
            # delimiter between attribute and function's return format
            attr_del=" " if attributes != "" else "",
            fct_ret_format=fct_type.output_format.get_name(language=language),
            fct_name=fct_type.name,
            arg_format_list=arg_format_list,
            final=final_symbol)

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
