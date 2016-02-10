# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 24th, 2013
# last-modified:    Apr 2nd,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from ..utility.common import ML_NotImplemented
from ..utility.log_report import Log

from ..core.ml_operations import Variable, Constant, ConditionBlock, Return, TableLoad, Statement, Loop, SpecificOperation, ExceptionOperation, ClearException, NoResultOperation, SwitchBlock, FunctionObject, ReferenceAssign
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from ..core.attributes import ML_Debug
from .code_constant import C_Code
from .code_element import CodeVariable, CodeExpression
from .code_function import CodeFunction


class CCodeGenerator: 
    language = C_Code

    """ C language code generator """
    def __init__(self, processor, declare_cst = True, disable_debug = False, libm_compliant = False, default_rounding_mode = ML_GlobalRoundMode, default_silent = None):
        self.memoization_map = [{}]
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.fp_context = FP_Context(rounding_mode = default_rounding_mode, silent = default_silent)


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


    def generate_expr(self, code_object, optree, folded = True, result_var = None, initial = False):
        """ code generation function """

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

        elif isinstance(optree, Constant):
            precision = optree.get_precision()
            if self.declare_cst or optree.get_precision().is_cst_decl_required():
                cst_varname = code_object.declare_cst(optree)
                result = CodeVariable(cst_varname, precision)
            else:
                if precision is ML_Integer:
                  result = CodeExpression("%d" % optree.get_value(), precision)
                else:
                  try:
                      result = CodeExpression(precision.get_c_cst(optree.get_value()), precision)
                  except:
                    Log.report(Log.Error, "Error during get_c_cst call for Constant: %s " % optree.get_str(display_precision = True)) # Exception print

        elif isinstance(optree, TableLoad):
            # declaring table
            table = optree.inputs[0]
            tag = table.get_tag()
            table_name = code_object.declare_table(table, prefix = tag if tag != None else "table") 

            index_code = [self.generate_expr(code_object, index_op, folded = folded).get() for index_op in optree.inputs[1:]]

            result = CodeExpression("%s[%s]" % (table_name, "][".join(index_code)), optree.inputs[0].get_storage_precision())

            # manually enforcing folding
            if folded:
                prefix = optree.get_tag(default = "tmp")
                result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
                code_object << self.generate_assignation(result_varname, result.get()) 
                result = CodeVariable(result_varname, optree.get_precision())


        elif isinstance(optree, SwitchBlock):
            switch_value = optree.inputs[0]
            
            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), folded = folded)

            switch_value_code = self.generate_expr(code_object, switch_value, folded = folded)
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
              self.generate_expr(code_object, case_statement, folded = folded) 
              code_object.close_level()
            code_object << "}\n"

            return None

        elif isinstance(optree, ReferenceAssign):
            output_var = optree.inputs[0]
            result_value = optree.inputs[1]

            output_var_code   = self.generate_expr(code_object, output_var, folded = False)

            if isinstance(result_value, Constant):
              # generate assignation
              result_value_code = self.generate_expr(code_object, result_value, folded = folded)
              code_object << self.generate_assignation(output_var_code.get(), result_value_code.get())
            else:
              result_value_code = self.generate_expr(code_object, result_value, folded = folded)
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

            self.generate_expr(code_object, init_statement, folded = folded)
            code_object << "\nfor (;%s;)" % self.generate_expr(code_object, exit_condition, folded = False).get()
            code_object.open_level()
            self.generate_expr(code_object, loop_body, folded = folded)
            code_object.close_level()

            return None

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
            return_result = optree.inputs[0]
            return_code = self.generate_expr(code_object, return_result, folded = folded)
            code_object << "return %s;\n" % return_code.get()
            return None #return_code

        elif isinstance(optree, ExceptionOperation):
            if optree.get_specifier() in [ExceptionOperation.RaiseException, ExceptionOperation.ClearException, ExceptionOperation.RaiseReturn]:
                result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = False, result_var = result_var)
                code_object << "%s;\n" % result_code.get()
                if optree.get_specifier() == ExceptionOperation.RaiseReturn:
                    if self.libm_compliant:
                        # libm compliant exception management
                        code_object.add_header("support_lib/ml_libm_compatibility.h")
                        return_value = self.generate_expr(code_object, optree.get_return_value(), folded = folded)
                        arg_value = self.generate_expr(code_object, optree.get_arg_value(), folded = folded)
                        function_name = optree.function_name
                        exception_list = [op.get_value() for op in optree.inputs]
                        if ML_FPE_Inexact in exception_list:
                            exception_list.remove(ML_FPE_Inexact)

                        if len(exception_list) > 1:
                            raise ML_NotImplemented()
                        if ML_FPE_Overflow in exception_list:
                            code_object << "return ml_raise_libm_overflowf(%s, %s, \"%s\");\n" % (return_value.get(), arg_value.get(), function_name)
                        elif ML_FPE_Underflow in exception_list:
                            code_object << "return ml_raise_libm_underflowf(%s, %s, \"%s\");\n" % (return_value.get(), arg_value.get(), function_name)
                        elif ML_FPE_Invalid in exception_list:
                            code_object << "return %s;\n" % return_value.get() 
                    else:
                        return_precision = optree.get_return_value().get_precision()
                        self.generate_expr(code_object, Return(optree.get_return_value(), precision = return_precision), folded = folded)
                return None
            else:
                result = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = folded, result_var = result_var)

        elif isinstance(optree, NoResultOperation):
            result_code = self.processor.generate_expr(self, code_object, optree, optree.inputs, folded = False, result_var = result_var)
            code_object << "%s;\n" % result_code.get() 
            return None

        elif isinstance(optree, Statement):
            for op in optree.inputs:
                if not self.has_memoization(op):
                    self.generate_expr(code_object, op, folded = folded, initial = True)

            return None

        else:
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, generate_pre_process = generate_pre_process, folded = folded, result_var = result_var)

        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            code_object << self.generate_debug_msg(optree, result, code_object)
            

        if initial and not isinstance(result, CodeVariable):
            final_var = result_var if result_var else code_object.get_free_var_name(optree.get_precision(), prefix = "result", declare = True)
            code_object << self.generate_assignation(final_var, result.get())
            return CodeVariable(final_var, optree.get_precision())

        return result

    def generate_clear_exception(self, code_generator, code_object, optree, var_arg_list, **kwords): 
        #generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
        self.generate_expr(code_object, ClearException())


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
            initial_symbol = (symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s = %s%s" % (initial_symbol, symbol, symbol_object.get_precision().get_c_cst(symbol_object.get_value()), final_symbol) 

        elif isinstance(symbol_object, Variable):
            initial_symbol = (symbol_object.get_precision().get_c_name() + " ") if initial else ""
            final_symbol = ";\n" if final else ""
            return "%s%s%s" % (initial_symbol, symbol, final_symbol) 

        elif isinstance(symbol_object, ML_Table):
            initial_symbol = (symbol_object.get_c_definition(symbol, final = "") + " ") if initial else ""
            table_content_init = symbol_object.get_c_content_init()
            return "%s = %s;\n" % (initial_symbol, table_content_init)

        elif isinstance(symbol_object, CodeFunction):
            return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            return "%s\n" % symbol_object.get_declaration()

        else:
            print symbol_object.__class__
            raise ML_NotImplemented()

    def generate_initialization(self, symbol, symbol_object, initial = True, final = True):
      if isinstance(symbol_object, Constant) or isinstance(symbol_object, Variable):
        final_symbol = ";\n" if final else ""
        init_code = symbol_object.get_precision().generate_c_initialization(symbol, symbol_object)
        if init_code != None:
          return "%s%s" % (init_code, final_symbol)
        else:
          return ""
      else:
        return ""


    def generate_debug_msg(self, optree, result, code_object, debug_object = None):
        debug_object = optree.get_debug() if debug_object is None else debug_object
        debug_object = debug_object.select_object(optree) if isinstance(debug_object, ML_Debug) else debug_object
        # adding required headers
        if isinstance(debug_object, ML_Debug):
            for header in debug_object.get_require_header():
              code_object.add_header(header)
        precision = optree.get_precision()
        display_format = debug_object.get_display_format(precision.get_c_display_format()) if isinstance(debug_object, ML_Debug) else precision.get_c_display_format()
        display_result = debug_object.get_pre_process(result.get(), optree) if isinstance(debug_object, ML_Debug) else result.get()
        debug_msg = "#ifdef ML_DEBUG\n"
        debug_msg += """printf("%s: %s\\n", %s);\n""" % (optree.get_tag(), display_format, display_result)
        debug_msg += "#endif\n"
        return debug_msg
