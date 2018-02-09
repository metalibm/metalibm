# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# VHDL code generation module
# Copyright (2016-)
# All rights reserved
# created:          Nov 19th, 2016
# last-modified:    May  9th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################


from ..utility.log_report import Log

import sollya

from ..core.ml_operations import Variable, Constant, ConditionBlock, Return, TableLoad, Statement, Loop, SpecificOperation, ExceptionOperation, ClearException, NoResultOperation, SwitchBlock, FunctionObject, ReferenceAssign
from ..core.ml_hdl_operations import *
from ..core.ml_table import ML_Table
from ..core.ml_formats import *
from ..core.attributes import ML_Debug, ML_AdvancedDebug
from .code_constant import VHDL_Code
from .code_element import CodeVariable, CodeExpression
from .code_function import CodeFunction
from .code_object import MultiSymbolTable


class VHDLCodeGenerator(object):
    language = C_Code

    """ C language code generator """
    def __init__(self, processor, declare_cst = False, disable_debug = False, libm_compliant = False, default_rounding_mode = ML_GlobalRoundMode, default_silent = None, language = C_Code):
        self.memoization_map = [{}]
        self.processor = processor
        self.declare_cst = declare_cst
        self.disable_debug = disable_debug
        self.libm_compliant = libm_compliant
        self.fp_context = FP_Context(rounding_mode = default_rounding_mode, silent = default_silent)
        self.language = language
        Log.report(Log.Info, "VHDLCodeGenerator initialized with language: %s" % self.language)

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


    def generate_expr(
            self, code_object, optree, folded = False,
            result_var = None, initial = False,
            language = None,
            ## force to store result in a variable, wrapping CodeExpression
            #  in CodeVariable
            force_variable_storing = False
        ):
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
              final_var =  code_object.get_free_var_name(optree.get_precision(), prefix = optree.get_tag(), declare = True, var_ctor = Variable)
              result = CodeVariable(final_var, optree.get_precision())
            else:
              result = CodeVariable(optree.get_tag(), optree.get_precision())

        elif isinstance(optree, Signal):
            if optree.get_var_type() is Variable.Local:
              final_var =  code_object.declare_signal(optree, optree.get_precision(), prefix = optree.get_tag())
              result = CodeVariable(final_var, optree.get_precision())
            else:
              result = CodeVariable(optree.get_tag(), optree.get_precision())

        elif isinstance(optree, Constant):
            precision = optree.get_precision() # .get_base_format()
            if force_variable_storing or self.declare_cst or optree.get_precision().is_cst_decl_required():
                cst_prefix = "cst" if optree.get_tag() is None else optree.get_tag()
                cst_varname = code_object.declare_cst(optree, prefix = cst_prefix)
                result = CodeVariable(cst_varname, precision)
            else:
                if precision is ML_Integer:
                  result = CodeExpression("%d" % optree.get_value(), precision)
                else:
                  try:
                      result = CodeExpression(precision.get_cst(optree.get_value(), language = language), precision)
                  except:
                    result = CodeExpression(precision.get_cst(optree.get_value(), language = language), precision)
                    Log.report(Log.Error, "Error during get_cst call for Constant: %s " % optree.get_str(display_precision = True)) # Exception print


        elif isinstance(optree, Assert):
            cond = optree.get_input(0)
            error_msg = optree.get_error_msg()
            severity = optree.get_severity()

            cond_code = self.generate_expr(code_object, cond, folded = False, language = language)

            code_object << " assert {cond} report {error_msg} severity {severity};\n".format(cond = cond_code.get(), error_msg = error_msg, severity = severity.descriptor)

            return None

        elif isinstance(optree, Wait):
            time_ns = optree.get_time_ns()
            code_object << "wait for {time_ns} ns;\n".format(time_ns = time_ns)
            return None

        elif isinstance(optree, SwitchBlock):
            switch_value = optree.inputs[0]
            
            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), folded = folded, language = language)

            switch_value_code = self.generate_expr(code_object, switch_value, folded = folded, language = language)
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
              self.generate_expr(code_object, case_statement, folded = folded, language = language) 
              code_object.close_level()
            code_object << "}\n"

            return None

        elif isinstance(optree, ReferenceAssign):
            output_var = optree.inputs[0]
            result_value = optree.inputs[1]

            output_var_code   = self.generate_expr(code_object, output_var, folded = False, language = language)

            assign_sign = "<=" if isinstance(output_var, Signal) else ":="

            if isinstance(result_value, Constant):
              # generate assignation
              result_value_code = self.generate_expr(code_object, result_value, folded = folded, language = language)
              code_object << self.generate_assignation(
                output_var_code.get(), result_value_code.get(),
                assign_sign = assign_sign
              )
            else:
              #result_value_code = self.generate_expr(code_object, result_value, folded = True, force_variable_storing = True, language = language)
              result_value_code = self.generate_expr(code_object, result_value, folded = True, language = language)
              code_object << self.generate_assignation(
                output_var_code.get(), result_value_code.get(),
                assign_sign = assign_sign
              )
            if optree.get_debug() and not self.disable_debug:
              self.generate_debug_msg(result_value, result_value_code, code_object, debug_object = optree.get_debug())

            #code_object << self.generate_assignation(output_var_code.get(), result_value_code.get())
            #code_object << output_var.get_precision().generate_c_assignation(output_var_code, result_value_code)
            
            return None

        elif isinstance(optree, RangeLoop):
            iterator  = optree.get_input(0)
            loop_body = optree.get_input(1)
            loop_range = optree.get_loop_range()
            specifier  = optree.get_specifier()

            range_pattern = "{lower} to {upper}" if specifier is RangeLoop.Increasing else "{upper} dowto {lower}"
            range_code = range_pattern.format(lower = sollya.inf(loop_range), upper = sollya.sup(loop_range))

            iterator_code = self.generate_expr(code_object, iterator, folded = folded, language = language)

            code_object << "\n for {iterator} in {loop_range} loop\n".format(iterator = iterator_code.get(), loop_range = range_code)
            code_object.inc_level()
            body_code = self.generate_expr(code_object, loop_body, folded = folded, language = language)
            assert body_code is None
            code_object.dec_level()
            code_object<< "end loop;\n"

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


        elif isinstance(optree, Process):
            # generating pre_statement for process
            pre_statement = optree.get_pre_statement()
            self.generate_expr(
                code_object, optree.get_pre_statement(),
                folded=folded, language=language
            )

            sensibility_list = [
                self.generate_expr(
                    code_object, op, folded=True, language=language
                ).get() for op in optree.get_sensibility_list()
            ]
            sensibility_list = "({})".format(", ".join(sensibility_list)) if len(sensibility_list) != 0 else ""
            code_object << "process{}\n".format(sensibility_list)
            self.open_memoization_level()
            code_object.open_level(
                extra_shared_tables = [MultiSymbolTable.SignalSymbol],
                var_ctor = Variable
            )
            for process_stat in optree.inputs:
              self.generate_expr(code_object, process_stat, folded = folded, initial = False, language = language)

            code_object.close_level()
            self.close_memoization_level()
            code_object << "end process;\n\n"
            return None

        elif isinstance(optree, PlaceHolder):
            first_input = optree.get_input(0)
            first_input_code = self.generate_expr(code_object, first_input, folded = folded, language = language)
            for op in optree.get_inputs()[1:]:
              _ = self.generate_expr(code_object, op, folded = folded, language = language)

            result = first_input_code

        elif isinstance(optree, ComponentInstance):
            component_object = optree.get_component_object()
            component_name = component_object.get_name()
            code_object.declare_component(component_name, component_object)
            io_map           = optree.get_io_map()
            component_tag = optree.get_tag()
            if component_tag is None:
              component_tag = "{component_name}_i{instance_id}".format(component_name = component_name, instance_id = optree.get_instance_id())
            mapped_io = {}
            for io_tag in io_map:
              mapped_io[io_tag] = self.generate_expr(code_object, io_map[io_tag], folded = True, language = language)

            code_object << "\n{component_tag} : {component_name}\n".format(component_name = component_name, component_tag = component_tag)
            code_object << "  port map (\n"
            code_object << "  " + ", \n  ".join("{} => {}".format(io_tag, mapped_io[io_tag].get()) for io_tag in mapped_io) 
            code_object << "\n);\n"

            return None

        elif isinstance(optree, ConditionBlock):
            condition = optree.inputs[0]
            if_branch = optree.inputs[1]
            else_branch = optree.inputs[2] if len(optree.inputs) > 2 else None

            # generating pre_statement
            self.generate_expr(code_object, optree.get_pre_statement(), folded = folded, language = language)

            cond_code = self.generate_expr(code_object, condition, folded = False, language = language)
            try:
              cond_likely = condition.get_likely()
            except AttributeError:
              Log.report(Log.Error, " the following condition has no (usable) likely attribute: %s" % (condition.get_str(depth = 1, display_precision = True, memoization_map = {}))) 
            code_object << "if %s then\n " % cond_code.get()
            #self.open_memoization_level()
            #code_object.open_level()
            code_object.inc_level()
            if_branch_code = self.generate_expr(code_object, if_branch, folded = False, language = language)
            code_object.dec_level()
            #code_object.close_level(cr = "")
            #self.close_memoization_level()
            if else_branch:
                code_object << " else "
                #code_object.open_level()
                #self.open_memoization_level()
                else_branch_code = self.generate_expr(code_object, else_branch, folded = True, language = language)
                #code_object.close_level()
                #self.close_memoization_level()
            else:
               #  code_object << "\n"
               pass
            code_object << "end if;\n"

            return None

        elif isinstance(optree, Select):
             # we go through all of select operands to
             # flatten the select tree
             def flatten_select(op, cond = None):
               """ Process recursively a Select operation to build a list
                   of tuple (result, condition) """
               if not isinstance(op, Select): return [(op, cond)]
               lcond = op.inputs[0] if cond is None else LogicalAnd(op.inputs[0], cond, precision = cond.get_precision())
               return flatten_select(op.inputs[1], lcond) + flatten_select(op.inputs[2], cond)

             def legalize_select_input(select_input):
                if select_input.get_precision().get_bit_size() != optree.get_precision().get_bit_size():
                    return Conversion(
                        select_input,
                        precision = optree.get_precision()
                    )
                else:
                    return select_input

             prefix = optree.get_tag(default = "setmp")
             result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
             result = CodeVariable(result_varname, optree.get_precision())
             select_opcond_list = flatten_select(optree);
             if not select_opcond_list[-1][1] is None:
                Log.report(Log.Error, "last condition in flatten select differs from None")

             gen_list = []
             for op, cond in select_opcond_list: 
               op = legalize_select_input(op)
               op_code = self.generate_expr(code_object, op, folded = folded, language = language)
               if not cond is None:
                 cond_code = self.generate_expr(code_object, cond, folded = True, force_variable_storing = True, language = language)
                 gen_list.append((op_code, cond_code))
               else:
                 gen_list.append((op_code, None))

             code_object << "{result} <= \n".format(result = result.get())
             code_object.inc_level()
             for op_code, cond_code in gen_list:
               if not cond_code is None:
                 code_object << "{op_code} when {cond_code} else\n".format(op_code = op_code.get(), cond_code = cond_code.get())
               else:
                 code_object << "{op_code};\n".format(op_code = op_code.get())
             code_object.dec_level()

        elif isinstance(optree, TableLoad):
            table = optree.get_input(0)
            index = optree.get_input(1)
            index_code = self.generate_expr(code_object, index, folded = folded, language = language)
            prefix = optree.get_tag(default = "table_value")
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            result = CodeVariable(result_varname, optree.get_precision())
            code_object << "with {index} select {result} <=\n".format(index = index_code.get(), result = result.get())

            table_dimensions = table.get_precision().get_dimensions()
            assert len(table_dimensions) == 1
            table_size = table_dimensions[0]

            default_value = 0

            # linearizing table selection
            for tabid, value in enumerate(table.get_data()):
              code_object << "\t{} when {},\n".format(table.get_precision().get_storage_precision().get_cst(value),index.get_precision().get_cst(tabid))
            # last_index = table_size - 1
            # last cell
            # code_object << "\t{} when {}".format(table.get_precision().get_storage_precision().get_cst(table.get_data()[last_index]),index.get_precision().get_cst(last_index))

            #if 2**int(sollya.log2(table_size)) == table_size:
            #  code_object << ";\n"
            #else:
            #  code_object << ",\n\t{} when others;\n".format(table.get_precision().get_storage_precision().get_cst(default_value))
            code_object << "\t{} when others;\n".format(table.get_precision().get_storage_precision().get_cst(default_value))

             # result is set 


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

        else:
            generate_pre_process = self.generate_clear_exception if optree.get_clearprevious() else None
            result = self.processor.generate_expr(self, code_object, optree, optree.inputs, generate_pre_process = generate_pre_process, folded = folded, result_var = result_var, language = language)

        # registering result into memoization table
        self.add_memoization(optree, result)

        # debug management
        if optree.get_debug() and not self.disable_debug:
            self.generate_debug_msg(optree, result, code_object)

        def result_too_long(result, threshold = 80):
            """ Checks if CodeExpression result exceeds a given threshold """
            if not isinstance(result, CodeExpression):
                return False
            else:
                return len(result.get()) > threshold
            

        if (initial or force_variable_storing or result_too_long(result)) and not isinstance(result, CodeVariable) and not result is None:
            # result could have been modified from initial optree
            result_precision = result.precision
            prefix_tag = optree.get_tag(default = "var_result") if force_variable_storing  else "tmp_result" 
            final_var = result_var if result_var else code_object.get_free_var_name(result_precision, prefix = prefix_tag, declare = True)
            code_object << self.generate_code_assignation(code_object, final_var, result.get())
            return CodeVariable(final_var, result_precision)

        return result

    def generate_clear_exception(self, code_generator, code_object, optree, var_arg_list, language = None, **kwords): 
        #generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
        self.generate_expr(code_object, ClearException(), language = language)

    def generate_code_assignation(self, code_object, result_var, expression_code, final=True, assign_sign=None):
        assign_sign_map = {
            Signal: "<=",
            Variable: ":=",
            None: "<="
        }
        assign_sign = assign_sign or assign_sign_map[code_object.default_var_ctor]
        return self.generate_assignation(result_var, expression_code, final=final, assign_sign = assign_sign)
        

    def generate_assignation(self, result_var, expression_code, final = True, assign_sign = "<="):
        """ generate code for assignation of value <expression_code> to 
            variable <result_var> """
        final_symbol = ";\n" if final else ""
        return "{result} {assign_sign} {expr}{final_symbol}".format(
            result = result_var,
            assign_sign = assign_sign,
            expr = expression_code,
            final_symbol = final_symbol
        )

    def generate_untied_statement(self, expression_code, final = True):
      final_symbol = ";\n" if final else ""
      return "%s%s" % (expression_code, final_symbol) 


    def generate_declaration(self, symbol, symbol_object, initial = True, final = True):
        if isinstance(symbol_object, Constant):
            precision_symbol = (symbol_object.get_precision().get_code_name(language = self.language) + " ") 
            final_symbol = ";\n" 
            return "constant %s : %s := %s%s" % (symbol, precision_symbol, symbol_object.get_precision().get_cst(symbol_object.get_value(), language = self.language), final_symbol) 

        elif isinstance(symbol_object, Variable):
            precision_symbol = (symbol_object.get_precision().get_code_name(language = self.language) + " ")
            return "variable %s : %s;\n" % (symbol, precision_symbol) 

        elif isinstance(symbol_object, Signal):
            precision_symbol = (symbol_object.get_precision().get_code_name(language = self.language) + " ") if initial else ""
            return "signal %s : %s;\n" % (symbol, precision_symbol) 

        elif isinstance(symbol_object, ML_Table):
            initial_symbol = (symbol_object.get_definition(symbol, final = "", language = self.language) + " ") if initial else ""
            table_content_init = symbol_object.get_content_init(language = self.language)
            return "%s = %s;\n" % (initial_symbol, table_content_init)

        elif isinstance(symbol_object, CodeFunction):
            return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, ComponentObject):
            return "%s\n" % symbol_object.get_declaration()

        elif isinstance(symbol_object, FunctionObject):
            return "%s\n" % symbol_object.get_declaration()

        else:
            print(symbol_object.__class__)
            raise NotImplementedError

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

    ## Generating debug message for a Constant optree node
    def generate_debug_msg_for_cst(self, optree, result, code_object, debug_object = None):
        assert isinstance(optree, Constant)
        cst_tag   = optree.get_tag()
        cst_value = optree.get_value() 
        debug_msg = "echo \"constant {cst_name} has value {cst_value}\";".format(cst_name = cst_tag, cst_value = cst_value)
        self.get_debug_code_object() << debug_msg

    def extract_debug_object_format(self, optree, code_object, debug_object):
        debug_object = optree.get_debug() if debug_object is None else debug_object
        debug_object = debug_object.select_object(optree) if isinstance(debug_object, ML_Debug) else debug_object
        # adding required headers
        if isinstance(debug_object, ML_Debug):
            for header in debug_object.get_require_header():
              code_object.add_header(header)
        precision = optree.get_precision().get_support_format()
        display_format = debug_object.get_display_format(precision.get_display_format(language = self.language)) if isinstance(debug_object, ML_Debug) else precision.get_display_format(language = self.language)
        return debug_object, display_format


    def generate_debug_msg(self, optree, result, code_object, debug_object = None):
      if isinstance(optree, Constant):
        return self.generate_debug_msg_for_cst(optree, result, code_object, debug_object)
      else:
        debug_object, display_format = self.extract_debug_object_format(optree, code_object, debug_object)
        if not isinstance(result, CodeVariable):
          final_var = code_object.get_free_signal_name(optree.get_precision(), prefix = "dbg_"+ optree.get_tag())
          #code_object << "{} <= {};\n".format(final_var, result.get())
          code_object << self.generate_code_assignation(code_object, final_var, result.get())
          result = CodeVariable(final_var, optree.get_precision())
        signal_name = "testbench.tested_entity.{}".format(result.get())
        # display_result = debug_object.get_pre_process(result.get(), optree) if isinstance(debug_object, ML_Debug) else result.get()
        display_result = debug_object.get_pre_process(signal_name, optree) if isinstance(debug_object, ML_AdvancedDebug) else "examine {display_format} {signal_name}".format(display_format = display_format, signal_name = signal_name)
        #debug_msg = "echo \"{tag}\"; examine {display_format} testbench.tested_entity.{display_result};\n".format(tag = optree.get_tag(), display_format = display_format, display_result = display_result)
        debug_msg = "echo \"{tag}\"; {display_result};\n".format(tag = optree.get_tag(), display_result = display_result)
        self.get_debug_code_object() << debug_msg

    ## define the code object for debug 
    def set_debug_code_object(self, debug_code_object):
        self.debug_code_object = debug_code_object
    ## retrieve the code object for debug
    def get_debug_code_object(self):
        return self.debug_code_object
