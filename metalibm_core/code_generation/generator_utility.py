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

from functools import reduce

from ..utility.log_report import Log
from ..core.ml_formats import (
    is_table_index_format, is_std_integer_format,
    ML_Fixed_Format, ML_Exact, ML_Void,
    ML_DoubleDouble,
    ML_FP_Format, ML_Binary32, ML_Binary64, ML_Integer,
    ML_RoundToNearest, ML_RoundTowardPlusInfty, ML_RoundTowardMinusInfty,
    ML_RoundTowardZero)
from .code_element import CodeVariable, CodeExpression
from .code_constant import C_Code, Gappa_Code

from ..utility.source_info import SourceInfo


class DummyTree(object):
    def __init__(self, tag = "tmp", precision = None):
        self.tag = tag
        self.precision = precision

    def get_tag(self, default = "tmp"):
        return self.tag

    def get_precision(self):
        return self.precision

def ordered_generation(gen_function, gen_list):
    # generate gen_list by index order and
    # return results with gen_list equivalent order
    index_map = {}
    ordered_arg_list = []
    for index, arg in enumerate(gen_list):
        index_map[index] = arg
        ordered_arg_list.append((index, arg))
    def local_key(p):
        index, arg = p
        return arg.get_index()
    ordered_arg_list.sort(key=local_key)
    result_list = [None] * len(gen_list)
    for index, arg in ordered_arg_list:
        result_list[index] = gen_function(arg)
    return result_list

def default_process_arg_list(code_object, code_generator, arg_list):
    return arg_list

class ML_CG_Operator(object):
    """ parent class for all code generation operators """
    def __init__(self,
            arity = 0, output_precision = None, pre_process = None,
            custom_generate_expr = None, force_folding = None,
            require_header = None, no_parenthesis = False,
            context_dependant = None, speed_measure = 0,
            force_input_variable = False,
            ## process argument list before assembling code
            process_arg_list = default_process_arg_list
        ):
        # number of inputs expected for the operator
        self.arity = arity
        # is the operator part of the composition
        self.compound = None
        # output precision
        self.output_precision = output_precision
        # pre process function 
        self.pre_process = pre_process
        # custom implementation of the generated_expr function
        self.custom_generate_expr = custom_generate_expr
        # flag to force folding
        self.force_folding = force_folding
        # list of required header associated to the operator
        self.require_header = require_header if require_header else []
        # flag to enable/disable parenthesis generation
        self.no_parenthesis = no_parenthesis
        ## if set, does not accept CodeExpression as input variables
        #  (forces CodeVariable)
        self.force_input_variable = force_input_variable
        # argument processing between argument code generation
        # and operator code assembling
        self.process_arg_list = process_arg_list
        # 
        self.context_dependant = context_dependant
        self.speed_measure = speed_measure

        ## source file information about opertor instantitation
        self.sourceinfo = SourceInfo.retrieve_source_info(1)

    def get_process_arg_list(self):
        return self.process_arg_list

    def get_source_info(self):
        return self.sourceinfo

    def get_force_input_variable(self):
        return self.force_input_variable

    def register_headers(self, code_object):
        for header in self.require_header: 
            code_object.add_header(header)

    def get_speed_measure(self):
        return self.speed_measure
    def set_speed_measure(self, new_speed_measure):
        self.speed_measure(new_speed_measure)

    def get_force_folding(self):
        return self.force_folding

    def get_arity(self):
        return self.arity

    def get_output_precision(self):
        return self.output_precision

    def __call__(self, *args):
        """ Operator implicit composition call wrapper """
        if len(args) != self.arity:
            Log.report(Log.Error, "operator arity is inconsistent")
        else:
            return CompoundOperator(self, args) 

    def generate_expr(self, code_generator, code_object, optree, arg_tuple, **kwords):
        """ base code generation function """
        raise NotImplementedError 

    def assemble_code(self, code_generator, code_object, optree, var_arg_list, **kwords):
        """ base code assembly function """
        raise NotImplementedError


class CompoundOperator(ML_CG_Operator):
    """ constructor for operator composition """
    def __init__(self, parent, args, **kwords):
        """ compound operator initialization """
        kwords["arity"] = parent.get_arity();
        kwords["output_precision"] = parent.get_output_precision()
        ML_CG_Operator.__init__(self, **kwords)
        self.parent = parent
        self.args = args

    ## return the expected operator latency
    def get_speed_measure(self):
        return self.parent.get_speed_measure() + max([arg.get_speed_measure() for arg in self.args])


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, force_variable_storing = False, **kwords): #folded = True, result_var = None):
        """ composed expression generator """
        # registering headers
        self.register_headers(code_object)
        # generating list of arguments
        compound_arg = []
        force_input_variable = self.get_force_input_variable()
        pre_arg_value = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, force_variable_storing = force_input_variable, **kwords), arg_tuple)
        # does a result appears as an argument (passed by reference)
        result_in_args = False
        for arg_function in self.args:
            if isinstance(arg_function, FO_Arg):
                compound_arg.append(pre_arg_value[arg_function.get_index()])

            elif isinstance(arg_function, FO_Result):
                # TODO: improve for multiple output nodes
                output_precision = optree.get_precision()
                compound_arg.append(FO_Result(arg_function.get_index(), output_precision))
                result_in_args = True

            elif isinstance(arg_function, FO_Value):
                output_precision = arg_function.get_output_precision()
                compound_arg.append(arg_function.get_value())

            elif isinstance(arg_function, FO_ResultRef):
                output_precision = optree.get_precision()
                compound_arg.append(
                    FO_ResultRef(
                        arg_function.get_index(),
                        output_precision
                    )
                )
                result_in_args = True

            else:
                # other compound operator ?
                dummy_precision = arg_function.get_output_precision()
                if dummy_precision == None:
                    dummy_precision = code_generator.get_unknown_precision()
                    if dummy_precision == None:
                        print(optree) # Error display
                        Log.report(Log.Error, "unknown output precision in compound operator")
            
                dummy_optree = DummyTree("carg", dummy_precision)

                if isinstance(arg_function, CompoundOperator):
                    compound_arg.append(
                        arg_function.generate_expr(
                            code_generator, code_object, dummy_optree,
                            arg_tuple, force_variable_storing = force_input_variable,
                            **kwords
                        )
                    )
                elif isinstance(arg_function, ML_CG_Operator):
                    if arg_function.custom_generate_expr:
                        compound_arg.append(
                            arg_function.generate_expr(
                                code_generator, code_object, dummy_optree,
                                arg_tuple, force_variable_storing = force_input_variable,
                                **kwords
                            )
                        )
                    else:
                        compound_arg.append(
                            arg_function.assemble_code(
                                code_generator, code_object, dummy_optree,
                                pre_arg_value, force_variable_storing = force_input_variable,
                                **kwords
                            )
                        )
        # assembling parent operator code
        folded = kwords["folded"]
        folded_arg = folded if self.get_force_folding() == None else \
                     self.get_force_folding()
        kwords["folded"] = folded_arg
        return self.parent.assemble_code(
            code_generator, code_object, optree,
            compound_arg,
            generate_pre_process = generate_pre_process,
            result_in_args = result_in_args, 
            force_variable_storing = force_input_variable,
            **kwords
        )


class IdentityOperator(ML_CG_Operator):
    """ symbol operator generator """
    def __init__(self, **kwords):
        """ symbol operator initialization function """
        kwords["arity"] = 1
        ML_CG_Operator.__init__(self, **kwords)

    def generate_expr(self, 
            code_generator, code_object, optree, arg_tuple,
            generate_pre_process = None, force_variable_storing = False,
            **kwords
        ):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)
        force_input_variable = self.get_force_input_variable()

        if self.custom_generate_expr:
            return self.custom_generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, force_variable_storing = force_variable_storing, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, force_variable_storing = force_input_variable, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, force_variable_storing = force_input_variable, **kwords)

    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, result_in_args = False, force_variable_storing = False, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # generating result code
        result_code = "".join([var_arg.get() for var_arg in var_arg_list])

        # generating assignation if required
        folded = kwords["folded"]
        if force_variable_storing or self.get_force_folding() or (folded and self.get_force_folding() != False) or generate_pre_process != None: 
            prefix = optree.get_tag(default = "id_tmp")
            result_var = kwords["result_var"]
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            if generate_pre_process != None:
                generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
            code_object << code_generator.generate_code_assignation(code_object, result_varname, result_code) 
            return CodeVariable(result_varname, optree.get_precision())
        else:
            if self.no_parenthesis:
                return CodeExpression("%s" % result_code, optree.get_precision())
            else:
                return CodeExpression("(%s)" % result_code, optree.get_precision())


class TransparentOperator(IdentityOperator):
    """ Identity operator which never assign a temporary variable """
    def assemble_code(
            self, code_generator, code_object, optree,
            var_arg_list, generate_pre_process=None,
            result_in_args=False, force_variable_storing=False,
            **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # generating result code
        result_code = "".join([var_arg.get() for var_arg in var_arg_list])

        # generating assignation if required
        folded = kwords["folded"]
        if self.no_parenthesis:
            return CodeExpression("%s" % result_code, optree.get_precision())
        else:
            return CodeExpression("(%s)" % result_code, optree.get_precision())


class SymbolOperator(ML_CG_Operator):
    """ symbol operator generator """
    def __init__(self, symbol, lspace = " ", rspace = " ", inverse = False, **kwords):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, **kwords)
        self.symbol = "%s%s%s" % (lspace, symbol, rspace)
        self.inverse = inverse


    def generate_expr(
            self, code_generator, code_object, optree, arg_tuple,
            generate_pre_process = None, force_variable_storing = False,
            **kwords
        ):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)
        force_input_variable = self.get_force_input_variable()

        if self.custom_generate_expr:
            return self.custom_generate_expr(
                self, code_generator, code_object, optree, arg_tuple,
                generate_pre_process = generate_pre_process,
                force_variable_storing = force_variable_storing, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, force_variable_storing = force_input_variable, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, force_variable_storing = force_variable_storing, **kwords)


    def assemble_code(
            self, code_generator, code_object, optree, var_arg_list,
            generate_pre_process = None, force_variable_storing = False,
            **kwords
        ):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # generating result code
        result_code = None
        if self.arity == 1:
            if not self.inverse:
                result_code = "%s%s" % (self.symbol, var_arg_list[0].get())
            else:
                result_code = "%s%s" % (var_arg_list[0].get(),self.symbol)
        else:
            result_code = self.symbol.join([var_arg.get() for var_arg in var_arg_list])

        # generating assignation if required
        if force_variable_storing or self.get_force_folding() or (kwords["folded"] and self.get_force_folding() != False) or generate_pre_process != None: 
            prefix = optree.get_tag(default = "tmp")
            result_var = kwords["result_var"]
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            if generate_pre_process != None:
                generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
            code_object << code_generator.generate_code_assignation(code_object, result_varname, result_code) 
            return CodeVariable(result_varname, optree.get_precision())
        else:
            if self.no_parenthesis:
                return CodeExpression("%s" % result_code, optree.get_precision())
            else:
                return CodeExpression("(%s)" % result_code, optree.get_precision())

class ConstantOperator(ML_CG_Operator):
    """ Code generator for constant node """
    def __init__(self, force_decl=False, **kwords):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, **kwords)
        ## forces constant declaration
        self.force_decl = force_decl

    def generate_expr(
            self, code_generator, code_object, optree, arg_tuple,
            generate_pre_process=None, language=C_Code,
            force_variable_storing = False, **kwords
        ):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)

        # assembling parent operator code
        return self.assemble_code(
            code_generator, code_object, optree,
            generate_pre_process = generate_pre_process,
            language=language, force_variable_storing = force_variable_storing,
            **kwords
        )

    def assemble_code(
            self, code_generator, code_object, optree,
            generate_pre_process = None, language=C_Code,
            force_variable_storing = False,
            **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        precision = optree.get_precision()

        if code_generator.declare_cst or \
            force_variable_storing or \
            self.force_decl or precision.is_cst_decl_required():
            cst_prefix = "cst" if optree.get_tag() is None else optree.get_tag()
            cst_varname = code_object.declare_cst(optree, prefix = cst_prefix)
            return CodeVariable(cst_varname, precision)
        else:
            if precision is ML_Integer:
                return CodeExpression("%d" % optree.get_value(), precision)
            else:
                return CodeExpression(
                    precision.get_cst(
                        optree.get_value(), language = language
                    ), precision
                )


class FO_Arg(object):
    def __init__(self, index):
        self.index = index

    def get_index(self):
        return self.index

class ML_VarArity(object):
    """ variable arity """
    pass


class FO_Value(object):
  """ Immediate value to be transmitted as is during code generation """
  def __init__(self, value, precision):
    self.value = value
    self.precision = precision

  def get_value(self):
    return CodeExpression(self.value, self.precision)

  def get_output_precision(self):
    return self.precision

class FO_Result(object):
  """ PlaceHolder for a function Operator result
      can be used to allocate result passed by reference
      as an operator arguments.
      output_precision should be determined by the generate expression
      function """
  def __init__(self, index = 0, output_precision = None):
    self.index = 0
    self.output_precision = output_precision

  def get_index(self):
    return self.index

  def get_output_precision(self):
    return self.output_precision

class FO_ResultRef(object):
  """ PlaceHolder for a function Operator result
      can be used to allocate result passed by reference
      as an operator arguments.
      output_precision should be determined by the generate expression
      function """
  def __init__(self, index = 0, output_precision = None):
    self.index = 0
    self.output_precision = output_precision

  def get_index(self):
    return self.index

  def get_output_precision(self):
    return self.output_precision


## Code generation operator for function
class FunctionOperator(ML_CG_Operator):
    default_prefix = "tmp"
    def __init__(
            self, function_name, arg_map = None, pre_process = None,
            declare_prototype = None, void_function = False, **kwords
        ):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, **kwords)
        self.function_name = function_name
        self.arg_map = None if self.arity is ML_VarArity else dict(
                [(i, FO_Arg(i)) for i in range(self.arity)]
            ) if arg_map == None else (arg_map if arg_map else {})
        self.total_arity = None if self.arg_map == None else len(self.arg_map)
        self.pre_process = pre_process
        self.declare_prototype = declare_prototype
        self.void_function = void_function


    ## Register the function protype of @self to
    #  the CodeObject @p code_object
    def register_prototype(self, optree, code_object):
        if self.declare_prototype:
            code_object.declare_function(
                self.function_name, self.declare_prototype
            )

    ## resolve 1 level of indirection for function arguments
    #  @param arg value to be materialized
    #  @param arg_map dictionnary index => arg template defined 
    #         at FunctionOperator instantiation
    #  @param arg_result_list list of arguments
    #  @param result_map dictionary of result value
    def materialize_argument(self, arg, arg_result_list, arg_map, result_map):
        if isinstance(arg, FO_Arg):
            try:
                arg_result = arg_result_list[arg.get_index()]
            except IndexError as e:
                Log.report(Log.Error, "error index {} for fct {} in list {}", arg.get_index(), self.function_name, arg_result_list, error=e)
            else:
                return self.materialize_argument(
                    arg_result_list[arg.get_index()],
                    arg_result_list,
                    arg_map,
                    result_map
                )
        elif isinstance(arg, FO_Result):
            return result_map[arg.get_index()]
        elif isinstance(arg, FO_ResultRef):
            return CodeExpression(
                "&%s" % result_map[arg.get_index()].get(), None
            )
        elif isinstance(arg, CodeVariable) or isinstance(arg, CodeExpression):
            return arg
        else:
            return CodeExpression(arg, None)

    ## Extract an argument with index @p index from
    #  the pre-built list of arguments and results
    #  @param index numerical position of the arguments in the list
    #  @param arg_result_list list of arguments submitted when calling
    #         to assemble_code method
    #  @param arg_map dictionnary of index => argument template
    #         declared at operator definition or built
    #  @param result_args_map dictionnary index => materialized result value 
    def get_arg_from_index(
            self, index, arg_result_list, arg_map, result_args_map
        ):
        """ return the function argument at position <index> """
        arg_index = arg_map[index]
        return self.materialize_argument(
            arg_index,
            arg_result_list,
            arg_map,
            result_args_map
        )

    ## generate source code corresponding to the implementation
    #  of the Operation Node @p optree
    def generate_expr(
            self, code_generator, code_object, optree, arg_tuple,
            generate_pre_process = None, force_variable_storing = False,
            **kwords
        ):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)
        self.register_prototype(optree, code_object)

        if self.custom_generate_expr:
            return self.custom_generate_expr(
                self, code_generator, code_object, optree, optree.inputs,
                generate_pre_process = generate_pre_process,
                force_variable_storing = force_variable_storing,
                **kwords
            )
        else:
            # generating list of arguments
            arg_result = ordered_generation(
                lambda arg: code_generator.generate_expr(
                    code_object, arg, force_variable_storing=self.get_force_input_variable(),
                    **kwords
                ), arg_tuple
            )
            # assembling parent operator code
            return self.assemble_code(
                code_generator, code_object, optree, arg_result,
                generate_pre_process = generate_pre_process,
                force_variable_storing = force_variable_storing,
                **kwords
            )


    def generate_call_code(self, result_arg_list):
        return "{function_name}({arg_list})".format(
            function_name = self.function_name,
            arg_list = ", ".join(
                [var_arg.get() for var_arg in result_arg_list]
            )
        )

    ## Function source-code assembling method
    # @param code_generator Code generation engine
    # @param code_object CodeObject output of the generated source code
    # @param optree FunctionCall operation node to be generated
    # @param var_arg_list list of pre-generated arguments as they appears in
    #        the function call parameter list
    # @param generate_pre_process pre-processing to be apply before code
    #        generation
    # @param results_in_args boolean indicating wheter the node results is
    #        part of the parameter list or not
    # @param kwords generic extra keyword parameters
    def assemble_code(
            self, code_generator, code_object,
            optree, var_arg_list,
            generate_pre_process = None, result_in_args = False,
            force_variable_storing = False,
            **kwords
        ):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # extracting extra generation parameters
        folded = kwords["folded"]
        result_var = kwords["result_var"]

        arg_map = dict([
            (i, FO_Arg(i)) for i in range(
                len(optree.inputs))]
        ) if self.arity is ML_VarArity else self.arg_map
        total_arity = len(optree.inputs) if self.arity is ML_VarArity \
                      else self.total_arity
        # check if there are results passed by reference
        # if so then set result_in_args to point towards the result variables
        # and materialize result variable code in result_arg_maps
        result_args_map = {}
        merged_arg_list = [self.arg_map[arg_index] for arg_index in self.arg_map] + var_arg_list
        for arg in merged_arg_list:
          if isinstance(arg, FO_Result) or isinstance(arg, FO_ResultRef):
            arg_index = arg.get_index()
            if not arg_index in result_args_map:
                prefix = optree.get_tag(default=self.default_prefix)
                result_varname = result_var if result_var != None \
                     else code_object.get_free_var_name(
                        optree.get_precision(), prefix = prefix
                    )
                result_in_args = CodeVariable(
                    result_varname, optree.get_precision()
                )
                result_args_map[arg_index] = result_in_args


        # generating result code
        result_arg_list = [
            self.get_arg_from_index(
                index,
                var_arg_list,
                arg_map = arg_map,
                result_args_map = result_args_map
            ) for index in range(total_arity)
        ]
        result_arg_list = self.get_process_arg_list()(
            code_object,
            code_generator,
            result_arg_list
        )
        result_code = self.generate_call_code(result_arg_list)

        if result_in_args:
          code_object << code_generator.generate_untied_statement(result_code)
          return result_in_args

        elif self.void_function:
          code_object << code_generator.generate_untied_statement(result_code)
          return None

        else:
          # generating assignation if required
          if force_variable_storing or self.get_force_folding() or (folded and self.get_force_folding() != False) or generate_pre_process != None:
              prefix = optree.get_tag(default=self.default_prefix)
              result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
              if generate_pre_process != None:
                  generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
              code_object << code_generator.generate_code_assignation(code_object, result_varname, result_code) 
              return CodeVariable(result_varname, optree.get_precision())
          else:
              return CodeExpression("%s" % result_code, optree.get_precision())

class FunctionObjectOperator(ML_CG_Operator):
    """ meta generator for FunctionObject """
    def __init__(self):
        ML_CG_Operator.__init__(self)

    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        fct_object = optree.get_function_object()
        fct_object_generator = fct_object.get_generator_object()
        if fct_object_generator is None:
            # default generator
            fct_object_generator = FunctionOperator(
                fct_object.name,
                arity=fct_object.arity
            )
        return fct_object_generator.generate_expr(
            code_generator, code_object, optree, arg_tuple,
            generate_pre_process=None, **kwords)


class TemplateOperator(FunctionOperator):
    """ template operator class """
    def generate_call_code(self, result_arg_list):
        """ overloading of FunctionOperator generate_call_code for template operator object """
        return self.function_name % tuple(var_arg.get() for var_arg in result_arg_list)

## Template operator using the format string construction
class TemplateOperatorFormat(FunctionOperator):
    """ template operator class """
    def generate_call_code(self, result_arg_list):
        """ overloading of FunctionOperator generate_call_code for template operator object """
        try:
            return self.function_name.format(*tuple(var_arg.get() for var_arg in result_arg_list))
        except IndexError as e:
            Log.report(Log.Error, "failed to generate call code with template {} and result_arg_list: {} (len={})", self.function_name, result_arg_list, len(result_arg_list))


class AsmInlineOperator(ML_CG_Operator):
    def __init__(self, asm_template, arity=2, arg_map=None, output_num=1, **kw):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, arity, **kw)
        self.asm_template = asm_template
        self.arg_map = dict([(0, FO_Result(0))] + [(i+1, FO_Arg(i)) for i in range(arity)]) if arg_map == None else arg_map
        # total arity
        self.total_arity = arity + output_num


    def get_arg_from_index(self, index, arg_result_list, result_var_list):
        """ return the function argument at position <index> """
        arg_index = self.arg_map[index]
        if isinstance(arg_index, FO_Arg):
            return arg_result_list[arg_index.index]
        elif isinstance(arg_index, FO_Result):
            return CodeVariable(result_var_list[arg_index.index], None)
        else:
            return CodeExpression(self.arg_map[index], None)


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, force_variable_storing = False, **kwords):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)

        force_input_variable = self.get_force_input_variable()

        if self.custom_generate_expr:
            return self.custom_generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, force_variable_storing = force_input_variable, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, **kwords)
            #[self.get_arg_from_index(index, arg_result) for index in range(self.arity)], 



    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        prefix = optree.get_tag(default = "tmp")
        result_var = kwords["result_var"]
        result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)

        # generating result code
        template_content = tuple([self.get_arg_from_index(index, var_arg_list, [result_varname]).get() for index in range(self.total_arity)])
        result_code = None
        result_code = self.asm_template % template_content 

        if generate_pre_process != None:
            generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)

        # generating assignation if required
        code_object << result_code + ";\n"
        return CodeVariable(result_varname, optree.get_precision())


class RoundOperator(FunctionOperator):
    """ Rounding operator """

    def __init__(self, precision, direction = ML_RoundToNearest, **kwords): 
        directions_strings = {
          ML_RoundToNearest: "ne",
          ML_RoundTowardZero: "zr",
          ML_RoundTowardPlusInfty: "up",
          ML_RoundTowardMinusInfty: "dn",
        }
        if isinstance(precision, ML_FP_Format):
          if precision == ML_Binary64:
            round_name = "float<ieee_64, " + directions_strings[direction] + ">"
          elif precision == ML_Binary32:
            round_name = "float<ieee_32, " + directions_strings[direction] + ">"
          elif precision == ML_DoubleDouble:
            Log.report(Log.Warning, "rounding operator to ml_dd used (approximated)")
            round_name = "float<102, -1022, {}>".format(directions_strings[direction])
          else:
            raise ValueError("Unknow floating point precision: ", precision)
        elif isinstance(precision, ML_Fixed_Format):
          round_name = "fixed<" + str(-precision.get_frac_size()) + ", " + directions_strings[direction] + ">"
        else:
          raise ValueError("Unknow precision type (is neither Fixed of Floating): ", precision)
        FunctionOperator.__init__(self, round_name, arity = 1, **kwords)


    def assemble_code(
            self, code_generator, code_object, optree, var_arg_list,
            generate_pre_process = None,
            force_variable_storing = False, **kwords
        ):
        # registering headers
        self.register_headers(code_object)

        force_exact = None if not "exact" in kwords else kwords["exact"]
        if code_generator.get_exact_mode() == True or force_exact == True or optree.get_precision() == ML_Exact:
            result_code = "".join([var_arg.get() for var_arg in var_arg_list])

            # generating assignation if required
            folded = kwords["folded"]
            result_var = kwords["result_var"]
            if force_variable_storing or self.get_force_folding() or (folded and self.get_force_folding() != False) or generate_pre_process != None: 
                prefix = optree.get_tag(default = "tmp")
                result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
                if generate_pre_process != None:
                    generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
                code_object << code_generator.generate_code_assignation(code_object, result_varname, result_code) 
                return CodeVariable(result_varname, optree.get_precision())
            else:
                return CodeExpression("%s" % result_code, optree.get_precision())
        else:
           return FunctionOperator.assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = generate_pre_process, force_variable_storing = force_variable_storing, **kwords) 
        
def type_all_match(*args, **kwords):
  """ match any type parameters """
  return True

def type_std_integer_match(*arg, **kwords):
  """ check that argument are all integers """
  return all(map(is_std_integer_format, arg))

def type_table_index_match(*arg, **kwords):
  """ check that argument are all integers """
  return all(map(is_table_index_format, arg))

## Type Class Match, used to described match-test function
#  based on the class of the precision rather than the format itself
class TCM(object):
  """ Type Class Match """
  def __init__(self, format_class):
    self.format_class = format_class

  def __call__(self, arg_format):
    return isinstance(arg_format, self.format_class)

class TCLM(object):
  """ Type Class List Match """
  def __init__(self, format_class_list):
    self.format_class_list = format_class_list

  def __call__(self, arg_format):
    return any(isinstance(arg_format, format_class) for format_class in self.format_class_list)

## Format Strict match, used to described match-test function
#  based on the strict comparison of argument formats against
#  expected formats
class FSM(object):
  """ Format Strict Match """
  def __init__(self, format_obj):
    self.format_obj = format_obj

  def __call__(self, arg_format):
    return self.format_obj == arg_format


class BackendImplMatchPredicated:
    """ weak, is set to True if this match succeed it  will be used only
        if it is the only one. If set to False, this match will be used if it
        is the first one encountered (after none or any number of weak matches)
    """
    def __init__(self, weak=False):
        self.weak = weak

    def evaluate_match_value(self, target, match_info):
        """ return a numerical value to evaluate match quality,
            lower is better """
        return None

class MatchResult:
    """ result of a match operation """
    def __init__(self, weak=False):
        self.weak = weak

class ImplemList(list):
    """ list of possible implementation, overloading list class
        to ease list of implementation detection """

def is_impl_list(obj):
    """ predicate testing if obj is an object of class ImplemList """
    return isinstance(obj, ImplemList)

class type_strict_match(object):
    """ Build a type matching predicate from a list of type,
        a node is matched by the predicate if it has as many operands
        as arguments were given to the initializer (minus one for the result)
        AND if the node precision matches the first format argument, and if each
        node's argument matches the other format arguments respectively """
    def __init__(self, *type_tuple):
        """ check that argument and constrain type match strictly """
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        # TODO/FIXME: strict match between type object (no inheritance allowed)
        return self.type_tuple == arg_tuple

class type_strict_match_list(object):
    """ Build a type matching predicate from list of formats,
        result and operands must match one of the item of the list formats
        corresponding to their position """
    def __init__(self, *type_tuple_list, weak=False):
        """ check that argument and constrain type match strictly """
        self.type_tuple_list = type_tuple_list

    def __call__(self, *arg_tuple, **kwords):
        for constraint_list, arg_format in zip(self.type_tuple_list, arg_tuple):
          if not arg_format in constraint_list:
            return False
        return True

def type_strict_match_or_list(type_tuple_list):
    """ Return a function which match strictly any of the tuple within
        type_tuple_list """
    def match_function(*arg_tuple, **kw):
        for constraint_tuple in type_tuple_list:
            if constraint_tuple == arg_tuple:
                return True
        return False
    return match_function

class type_fixed_match(object):
    """ type_strict_match + match any instance of ML_Fixed_Format to 
        ML_Fixed_Format descriptor """
    def __init__(self, *type_tuple):
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        return reduce(lambda acc, v: acc and (v[0] == v[1] or (v[0] == ML_Fixed_Format)) and isinstance(v[1], ML_Fixed_Format), zip(self.type_tuple, arg_tuple))

class type_custom_match(BackendImplMatchPredicated):
    """ Callable class that checks whether all arguments match with their
        respective custom matching function. """
    def __init__(self, *type_tuple, weak=False):
        BackendImplMatchPredicated.__init__(self, weak=weak)
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        acc = True
        if len(self.type_tuple) != len(arg_tuple):
            return False
        for match_func, t in zip(self.type_tuple, arg_tuple):
            acc = acc and match_func(t)
        if acc:
            return MatchResult(weak=self.weak)
        return acc
        #return reduce((lambda acc, v: acc and (v[0](v[1]))), zip(self.type_tuple, arg_tuple))

class type_relax_match(object):
    """ implement a relaxed type comparison including ML_Exact as possible true answer """
    def __init__(self, *type_tuple):
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        acc = True
        for v in zip(self.type_tuple, arg_tuple):
            acc = acc and (v[0] == v[1] or v[1] == ML_Exact)
        return acc
        # return reduce(lambda acc, v: acc and (v[0] == v[1] or v[1] == ML_Exact), zip(self.type_tuple, arg_tuple))

class type_result_match(object):
    def __init__(self, result_type):
        self.result_type = result_type

    def __call__(self, *arg_tuple, **kwords):
        return arg_tuple[0] == self.result_type

def type_function_match(*arg_tuple, optree=None, **kw):
    """ Matching predicate for function call operation.
        This function performs on-the-fly validation of FunctionCall
        operation node, by checking the function call inputs and output
        formats against the FunctionObject it references """
    #optree = kwords["optree"]
    function_object = optree.get_function_object()
    arg_tuple = tuple(inp.get_precision() for inp in optree.inputs)
    expected_arg_tuple = function_object.get_arg_precision_tuple()
    match_function = type_strict_match(*expected_arg_tuple)
    match_result = match_function(*arg_tuple)
    if not match_result:
        Log.report(Log.Info, "could not match FunctionCall {} with arg_tuple {} vs expected {}", optree, arg_tuple, expected_arg_tuple)
    return match_result


def build_simplified_operator_generation_nomap(
        precision_list, arity, operator, result_precision=None,
        explicit_rounding=False, match_function=type_strict_match,
        extend_exact=False, cond=lambda optree: True):
    """ generate a code generation table for the interfaces describes in precision_list """
    result_map = {}
    for precision_hint in precision_list:
        precision = precision_hint if isinstance(precision_hint, tuple) else (result_precision if result_precision != None else precision_hint,) + (precision_hint,) * (arity)
        if explicit_rounding == True:
            rounding_precision = precision[0]
            result_map[match_function(*precision)] = RoundOperator(rounding_precision)(operator)
        else:
            result_map[match_function(*precision)] = operator

        # extending with exact version of the expression
        if extend_exact:
            result_map[type_result_match(ML_Exact)] = operator
    return result_map

def build_simplified_operator_generation(
        precision_list, arity, operator, result_precision=None,
        explicit_rounding=False, match_function=type_strict_match,
        extend_exact=False, cond=lambda optree: True):
    """ precision_list list of precision to be supported
        arity number of operands of the operator
        result_precision format of the operator result
        explicit_rounding force explicit rounding
        match_function function to be used for format comparison
        cond is a lambda function optree -> boolean used as predicate to select the implementation
    
    """
    return {
        cond: build_simplified_operator_generation_nomap(
            precision_list, arity, operator, result_precision,
            explicit_rounding, match_function, extend_exact, cond)
    }
