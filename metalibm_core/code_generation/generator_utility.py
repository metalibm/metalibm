# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Dec 24th, 2013
# last-modified:    Oct  6th, 2015
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import Log
from ..utility.common import ML_NotImplemented, zip_index
from ..core.ml_formats import *
from .code_element import CodeVariable, CodeExpression
from .code_constant import C_Code, Gappa_Code


class DummyTree:
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
    for arg, index in zip_index(gen_list):
        index_map[index] = arg
        ordered_arg_list.append((index, arg))
    ordered_arg_list.sort(key = (lambda (index, arg): arg.get_index()))
    result_list = [None] * len(gen_list)
    for index, arg in ordered_arg_list:
        result_list[index] = gen_function(arg)
    return result_list

class ML_CG_Operator:
    """ parent class for all code generation operators """
    def __init__(self, arity = 0, output_precision = None, pre_process = None, custom_generate_expr = None, force_folding = None, require_header = None, no_parenthesis = False, context_dependant = None, speed_measure = None):
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
        # flag for force folding
        self.force_folding = force_folding
        # list of required header associated to the operator
        self.require_header = require_header if require_header else []
        # flag to enable/disable parenthesis generation
        self.no_parenthesis = no_parenthesis
        # 
        self.context_dependant = context_dependant
        self.speed_measure = speed_measure

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
        raise ML_NotImplemented() 

    def assemble_code(self, code_generator, code_object, optree, var_arg_list, **kwords):
        """ base code assembly function """
        raise ML_NotImplemented()


class CompoundOperator(ML_CG_Operator):
    """ constructor for operator composition """
    def __init__(self, parent, args, **kwords):
        """ compound operator initialization """
        kwords["arity"] = parent.get_arity();
        kwords["output_precision"] = parent.get_output_precision()
        ML_CG_Operator.__init__(self, **kwords)
        self.parent = parent
        self.args = args


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords): #folded = True, result_var = None):
        """ composed expression generator """
        # registering headers
        self.register_headers(code_object)
        # generating list of arguments
        compound_arg = []
        #pre_arg_value = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
        pre_arg_value = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, **kwords), arg_tuple)
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

            else:
                # other compound operator ?
                dummy_precision = arg_function.get_output_precision()
                if dummy_precision == None:
                    dummy_precision = code_generator.get_unknown_precision()
                    if dummy_precision == None:
                        print optree.get_str(depth = 2, display_precision = True) # Error display
                        Log.report(Log.Error, "unknown output precision in compound operator")
            
                dummy_optree = DummyTree("carg", dummy_precision)

                if isinstance(arg_function, CompoundOperator):
                    compound_arg.append(arg_function.generate_expr(code_generator, code_object, dummy_optree, arg_tuple, **kwords))
                elif isinstance(arg_function, ML_CG_Operator):
                    if arg_function.custom_generate_expr:
                        compound_arg.append(arg_function.generate_expr(code_generator, code_object, dummy_optree, arg_tuple, **kwords))
                    else:
                        compound_arg.append(arg_function.assemble_code(code_generator, code_object, dummy_optree, pre_arg_value, **kwords))
        # assembling parent operator code
        folded = kwords["folded"]
        folded_arg = folded if self.get_force_folding() == None else self.get_force_folding()
        kwords["folded"] = folded_arg
        return self.parent.assemble_code(code_generator, code_object, optree, compound_arg, generate_pre_process = generate_pre_process, result_in_args = result_in_args, **kwords)
            

class IdentityOperator(ML_CG_Operator):
    """ symbol operator generator """
    def __init__(self, **kwords):
        """ symbol operator initialization function """
        kwords["arity"] = 1
        ML_CG_Operator.__init__(self, **kwords)

    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)

        if self.custom_generate_expr:
            return self.custom_generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, **kwords)

    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, result_in_args = False, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # generating result code
        result_code = "".join([var_arg.get() for var_arg in var_arg_list])

        # generating assignation if required
        folded = kwords["folded"]
        if (folded and self.get_force_folding() != False) or generate_pre_process != None: 
            prefix = optree.get_tag(default = "tmp")
            result_var = kwords["result_var"]
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            if generate_pre_process != None:
                generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
            code_object << code_generator.generate_assignation(result_varname, result_code) 
            return CodeVariable(result_varname, optree.get_precision())
        else:
            if self.no_parenthesis:
                return CodeExpression("%s" % result_code, optree.get_precision())
            else:
                return CodeExpression("(%s)" % result_code, optree.get_precision())


class SymbolOperator(ML_CG_Operator):
    """ symbol operator generator """
    def __init__(self, symbol, **kwords):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, **kwords)
        self.symbol = " %s " % symbol


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)

        if self.custom_generate_expr:
            return self.custom_generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, **kwords)


    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # generating result code
        result_code = None
        if self.arity == 1:
            result_code = "%s%s" % (self.symbol, var_arg_list[0].get())
        else:
            result_code = self.symbol.join([var_arg.get() for var_arg in var_arg_list])

        # generating assignation if required
        if (kwords["folded"] and self.get_force_folding() != False) or generate_pre_process != None: 
            prefix = optree.get_tag(default = "tmp")
            result_var = kwords["result_var"]
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            if generate_pre_process != None:
                generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
            code_object << code_generator.generate_assignation(result_varname, result_code) 
            return CodeVariable(result_varname, optree.get_precision())
        else:
            if self.no_parenthesis:
                return CodeExpression("%s" % result_code, optree.get_precision())
            else:
                return CodeExpression("(%s)" % result_code, optree.get_precision())


class FO_Arg:
    def __init__(self, index):
        self.index = index

    def get_index(self):
        return self.index

class ML_VarArity: 
    """ variable arity """
    pass

class FO_Result: 
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

class FO_ResultRef: 
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
  

class FunctionOperator(ML_CG_Operator):
    def __init__(self, function_name, arg_map = None, pre_process = None, declare_prototype = None, **kwords):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, **kwords)
        self.function_name = function_name
        self.arg_map = None if self.arity is ML_VarArity else dict([(i, FO_Arg(i)) for i in xrange(self.arity)]) if arg_map == None else (arg_map if arg_map else {})
        self.total_arity = None if self.arg_map == None else len(self.arg_map)
        self.pre_process = pre_process
        self.declare_prototype = declare_prototype


    def register_prototype(self, optree, code_object):
        if self.declare_prototype:
            code_object.declare_function(self.function_name, self.declare_prototype)


    def get_arg_from_index(self, index, arg_result_list, arg_map, result_args_map):
        """ return the function argument at position <index> """
        arg_index = arg_map[index]
        if isinstance(arg_index, FO_Arg):
          return arg_result_list[arg_index.index]

        elif isinstance(arg_index, FO_Result):
          return result_args_map[arg_index.get_index()]
          # return FO_Result(arg_index.get_index(), arg_index.get_output_precision()) 

        elif isinstance(arg_index, FO_ResultRef):
          return CodeExpression("&%s" % result_args_map[arg_index.get_index()].get(), None)

        else:
          return CodeExpression(arg_map[index], None)

    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)
        self.register_prototype(optree, code_object)

        if self.custom_generate_expr:
            return self.custom_generate_expr(self, code_generator, code_object, optree, optree.inputs, generate_pre_process = generate_pre_process, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, **kwords)
            #[self.get_arg_from_index(index, arg_result) for index in xrange(self.arity)],


    def generate_call_code(self, result_arg_list):
        return "%s(%s)" % (self.function_name, ", ".join([var_arg.get() for var_arg in result_arg_list]))

    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        # extracting extra generation parameters
        folded = kwords["folded"]
        result_var = kwords["result_var"]

        arg_map = dict([(i, FO_Arg(i)) for i in xrange(len(optree.inputs))]) if self.arity is ML_VarArity else self.arg_map
        total_arity = len(optree.inputs) if self.arity is ML_VarArity else self.total_arity
        
        # is their result passed by reference
        result_in_args = False
        result_args_map = {}
        for arg_index in self.arg_map:
          arg = self.arg_map[arg_index]
          if isinstance(arg, FO_Result) or isinstance(arg, FO_ResultRef):
            prefix = optree.get_tag(default = "tmp")
            result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
            result_in_args = CodeVariable(result_varname, optree.get_precision())
            result_args_map[arg.get_index()] = result_in_args 


        # generating result code
        result_arg_list = [self.get_arg_from_index(index, var_arg_list, arg_map = arg_map, result_args_map = result_args_map) for index in xrange(total_arity)]
        #result_code = None
        #result_code = "%s(%s)" % (self.function_name, ", ".join([var_arg.get() for var_arg in result_arg_list]))
        result_code = self.generate_call_code(result_arg_list)


        if result_in_args:
          code_object << code_generator.generate_untied_statement(result_code)
          return result_in_args

        else:
          # generating assignation if required
          if (folded and self.get_force_folding() != False) or generate_pre_process != None:
              prefix = optree.get_tag(default = "tmp")
              result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
              if generate_pre_process != None:
                  generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
              code_object << code_generator.generate_assignation(result_varname, result_code) 
              return CodeVariable(result_varname, optree.get_precision())
          else:
              return CodeExpression("%s" % result_code, optree.get_precision())

class FunctionObjectOperator:
    """ meta generator for FunctionObject """
    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        return optree.get_function_object().get_generator_object().generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords)


class TemplateOperator(FunctionOperator):
    """ template operator class """
    def generate_call_code(self, result_arg_list):
        """ overloading of FunctionOperator generate_call_code for template operator object """
        return self.function_name % tuple(var_arg.get() for var_arg in result_arg_list)


class AsmInlineOperator(ML_CG_Operator):
    def __init__(self, asm_template, arity = 2, arg_map = None, output_num = 1):
        """ symbol operator initialization function """
        ML_CG_Operator.__init__(self, arity)
        self.asm_template = asm_template
        self.arg_map = dict([(0, FO_Result(0))] + [(i+1, FO_Arg(i)) for i in xrange(arity)]) if arg_map == None else arg_map
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


    def generate_expr(self, code_generator, code_object, optree, arg_tuple, generate_pre_process = None, **kwords):
        """ generate expression function """
        # registering headers
        self.register_headers(code_object)

        if self.custom_generate_expr:
            return self.custom_generate_expr(code_generator, code_object, optree, arg_tuple, generate_pre_process = generate_pre_process, **kwords)
        else:
            # generating list of arguments
            #arg_result = [code_generator.generate_expr(code_object, arg, **kwords) for arg in arg_tuple]
            arg_result = ordered_generation(lambda arg: code_generator.generate_expr(code_object, arg, **kwords), arg_tuple)
            # assembling parent operator code
            return self.assemble_code(code_generator, code_object, optree, arg_result, generate_pre_process = generate_pre_process, **kwords)
            #[self.get_arg_from_index(index, arg_result) for index in xrange(self.arity)], 


    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, **kwords):
        """ base code assembly function """
        # registering headers
        self.register_headers(code_object)

        prefix = optree.get_tag(default = "tmp")
        result_var = kwords["result_var"]
        result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)

        # generating result code
        template_content = tuple([self.get_arg_from_index(index, var_arg_list, [result_varname]).get() for index in xrange(self.total_arity)])
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
          else:
            raise ValueError("Unknow floating point precision: ", precision)
        elif isinstance(precision, ML_Fixed_Format):
          round_name = "fixed<" + str(-precision.get_frac_size()) + ", " + directions_strings[direction] + ">"
        else:
          raise ValueError("Unknow precision type (is neither Fixed of Floating): ", precision)
        FunctionOperator.__init__(self, round_name, arity = 1, **kwords)


    def assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = None, **kwords):
        # registering headers
        self.register_headers(code_object)

        force_exact = None if not "exact" in kwords else kwords["exact"]
        if code_generator.get_exact_mode() == True or force_exact == True or optree.get_precision() == ML_Exact:
            result_code = "".join([var_arg.get() for var_arg in var_arg_list])

            # generating assignation if required
            folded = kwords["folded"]
            result_var = kwords["result_var"]
            if (folded and self.get_force_folding() != False) or generate_pre_process != None: 
                prefix = optree.get_tag(default = "tmp")
                result_varname = result_var if result_var != None else code_object.get_free_var_name(optree.get_precision(), prefix = prefix)
                if generate_pre_process != None:
                    generate_pre_process(code_generator, code_object, optree, var_arg_list, **kwords)
                code_object << code_generator.generate_assignation(result_varname, result_code) 
                return CodeVariable(result_varname, optree.get_precision())
            else:
                return CodeExpression("%s" % result_code, optree.get_precision())
        else:
           return FunctionOperator.assemble_code(self, code_generator, code_object, optree, var_arg_list, generate_pre_process = generate_pre_process, **kwords) 
        
def type_all_match(*args, **kwords):
  """ match any type parameters """
  return True

def type_std_integer_match(*arg, **kwords):
  """ check that argument are all integers """
  return all(map(is_std_integer_format, arg))

## Type Class Match, used to described match-test function
#  based on the class of the precision rather than the format itself
class TCM:
  """ Type Class Match """
  def __init__(self, format_class):
    self.format_class = format_class

  def __call__(self, arg_format):
    return isinstance(arg_format, self.format_class)


## Format Strict match, used to described match-test function
#  based on the strict comparison of argument formats against
#  expected formats
class FSM:
  """ Format Strict Match """
  def __init__(self, format_obj):
    self.format_obj = format_obj

  def __call__(self, arg_format):
    return self.format_obj == arg_format


class type_strict_match:
    def __init__(self, *type_tuple):
        """ check that argument and constrain type match strictly """
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        return self.type_tuple == arg_tuple

class type_strict_match_list:
    def __init__(self, *type_tuple_list):
        """ check that argument and constrain type match strictly """
        self.type_tuple_list = type_tuple_list

    def __call__(self, *arg_tuple, **kwords):
        for constraint_list, arg_format in zip(self.type_tuple_list, arg_tuple):
          if not arg_format in constraint_list:
            return False
        return True

class type_fixed_match: 
    """ type_strict_match + match any instance of ML_Fixed_Format to 
        ML_Fixed_Format descriptor """
    def __init__(self, *type_tuple):
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        return reduce(lambda acc, v: acc and (v[0] == v[1] or (v[0] == ML_Fixed_Format)) and isinstance(v[1], ML_Fixed_Format), zip(self.type_tuple, arg_tuple))

class type_custom_match: 
    """ type_strict_match + match any instance of ML_Fixed_Format to 
        ML_Fixed_Format descriptor """
    def __init__(self, *type_tuple):
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        acc = True
        for match_func, t in zip(self.type_tuple, arg_tuple):
          acc = acc and match_func(t)
        return acc
        #return reduce((lambda acc, v: acc and (v[0](v[1]))), zip(self.type_tuple, arg_tuple))

class type_relax_match:
    """ implement a relaxed type comparison including ML_Exact as possible true answer """
    def __init__(self, *type_tuple):
        self.type_tuple = type_tuple

    def __call__(self, *arg_tuple, **kwords):
        return reduce(lambda acc, v: acc and (v[0] == v[1] or v[1] == ML_Exact), zip(self.type_tuple, arg_tuple))

class type_result_match:
    def __init__(self, result_type):
        self.result_type = result_type

    def __call__(self, *arg_tuple, **kwords):
        return arg_tuple[0] == self.result_type

def type_function_match(*arg_tuple, **kwords): #optree = None):
    optree = kwords["optree"]
    function_object = optree.get_function_object()
    arg_tuple = tuple(inp.get_precision() for inp in optree.inputs)
    match_function = type_strict_match(*function_object.get_arg_precision_tuple())
    return type_strict_match(*arg_tuple)


def build_simplified_operator_generation_nomap(precision_list, arity, operator, result_precision = None, explicit_rounding = False, match_function = type_strict_match, extend_exact = False, cond = lambda optree: True):
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

def build_simplified_operator_generation(precision_list, arity, operator, result_precision = None, explicit_rounding = False, match_function = type_strict_match, extend_exact = False, cond = lambda optree: True):
  return {cond: build_simplified_operator_generation_nomap(precision_list, arity, operator, result_precision, explicit_rounding, match_function, extend_exact, cond)}
