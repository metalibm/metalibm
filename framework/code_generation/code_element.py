# -*- coding: utf-8 -*-

###############################################################################
# This file is part of KFG
# Copyright (2013)
# All rights reserved
# created:          Apr  7th, 2014
# last-modified:    Apr  7th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from core.ml_operations import Variable
from code_generation.code_object import NestedCode


class CodeVariable: 
    def __init__(self, name, precision):
        self.name = name
        self.precision = precision

    def get(self):
        return self.name

    def get_stored(self, code_object):
        return self.name

    def get_variable(self, code_object):
        return Variable(self.name, precision = self.precision)


class CodeExpression:
    def __init__(self, expression, vartype):
        self.expression = expression
        self.precision = vartype
        self.store_variable = None

    def get(self):
        return self.expression

    def get_stored(self, code_object):
        storage_var = code_object.get_free_var_name(self.precision)
        self.store_variable = storage_var
        return storage_var

    def get_variable(self, code_object):
        if self.store_variable:
            return Variable(self.store_variable, precision = self.precision)
        else:
            var_name = self.get_stored(code_object)
            return Variable(var_name, precision = self.precision)


class CodeFunction:
    """ function code object """
    def __init__(self, name, arg_list = [], output_format = None, code_object = None):
        """ code function initialization """
        self.name = name
        self.arg_list = arg_list
        self.code_object = code_object
        self.output_format = output_format 

    def get_name(self):
        return self.name

    def add_input_variable(self, name, vartype):
        input_var = Variable(name, precision = vartype) 
        self.arg_list.append(input_var)
        return input_var

    def get_declaration(self, final = True):
        arg_format_list = ", ".join("%s %s" % (inp.get_precision().get_c_name(), inp.get_tag()) for inp in self.arg_list)
        final_symbol = ";" if final else ""
        return "%s %s(%s)%s" % (self.output_format.get_c_name(), self.name, arg_format_list, final_symbol)

    def set_scheme(self, scheme):
        self.scheme = scheme

    def get_definition(self, code_generator, language, folded = True, static_cst = False):
        code_object = NestedCode(code_generator, static_cst = static_cst)
        code_object << self.get_declaration(final = False)
        code_object.open_level()
        code_generator.generate_expr(code_object, self.scheme, folded = folded, initial = False)
        code_object.close_level()
        return code_object
