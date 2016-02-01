# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2016)
# All rights reserved
# created:          Apr  7th, 2014
# last-modified:    Feb  1st, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..core.ml_operations import Variable, FunctionObject
from .code_object import NestedCode


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

    def strip_outer_parenthesis(self):
        if self.expression[0] == "(" and self.expression[-1] == ")":
          self.expression = self.expression[1:-1]

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


        
