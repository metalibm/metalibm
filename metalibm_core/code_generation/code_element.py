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
# created:          Apr  7th, 2014
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..core.ml_operations import Variable, FunctionObject
from .code_object import NestedCode


class CodeVariable(object):
    def __init__(self, name, precision):
        self.name = name
        self.precision = precision

    def get(self):
        return self.name

    def get_stored(self, code_object):
        return self.name

    def get_variable(self, code_object):
        return Variable(self.name, precision = self.precision)

    def __str__(self):
        return "CodeVariable(%s[%s])" % (self.name, self.precision)


class CodeExpression(object):
    def __init__(self, expression, vartype):
        self.expression = expression
        self.precision = vartype
        self.store_variable = None

    def get(self):
        return self.expression

    def __str__(self):
        return "CodeExpression(%s[%s])" % (self.expression, self.precision)

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


        
