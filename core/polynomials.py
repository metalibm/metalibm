# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 24th, 2013
# last-modified:    Jan 3rd,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import *

from ..utility.common import Callable
from ..utility.log_report import Log
from .ml_operations import Constant, Variable, Multiplication, Addition

class Polynomial:
    """ Mathematical polynomial object class """

    def __init__(self, init_object = None):
        """ Polynomial initialization function
            init_object can be one of 
                - list of coefficient
                - dict of index, coefficient
                - SollyaObject (sollya polynomial)
        """
        self.degree = None
        self.coeff_map = {}
        self.sollya_object = None

        if isinstance(init_object, list):
            self.degree = len(init_object)
            for coeff_value, index in zip_index(init_object):
                self.coeff_map[index] = coeff_value

        elif isinstance(init_object, dict):
            self.degree = 0
            for index in init_object:
                self.degree = self.degree if index <= self.degree else index
                self.coeff_map[index] = init_object[index]

        elif isinstance(init_object, SollyaObject):
            self.degree = degree(init_object)
            for index in xrange(self.degree+1):
                self.coeff_map[index] = coeff(init_object, index)

        self.sollya_object = 0
        # building sollya object
        for index in self.coeff_map:
            self.sollya_object += self.coeff_map[index] * x**index
            

    def get_sollya_object(self):
        return self.sollya_object

    def get_coeff_num(self):
        """ return the number of coefficients """
        return len(self.coeff_map)

    def get_degree(self):
        """ degree getter """
        return self.degree

    def sub_poly(self, start_index = 0, stop_index = None, offset = 0):
        """ sub polynomial extraction """
        new_coeff_map = {}
        end_index = self.degree + 1 if stop_index == None else stop_index + 1
        for index in range(start_index, end_index):
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)

    def sub_poly_index_list(self, index_list, offset = 0):
        """ sub polynomial extraction from coefficient index list """
        new_coeff_map = {}
        for index in index_list:
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)

    def get_ordered_coeff_list(self):
        """ return the list of (index, coefficient) for the polynomial <self>
            ordered in increasing order """
        coeff_list = []
        for index in self.coeff_map:
            coeff = self.coeff_map[index]
            coeff_list.append((index, coeff))
        coeff_list.sort(key = lambda v: v[0])
        return coeff_list


    def build_from_approximation(function, poly_degree, coeff_formats, approx_interval, *modifiers):
        """ construct a polynomial object from a function approximation using sollya's fpminimax """
        Log.report(Log.Info,  "approx_interval: %s" % approx_interval)
        sollya_poly = fpminimax(function, poly_degree, [c.sollya_object for c in coeff_formats], approx_interval, *modifiers)
        return Polynomial(sollya_poly)

    def build_from_approximation_with_error(function, poly_degree, coeff_formats, approx_interval, *modifiers, **kwords): 
        """ construct a polynomial object from a function approximation using sollya's fpminimax """
        tightness = kwords["tightness"] if "tightness" in kwords else S2**-24
        Log.report(Log.Info,  "approx_interval: %s" % approx_interval)
        sollya_poly = fpminimax(function, poly_degree, [c.sollya_object for c in coeff_formats], approx_interval, *modifiers)
        fpnorm_modifiers = absolute if absolute in modifiers else relative
        approx_error = supnorm(sollya_poly, function, approx_interval, fpnorm_modifiers, tightness)
        return Polynomial(sollya_poly), approx_error

    build_from_approximation = Callable(build_from_approximation)
    build_from_approximation_with_error = Callable(build_from_approximation_with_error)


def generate_power(variable, power, power_map = {}, precision = None):
    """ generate variable^power, using power_map for memoization 
        if precision is defined, every created operation is assigned that format """
    if power in power_map:
        return power_map[power]
    else:
        result = None
        if power == 1:  
            result = variable
        else:
            sub_power = generate_power(variable, power / 2, power_map, precision = precision)
            sub_square = Multiplication(sub_power, sub_power, precision = precision)
            if power % 2 == 1:
                result = Multiplication(variable, sub_square, precision = precision)
            else:
                result = sub_square
        power_map[power] = result
        return result

class PolynomialSchemeEvaluator:
    """ class for polynomial evaluation scheme generation """

    def generate_horner_scheme(polynomial_object, variable, unified_precision = None):
        """ generate a Horner evaluation scheme for the polynomial <polynomial_object>
            on variable <variable>, arithmetic operation are performed in format 
            <unified_precision> if specified """
        power_map = {}
        coeff_list = polynomial_object.get_ordered_coeff_list()[::-1]
        current_index = coeff_list[0][0]
        current_scheme = Constant(coeff_list[0][1])
        for index, coeff in coeff_list[1:]:
            current_coeff = Constant(coeff)

            index_diff = current_index - index
            current_index = index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision = unified_precision)
            current_scheme = Addition(current_coeff, mult_op, precision = unified_precision)
        if current_index > 0:
            last_power = generate_power(variable, current_index, power_map, precision = unified_precision)
            current_scheme = Multiplication(last_power, current_scheme, precision = unified_precision)

        return current_scheme
            

    def generate_estrin_scheme(polynomial_object, variable, unified_precision, power_map = {}):
        """ generate a Estrin evaluation scheme """
        if polynomial_object.get_coeff_num() == 1: 
            index, coeff = polynomial_object.get_ordered_coeff_list()
            coeff_node = Constant(coeff)
            if index == 0:
                return coeff_node
            else: 
                power_node = generate_power(variable, index, power_map, unified_precision)
                return Multiplication(coeff_node, power_node, precision = unified_precision)
        else:
            poly_degree = (polynomial_object.get_degree() + 1) / 2
            sub_poly_lo = polynomial_object.sub_poly(stop_index = poly_degree) 
            sub_poly_hi = polynomial_object.sub_poly(start_index = poly_degree + 1)
            lo_node = generate_estrin_scheme(sub_poly_lo, variable, unified_precision, power_map)
            hi_node = generate_estrin_scheme(sub_poly_hi, variable, unified_precision, power_map)
            return Addition(lo_node, hi_node, precision = unified_precision)


    # class function static binding
    generate_horner_scheme = Callable(generate_horner_scheme)
    generate_estrin_scheme = Callable(generate_estrin_scheme)
