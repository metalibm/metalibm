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

from ..utility.log_report import Log
from .ml_operations import Constant, Variable, Multiplication, Addition, Subtraction
from .ml_formats import ML_Format, ML_FP_Format, ML_Fixed_Format

class Polynomial(object):
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
            for index, coeff_value in enumerate(init_object):
                self.coeff_map[index] = coeff_value

        elif isinstance(init_object, dict):
            self.degree = 0
            for index in init_object:
                self.degree = self.degree if index <= self.degree else index
                self.coeff_map[index] = init_object[index]

        elif isinstance(init_object, SollyaObject):
            self.degree = degree(init_object)
            for index in xrange(self.degree+1):
                coeff_value = coeff(init_object, index)
                if coeff_value != 0:
                  self.coeff_map[index] = coeff_value

        self.sollya_object = 0
        # building sollya object
        for index in self.coeff_map:
            self.sollya_object += self.coeff_map[index] * x**index

    def get_min_monomial_degree(self):
        monomial_degrees = [index for index in self.coeff_map]
        return min(monomial_degrees)
            

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
            if not index in self.coeff_map: continue
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)

    def sub_poly_index_list(self, index_list, offset = 0):
        """ sub polynomial extraction from coefficient index list """
        new_coeff_map = {}
        for index in index_list:
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)


    def sub_poly_cond(self, monomial_cond = lambda i, c: True, offset = 0): 
        """ sub polynomial extraction, each monomial C*x^i verifying monomial_cond(i, C)
            is selected, others are discarded """
        new_coeff_map = {}
        for index in self.coeff_map:
          coeff_value = self.coeff_map[index]
          if monomial_cond(index, coeff_value):
            new_coeff_map[index - offset] = coeff_value
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


    def __str__(self):
        return str(self.get_sollya_object())


    @staticmethod
    def build_from_approximation(function, poly_degree, coeff_formats, approx_interval, *modifiers):
        """ construct a polynomial object from a function approximation using sollya's fpminimax """
        Log.report(Log.Info,  "approx_interval: %s" % approx_interval)
        precision_list = []
        for c in coeff_formats:
          if isinstance(c, ML_FP_Format):
            precision_list.append(c.get_sollya_object())
          elif isinstance(c, ML_Fixed_Format):
            precision_list.append(c.get_bit_size())
          else:
            precision_list.append(c)
        sollya_poly = fpminimax(function, poly_degree, precision_list, approx_interval, *modifiers)
        return Polynomial(sollya_poly)


    ## Approximation computation with built-in approximation error computation
    #  @return a tuple poly_object, error: poly_object is a Polynomial 
    #          approximating the given function on the given interval, 
    #          according to parameter indications, error is the approximation
    #          error on the interval
    #  @param function mathematical function (pythonsollya object) describing the function to be approximated
    #  @param poly_degree the degree of the approximation polynomial request
    #  @param coeff_formats list of coefficient format (as expected by
    #         (python)sollya fpminimax function, which describes the format
    #         of each coefficient of the polynomial approximation
    #  @param approx_interval the interval where the approximation applies
    #  @param modifiers tuple of extra arguments (see (python)sollya's fpminimax documentation for more information, e.g absolute)
    #  @param kwords dictionnary of extra arguments for the approximation computation (e.g tightness, error_function)
    @staticmethod
    def build_from_approximation_with_error(function, poly_degree, coeff_formats, approx_interval, *modifiers, **kwords): 
        """ construct a polynomial object from a function approximation using sollya's fpminimax """
        tightness = kwords["tightness"] if "tightness" in kwords else S2**-24
        error_function = kwords["error_function"] if "error_function" in kwords else lambda p, f, ai, mod, t: supnorm(p, f, ai, mod, t)
        precision_list = []
        for c in coeff_formats:
            if isinstance(c, ML_FP_Format):
                precision_list.append(c.get_sollya_object())
            else:
                precision_list.append(c)
        sollya_poly = fpminimax(function, poly_degree, precision_list, approx_interval, *modifiers)
        fpnorm_modifiers = absolute if absolute in modifiers else relative
        #approx_error = supnorm(sollya_poly, function, approx_interval, fpnorm_modifiers, tightness)
        approx_error = error_function(sollya_poly, function, approx_interval, fpnorm_modifiers, tightness)
        return Polynomial(sollya_poly), approx_error

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
            if power % 2 == 1:
                sub_square = generate_power(variable, power - 1, power_map, precision = precision)
                result_tag = "%s%d_" % (variable.get_tag() if variable.get_tag() else "X", power)
                result = Multiplication(variable, sub_square, precision = precision, tag = result_tag)
            else:
                sub_power = generate_power(variable, power / 2, power_map, precision = precision)
                sub_square_tag = "%s%d_" % (variable.get_tag() if variable.get_tag() else "X", (power / 2) * 2)
                sub_square = Multiplication(sub_power, sub_power, precision = precision, tag = sub_square_tag)
                result = sub_square
        power_map[power] = result
        return result

class PolynomialSchemeEvaluator(object):
    """ class for polynomial evaluation scheme generation """

    @staticmethod
    def generate_horner_scheme(polynomial_object, variable, unified_precision = None, power_map_ = None, constant_precision = None):
        """ generate a Horner evaluation scheme for the polynomial <polynomial_object>
            on variable <variable>, arithmetic operation are performed in format 
            <unified_precision> if specified """
        power_map = power_map_ if power_map_ != None else {}
        coeff_list = polynomial_object.get_ordered_coeff_list()[::-1]
        cst_precision = unified_precision if constant_precision == None else constant_precision
        if len(coeff_list) == 0:
            return Constant(0)
        elif len(coeff_list) == 1:
            index, coeff = coeff_list[0]
            if index == 0:
                return Constant(coeff, precision = cst_precision, tag = "coeff_%d" % index)
            else:
                return Multiplication(generate_power(variable, index, power_map, precision = unified_precision), Constant(coeff, precision = cst_precision), tag = "pm_%d" % index)
            
        current_index = coeff_list[0][0]
        current_scheme = Constant(coeff_list[0][1], precision = cst_precision)
        for index, coeff in coeff_list[1:-1]:
            current_coeff = Constant(coeff, precision = cst_precision, tag = "coeff_%d" % index)

            index_diff = current_index - index
            current_index = index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision = unified_precision, tag = "pm_%d" % index)
            current_scheme = Addition(current_coeff, mult_op, precision = unified_precision, tag = "pa_%d" % index)
        # last coefficient
        index, coeff = coeff_list[-1]
        current_coeff = Constant(coeff, precision = cst_precision, tag = "coeff_%d" % index)
        if (coeff == 1.0 or coeff == -1.0) and index <= 1:
            # generating FMA
            index_diff = current_index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision = unified_precision, tag = "pm_%d" % index)
            if index == 0:
              current_scheme = Addition(current_coeff, mult_op, precision = unified_precision, tag = "pa_%d" % index)
            elif index == 1:
              if coeff == 1.0:
                current_scheme = Addition(variable, mult_op, precision = unified_precision, tag = "pa_%d" % index)
              elif coeff == -1.0:
                current_scheme = Subtraction(mult_op, variable, precision = unified_precision, tag = "pa_%d" % index)

            


        else:
            index_diff = current_index - index
            current_index = index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision = unified_precision)
            current_scheme = Addition(current_coeff, mult_op, precision = unified_precision)

            if current_index > 0:
                last_power = generate_power(variable, current_index, power_map, precision = unified_precision)
                current_scheme = Multiplication(last_power, current_scheme, precision = unified_precision)

        return current_scheme

    @staticmethod
    def generate_estrin_scheme(polynomial_object, variable, unified_precision, power_map_ = None):
        """ generate a Estrin evaluation scheme """
        power_map = power_map_ if power_map_ != None else {}
        if polynomial_object.get_coeff_num() == 1: 
            index, coeff = polynomial_object.get_ordered_coeff_list()[0]
            coeff_node = Constant(coeff)
            if index == 0:
                return coeff_node
            else: 
                power_node = generate_power(variable, index, power_map, unified_precision)
                return Multiplication(coeff_node, power_node, precision = unified_precision)
        else:
            min_degree = int(polynomial_object.get_min_monomial_degree())
            max_degree = int(polynomial_object.get_degree())
            poly_degree = (max_degree - min_degree + 2) / 2 + min_degree - 1
            offset_degree = poly_degree + 1 - min_degree
            sub_poly_lo = polynomial_object.sub_poly(stop_index = poly_degree) 
            sub_poly_hi = polynomial_object.sub_poly(start_index = poly_degree + 1, offset = offset_degree)
            lo_node = PolynomialSchemeEvaluator.generate_estrin_scheme(sub_poly_lo, variable, unified_precision, power_map)
            hi_node = PolynomialSchemeEvaluator.generate_estrin_scheme(sub_poly_hi, variable, unified_precision, power_map)

            offset_degree_monomial = generate_power(variable, offset_degree, power_map, unified_precision)
            return Addition(lo_node, Multiplication(offset_degree_monomial, hi_node, precision = unified_precision), precision = unified_precision)
