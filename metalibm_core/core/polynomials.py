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

import sollya

from sollya import SollyaObject, coeff
S2 = SollyaObject(2)
from ..utility.log_report import Log


def is_cst_with_value(coeff, value):
    """ Predicate testing if coeff is a constant node with numerical value value

        Args:
            coeff (ML_Operation): node to be tested
            value (int, float, ...): numerical value 
        Return:
            bool: True if coeff is Constant(value), False otherwise
    """
    return isinstance(coeff, Constant) and coeff.get_value() == value

try:
  from cgpe import (
          PolynomialScheme as CgpePolynomialScheme,
          Variable as CgpeVar,
          Constant as CgpeConstant,
          Addition as CgpeAdd,
          Multiplication as CgpeMul,
          Subtraction as CgpeSub,
          CgpeDriver,
          )
  cpge_available = True
except ImportError:
  cpge_available = False
  Log.report(Log.Warning, "CPGE import failed")

from .ml_operations import Constant, Variable, Multiplication, Addition, Subtraction
from .ml_formats import ML_Format, ML_FP_Format, ML_Fixed_Format


def is_cgpe_available():
  return cpge_available

class SollyaError(Exception):
    """ Exception to indicate an error in pythonsollya """
    pass

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
            self.degree = int(sollya.degree(init_object))
            for index in range(int(self.degree)+1):
                coeff_value = coeff(init_object, index)
                if coeff_value != 0:
                  self.coeff_map[index] = coeff_value

        self.sollya_object = 0
        # building sollya object
        for index in self.coeff_map:
            self.sollya_object += self.coeff_map[index] * sollya.x**index

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

    def sub_poly(self, start_index=0, stop_index=None, offset = 0):
        """ sub polynomial extraction """
        new_coeff_map = {}
        end_index = self.degree + 1 if stop_index == None else stop_index + 1
        for index in range(int(start_index), int(end_index)):
            if not index in self.coeff_map: continue
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)

    def sub_poly_index_list(self, index_list, offset = 0):
        """ sub polynomial extraction from coefficient index list """
        new_coeff_map = {}
        for index in index_list:
            new_coeff_map[index - offset] = self.coeff_map[index]
        return Polynomial(new_coeff_map)

    def get_cst_coeff(self, index, precision=None):
        return Constant(self.coeff_map[index], precision=precision)


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

        sollya_poly = sollya.fpminimax(function, poly_degree, precision_list,
                                       approx_interval, *modifiers)
        while sollya_poly.is_error() and sollya.settings.points < 10000:
            # We don't want sollya.settings.points to be too large. A value <
            # 20000 does not impact too much the timings for the moment.
            # We also give an odd value to sollya.settings.points (even though
            # it should not be needed anymore) to avoid errors when working on a
            # symmetric interval. See this clear explanation by Sylvain
            # Chevillard on the Sollya mailing list at
            # https://lists.gforge.inria.fr/pipermail/sollya-users/2017-August/000056.html
            sollya.settings.points = 2 * sollya.settings.points - 1
            Log.report(Log.Warning,
                       "Trying with more points: {}"
                       .format(sollya.settings.points))
            sollya_poly = sollya.fpminimax(function, poly_degree,
                                           precision_list, approx_interval,
                                           *modifiers)

        # Reset points to its default value
        sollya.settings.points = sollya.default

        if sollya_poly.is_error():
            # We could try other parameters before crashing Metalibm:
            #   * increase the Sollya 'points' variable even more
            #   * slightly relax approx_interval bounds
            raise SollyaError

        return Polynomial(sollya_poly)


    ## Approximation computation with built-in approximation error computation
    #  @return a tuple poly_object, error: poly_object is a Polynomial
    #          approximating the given function on the given interval,
    #          according to parameter indications, error is the approximation
    #          error on the interval
    #  @param function mathematical function (pythonsollya object) describing
    #         the function to be approximated
    #  @param poly_degree the degree of the approximation polynomial request
    #  @param coeff_formats list of coefficient format (as expected by
    #         (python)sollya fpminimax function, which describes the format
    #         of each coefficient of the polynomial approximation
    #  @param approx_interval the interval where the approximation applies
    #  @param modifiers tuple of extra arguments (see (python)sollya's fpminimax
    #         documentation for more information, e.g absolute)
    #  @param kwords dictionnary of extra arguments for the approximation
    #         computation (e.g tightness, error_function)
    @staticmethod
    def build_from_approximation_with_error(
            function, poly_degree, coeff_formats, approx_interval,
            *modifiers, **kwords):
        """ construct a polynomial object from a function approximation using
            sollya's fpminimax """
        tightness = kwords["tightness"] if "tightness" in kwords else S2**-24
        error_function = kwords["error_function"] if "error_function" in kwords else lambda p, f, ai, mod, t: sollya.supnorm(p, f, ai, mod, t)
        precision_list = []
        for c in coeff_formats:
            if isinstance(c, ML_FP_Format):
                precision_list.append(c.get_sollya_object())
            else:
                precision_list.append(c)
        sollya_poly = sollya.fpminimax(function, poly_degree, precision_list, approx_interval, *modifiers)
        if sollya_poly.is_error():
            print("function: {}, poly_degree: {}, precision_list: {}, approx_interval: {}, modifiers: {}".format(function, poly_degree, precision_list, approx_interval, modifiers))
            raise SollyaError()

        fpnorm_modifiers = sollya.absolute if sollya.absolute in modifiers else sollya.relative
        #approx_error = sollya.supnorm(sollya_poly, function, approx_interval, fpnorm_modifiers, tightness)
        approx_error = error_function(sollya_poly, function, approx_interval, fpnorm_modifiers, tightness)
        return Polynomial(sollya_poly), approx_error

def generate_power(variable, power, power_map = {}, precision = None):
    """ generate variable^power, using power_map for memoization
        if precision is defined, every created operation is assigned
        that format """
    power_key = (power, precision)
    try:
        return power_map[power_key]
    except KeyError:
        result = None
        if power == 1:
            result = variable
        else:
            if power % 2 == 1:
                sub_square = generate_power(
                    variable, power - 1, power_map, precision=precision)
                result_tag = "%s%d_" % (variable.get_tag() or "X", power)
                result = Multiplication(variable, sub_square,
                                        precision=precision, tag=result_tag)
            else:
                sub_power = generate_power(variable, power // 2, power_map,
                                          precision=precision)
                sub_square_tag = "%s%d_" % (variable.get_tag() or "X", (power / 2) * 2)
                sub_square = Multiplication(sub_power, sub_power,
                                            precision=precision,
                                            tag=sub_square_tag)
                result = sub_square
        # memoization
        power_map[power_key] = result
        return result

class PolynomialSchemeEvaluator(object):
    """ class for polynomial evaluation scheme generation """

    @staticmethod
    def generate_horner_scheme(polynomial_object, variable,
            unified_precision=None, power_map_=None, constant_precision = None):
        """ generate a Horner evaluation scheme for the polynomial <polynomial_object>
            on variable <variable>, arithmetic operation are performed in format
            <unified_precision> if specified """
        cst_precision = unified_precision if constant_precision == None else constant_precision
        coeff_list = [
            (index, Constant(coeff, precision=cst_precision, tag="c_{}".format(index))) 
            for (index, coeff) in polynomial_object.get_ordered_coeff_list()[::-1]
        ]
        return PolynomialSchemeEvaluator.generate_horner_scheme2(
            coeff_list, variable, unified_precision,
            power_map_, constant_precision)

    @staticmethod
    def generate_horner_scheme2(coeff_list, variable,
            unified_precision=None, power_map_=None, constant_precision = None):
        """ generate a Horner evaluation scheme for the list of pairs (coeff, index)
            <coeff_list>
            on variable <variable>, arithmetic operation are performed in format 
            <unified_precision> if specified """
        power_map = power_map_ if power_map_ != None else {}
        cst_precision = unified_precision if constant_precision is None else constant_precision
        if len(coeff_list) == 0:
            return Constant(0)
        elif len(coeff_list) == 1:
            index, coeff = coeff_list[0]
            if index == 0:
                return coeff
            else:
                return Multiplication(
                    generate_power(variable, index, power_map, precision = unified_precision),
                    coeff, tag="pm_%d" % index
                )

        current_index = coeff_list[0][0]
        current_scheme = coeff_list[0][1]
        for index, coeff in coeff_list[1:-1]:
            current_coeff = coeff 

            index_diff = current_index - index
            current_index = index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision=unified_precision, tag="pm_%d" % index)
            current_scheme = Addition(current_coeff, mult_op, precision=unified_precision, tag="pa_%d" % index)
        # last coefficient
        index, coeff = coeff_list[-1]
        current_coeff = coeff
        # operation optimization
        if (is_cst_with_value(coeff, 1.0) or is_cst_with_value(coeff, -1.0)) and index <= 1:
            # generating FMA
            index_diff = current_index

            diff_power = generate_power(variable, index_diff, power_map, precision = unified_precision)
            mult_op = Multiplication(diff_power, current_scheme, precision = unified_precision, tag = "pm_%d" % index)
            if index == 0:
              current_scheme = Addition(current_coeff, mult_op, precision = unified_precision, tag = "pa_%d" % index)
            elif index == 1:
              if is_cst_with_value(coeff, 1.0):
                current_scheme = Addition(variable, mult_op, precision = unified_precision, tag = "pa_%d" % index)
              elif is_cst_with_value(coeff,-1.0):
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
    def generate_adaptative_horner_scheme(poly_object, variable,
            error_constraint,
            out_precision=None,
            start_precision=None,
            approx_interval=None):
        """ Generate a horner evaluation scheme for poly_object
            which enforces the error_constraint : the overall evaluation
            error must be less than error_constraint"""
        # setting output precision
        out_precision = out_precision or variable.get_precision()
        # setting start precision (higher degree monomial)
        start_precision = start_precision or variable.get_precision()
        # setting approximation interval
        approx_interval = approx_interval or variable.get_interval()
        # coefficients in reverse order
        coeff_list = poly_object.get_ordered_coeff_list()[::-1]
        # initial approx
        highest_coeff = coeff_list.pop(0)
        initial_error_constraint = error_constraint
        raise NotImplementedError


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
            poly_degree = (max_degree - min_degree + 2) // 2 + min_degree - 1
            offset_degree = poly_degree + 1 - min_degree
            sub_poly_lo = polynomial_object.sub_poly(stop_index = poly_degree)
            sub_poly_hi = polynomial_object.sub_poly(start_index = poly_degree + 1, offset = offset_degree)
            lo_node = PolynomialSchemeEvaluator.generate_estrin_scheme(sub_poly_lo, variable, unified_precision, power_map)
            hi_node = PolynomialSchemeEvaluator.generate_estrin_scheme(sub_poly_hi, variable, unified_precision, power_map)

            offset_degree_monomial = generate_power(variable, offset_degree, power_map, unified_precision)
            return Addition(lo_node, Multiplication(offset_degree_monomial, hi_node, precision = unified_precision), precision = unified_precision)

    @staticmethod
    def generate_cgpe_scheme(polynomial_object, variable,
                             unified_precision=None, power_map={},
                             constant_precision=None, scheme_id = 0):
        """PolynomialSchemeEvaluator.generate_cgpe_scheme(...)

        Return a polynomial scheme generated by CGPE.

        polynomial_object -- a Polynomial object
        variable -- the Variable at which the polynomial is to be evaluated.
        unified_precision -- unused yet (default None)
        power_map -- unused yet (default {})
        constant_precision -- unused yet (default None)
        """
        degree = polynomial_object.get_degree()
        cgpe_coeff_list = [
                (index in polynomial_object.coeff_map)
                for index in range(degree+1)
                ] # True/False list for coefficients
        cgpe = CgpeDriver()
        # XXX latencies for add/mul could be retrieved from target specs...
        if scheme_id == 0:
            scheme = cgpe.get_low_latency_scheme(degree, cgpe_coeff_list)
        else:
            schemes = cgpe.get_expressions(degree, cgpe_coeff_list)
            if scheme_id > len(schemes):
                return None
            scheme = schemes[scheme_id]
        del cgpe # Safer because of internal issues not fixed yet.

        print("CGPE scheme is: {}".format(scheme))

        def cgpe_to_metalibm(node):
            """Traverse the AST and convert nodes to Metalibm nodes."""
            if isinstance(node, CgpeVar):
                return variable
            elif isinstance(node, CgpeConstant):
                # We make use of CGPE --no-indices-contraction option
                index = int(node.name[1:]) # First char is 'a'
                value = polynomial_object.coeff_map[index]
                return Constant(value, tag="poly_coeff{}".format(index),
                                precision = constant_precision)
            elif isinstance(node, CgpeAdd):
                return Addition(cgpe_to_metalibm(node.op1),
                        cgpe_to_metalibm(node.op2), tag='poly_add',
                        unified_precision = unified_precision)
            elif isinstance(node, CgpeMul):
                return Multiplication(cgpe_to_metalibm(node.op1),
                        cgpe_to_metalibm(node.op2), tag='poly_mul',
                        unified_precision = unified_precision)
            elif isinstance(node, CgpeSub):
                return Subtraction(cgpe_to_metalibm(node.op1),
                        cgpe_to_metalibm(node.op2), tag='poly_sub',
                        unified_precision = unified_precision)
            elif isinstance(node, CgpePolynomialScheme):
                return cgpe_to_metalibm(node.root)
            else:
                raise ValueError

        return cgpe_to_metalibm(scheme)
