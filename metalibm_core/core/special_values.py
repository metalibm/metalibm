# -*- coding: utf-8 -*-

## @package special_values
#  Metalibm Formats special_values

###############################################################################
# This file is part of the New Metalibm tool
# Copyright (2017-)
# All rights reserved
# created:          Aug 20th, 2017
# last-modified:    Aug 20th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################

import sollya

class NumericValue(sollya.SollyaObject):
    """ numerical object encapsulating sollya's number to 
        provide interaction with special values """
    def __add__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs + lhs
        else:
            return sollya.SollyaObject.__add__(lhs, rhs)
    def __mul__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs + lhs
        else:
            return sollya.SollyaObject.__mul__(lhs, rhs)
    def __sub__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs.__rsub__(lhs)
        else:
            return sollya.SollyaObject.__sub__(lhs, rhs)

            
###############################################################################
#                     FLOATING-POINT SPECIAL VALUES
###############################################################################
class FP_SpecialValue(object):
  ml_support_name = "undefined"

  """ parent to all floating-point constants """
  def __init__(self, precision):
    self.precision = precision

  def get_c_cst(self):
    prefix = self.precision.get_ml_support_prefix()
    suffix = "." + self.precision.get_union_field_suffix()
    return prefix + self.ml_support_name + suffix

  def __str__(self):
    return "%s" % (self.ml_support_name)

  def get_precision(self):
    return self.precision

  def get_integer_coding(self):
    return NotImplementedError

  @staticmethod
  def is_special_value(value):
    """ Predicate testing if value is a special value object """
    return isinstance(value, FP_SpecialValue)

  def __add__(lhs, rhs):
    return special_value_add(lhs, rhs)
  def __radd__(rhs, lhs):
    return special_value_add(lhs, rhs)

  def __sub__(lhs, rhs):
    return special_value_sub(lhs, rhs)
  def __rsub__(rhs, lhs):
    return special_value_sub(lhs, rhs)

  def __mul__(lhs, rhs):
    return special_value_mul(lhs, rhs)
  def __rmul__(rhs, lhs):
    return special_value_mul(lhs, rhs)

def FP_SpecialValue_get_c_cst(self):
    prefix = self.precision.get_ml_support_prefix()
    suffix = "." + self.precision.get_union_field_suffix()
    return prefix + self.ml_support_name + suffix

def FP_SpecialValue_init(self, precision):
    self.precision = precision

def FP_SpecialValue_get_str(self):
    return "%s" % (self.ml_support_name)


class FP_MathSpecialValue(FP_SpecialValue):
    def get_c_cst(self):
        return self.ml_support_name


def special_value_add(lhs, rhs):
    """ Addition between special values or between special values and
        numbers """
    if is_nan(lhs) or is_nan(rhs):
        return FP_QNaN(lhs.precision)
    elif (is_plus_infty(lhs) and is_minus_infty(rhs)) or \
         (is_minus_infty(rhs) and is_plus_infty(rhs)):
        return FP_QNaN(lhs.precision)
    elif is_infty(lhs) and is_infty(rhs):
        # non-symmetrical infty case have already been excluded
        return lhs
    elif is_zero(lhs) and is_zero(rhs):
        # TODO: ignore rounding mode
        return lhs
    elif is_zero(lhs):
        return rhs
    elif is_zero(rhs):
        return lhs
    elif is_infty(lhs) and is_number(rhs):
        return lhs
    elif is_number(lhs) and is_infty(rhs):
        return rhs
    elif is_number(lhs) and is_number(rhs):
        return lhs + rhs
    else:
        raise NotImplementedError

def special_value_sub(lhs, rhs):
    """ Subtraction between special values or between special values and
        numbers """
    if is_nan(lhs):
        return FP_QNaN(lhs.precision)
    elif is_nan(rhs):
        return FP_QNaN(rhs.precision)
    elif (is_plus_infty(lhs) and is_plus_infty(rhs)) or \
         (is_minus_infty(lhs) and is_minus_infty(rhs)):
        return FP_QNaN(lhs.precision)
    elif is_plus_infty(lhs) and is_minus_infty(rhs):
        return lhs
    elif is_minus_infty(lhs) and is_plus_infty(rhs):
        return lhs
    elif is_infty(lhs) and is_zero(rhs):
        return lhs
    elif is_infty(lhs):
        # invalid inf - inf excluded previous
        return lhs
    elif is_infty(rhs):
        return -rhs
    else:
        return lhs + (-rhs)

def special_value_mul(lhs, rhs):
    """ Multiplication between special values or between special values and
        numbers """
    if is_nan(lhs) or is_nan(rhs):
        return FP_QNaN(lhs.precision)
    elif (is_plus_infty(lhs) and is_plus_infty(rhs)) or \
         (is_minus_infty(rhs) and is_minus_infty(rhs)):
        return FP_PlusInfty(lhs.precision)
    elif is_infty(lhs) and is_infty(rhs):
        # positive infinity results are processed by the previous case
        return FP_MinusInfty(lhs.precision)
    elif is_infty(lhs) and is_zero(rhs):
        return FP_QNaN(lhs.precision)
    elif is_zero(lhs) and is_infty(rhs):
        return FP_QNaN(rhs.precision)
    elif is_number(rhs) and is_plus_infty(lhs):
        return FP_PlusInfty(lhs.precision) if rhs > 0 else FP_MinusInfty(lhs.precision)
    elif is_number(rhs) and is_minus_infty(lhs):
        return FP_PlusInfty(lhs.precision) if rhs < 0 else FP_MinusInfty(lhs.precision)
    elif is_number(lhs) and is_plus_infty(rhs):
        return FP_PlusInfty(rhs.precision) if lhs > 0 else FP_MinusInfty(rhs.precision)
    elif is_number(lhs) and is_minus_infty(rhs):
        return FP_PlusInfty(rhs.precision) if lhs < 0 else FP_MinusInfty(rhs.precision)
    elif is_number(lhs) and is_number(rhs):
        return lhs * rhs
    elif (is_plus_zero(lhs) and is_plus_zero(rhs)) or \
         (is_minus_zero(lhs) and is_minus_zero(rhs)):
        return FP_PlusZero(lhs.precision)
    elif is_zero(lhs) and is_zero(rhs):
        return FP_MinusZero(lhs.precision)
    elif (is_zero(lhs) and is_positive(rhs)):
        return lhs
    elif (is_zero(rhs) and is_positive(lhs)):
        return rhs
    elif is_zero(lhs) and is_negative(rhs):
        return -lhs
    elif is_zero(rhs) and is_negative(lhs):
        return -rhs
    else:
        raise NotImplementedError

def is_positive(value):
    return is_plus_zero(value) or is_plus_infty(value) or \
        (is_number(value) and value >= 0)
def is_negative(value):
    return is_minus_zero(value) or is_minus_infty(value) or \
        (is_number(value) and value < 0)

class FP_PlusInfty(FP_MathSpecialValue):
  ml_support_name = "INFINITY"
  def get_integer_coding(self):
    exp = self.precision.get_nanorinf_exp_field()
    return exp << self.precision.get_field_size()
  def __str__(self):
    return "+inf"
  def __neg__(self):
    return FP_MinusInfty(self.precision)

class FP_MinusInfty(FP_SpecialValue):
  ml_support_name = "_sv_MinusInfty"
  def get_integer_coding(self):
    exp = self.precision.get_nanorinf_exp_field()
    sign = 1
    field_size = self.precision.get_field_size()
    exp_size = self.precision.get_exponent_size()
    return ((sign << exp_size) | exp) << field_size
  def __str__(self):
    return "-inf"
  def __neg__(self):
    return FP_PlusInfty(self.precision)

class FP_PlusOmega(FP_SpecialValue):
  ml_support_name = "_sv_PlusOmega"
  def get_integer_coding(self):
    return self.precision.get_integer_coding(self.precision.get_omega())
class FP_MinusOmega(FP_SpecialValue):
  ml_support_name = "_sv_MinusOmega"
  def get_integer_coding(self):
    return self.precision.get_integer_coding(-self.precision.get_omega())
class FP_PlusZero(FP_SpecialValue):
  ml_support_name = "_sv_PlusZero"
  def get_integer_coding(self):
    return 0
  def __str__(self):
    return "+0"
  def __neg__(self):
    return FP_MinusZero(self.precision)

class FP_MinusZero(FP_SpecialValue):
  ml_support_name = "_sv_MinusZero"
  def get_integer_coding(self):
    sign = 1
    field_size = self.precision.get_field_size()
    exp_size = self.precision.get_exponent_size()
    return sign << (exp_size + field_size)
  def __str__(self):
    return "-0"
  def __neg__(self):
    return FP_PlusZero(self.precision)

class FP_QNaN(FP_MathSpecialValue):
  """ Floating-point quiet NaN """
  ml_support_name = "NAN"
  def get_integer_coding(self):
    exp = self.precision.get_nanorinf_exp_field()
    sign = 1
    field_size = self.precision.get_field_size()
    exp_size = self.precision.get_exponent_size()
    ## field MSB is 0
    mant = S2**(field_size - 1) - 1 
    return mant | (((sign << exp_size) | exp) << field_size)
  def __str__(self):
    return "qNaN"

class FP_SNaN(FP_SpecialValue):
  """ Floating-point signaling NaN """
  ml_support_name = "_sv_SNaN"
  def get_integer_coding(self):
    exp = self.precision.get_nanorinf_exp_field()
    sign = 1
    field_size = self.precision.get_field_size()
    exp_size = self.precision.get_exponent_size()
    ## field MSB is 1
    mant = S2**(field_size) - 1 
    return mant | (((sign << exp_size) | exp) << field_size)
  def __str__(self):
    return "sNaN"

def is_nan(value):
    """ testing if a value is an instance of a NaN """
    return isinstance(value, FP_QNaN) or isinstance(value, FP_SNaN)
def is_plus_zero(value):
    return isinstance(value, FP_PlusZero) 
def is_minus_zero(value):
    return isinstance(value, FP_MinusZero)
def is_zero(value):
    return is_plus_zero(value) or is_minus_zero(value) or value == 0
def is_plus_infty(value):
    return isinstance(value, FP_PlusInfty)
def is_minus_infty(value):
    return isinstance(value, FP_MinusInfty)
def is_infty(value):
    return is_plus_infty(value) or is_minus_infty(value)
def is_number(value):
    """ Only partially valid """
    return not isinstance(value, FP_SpecialValue)


if __name__ == "__main__":
    PRECISION = ML_Binary64
    value_list = [
        FP_PlusInfty(PRECISION),
        FP_MinusInfty(PRECISION),
        FP_PlusZero(PRECISION),
        FP_MinusZero(PRECISION),
        FP_QNaN(PRECISION),
        FP_SNaN(PRECISION),
        NumericValue(7.0),
        NumericValue(-3.0),
    ]
    op_map = {
        "+": operator.__add__,
        "-": operator.__sub__,
        "*": operator.__mul__,
    }
    for op in op_map:
        for lhs in value_list:
            for rhs in value_list:
                print "{} {} {} = ".format(lhs, op, rhs),
                print "{}".format(op_map[op](lhs, rhs))
            

