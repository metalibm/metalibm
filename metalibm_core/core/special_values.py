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
# created:          Aug 20th, 2017
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
###############################################################################

## @package special_values
#  Metalibm Formats special_values

import collections
import sollya

SOLLYA_NAN   = sollya.parse("nan")
SOLLYA_INFTY = sollya.parse("infty")

# generic number 2 as a SollyaObject (to force conversion
# to sollya objects when doing arithmetic)
S2 = sollya.SollyaObject(2)

class NumericValue(sollya.SollyaObject):
    """ numerical object encapsulating sollya's number to
        provide interaction with special values """
    def __add__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs + lhs
        else:
            return NumericValue(sollya.SollyaObject.__add__(lhs, rhs))
    def __mul__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs * lhs
        else:
            return NumericValue(sollya.SollyaObject.__mul__(lhs, rhs))
    def __sub__(lhs, rhs):
        if FP_SpecialValue.is_special_value(rhs):
            return rhs.__rsub__(lhs)
        else:
            return NumericValue(sollya.SollyaObject.__sub__(lhs, rhs))

    def __lt__(lhs, rhs):
        # TODO/FIXME: could be optimized as we know lhs to be a numerical value
        return number_less_than_special_value(lhs, rhs)
    def __le__(lhs, rhs):
        # TODO/FIXME: could be optimized as we know lhs to be a numerical value
        return rhs >= lhs
    def __gt__(lhs, rhs):
        # TODO/FIXME: could be optimized as we know lhs to be a numerical value
        return number_less_than_special_value(rhs, lhs)
    def __ge__(lhs, rhs):
        # TODO/FIXME: could be optimized as we know lhs to be a numerical value
        return negate_predicate(number_less_than_special_value(lhs, rhs))
    def __eq__(lhs, rhs):
        if not is_numeric_value(rhs):
            return False
        else:
            return sollya.SollyaObject.__eq__(sollya.SollyaObject(lhs), sollya.SollyaObject(rhs))
    def __ne__(lhs, rhs):
        return negate_predicate(NumericValue.__eq__(lhs, rhs))

    @staticmethod
    def parse(s, precision=None):
        """ parse a numerical value from a string """
        obj = sollya.parse(s)
        if obj == SOLLYA_INFTY:
            return FP_PlusInfty(precision)
        elif obj == -SOLLYA_INFTY:
            return FP_MinusInfty(precision)
        elif obj != obj:
            # by default Sollya's NaNs are assumed to be quiet NaNs
            return FP_QNaN(precision)
        else:
            return NumericValue(obj)


###############################################################################
#                     FLOATING-POINT SPECIAL VALUES
###############################################################################
class FP_SpecialValue(object):
  ml_support_name = "undefined"

  # Rounding-mode (should be None or a rounding mode object
  # from sollya.
  # It will be used for operation on specific values whose result
  # depends on the rounding mode (e.g. +0 + -0)
  # This is a global/class value which must be handled with care
  # (save/restore)
  rounding_mode = None

  """ parent to all floating-point constants """
  def __init__(self, precision):
    self.precision = precision

  def get_c_cst(self):
    prefix = self.get_base_precision().get_ml_support_prefix()
    suffix = "." + self.get_base_precision().get_union_field_suffix()
    return prefix + self.ml_support_name + suffix

  def __str__(self):
    return "%s" % (self.ml_support_name)

  def get_base_precision(self):
    return self.precision.get_base_format()

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


  def __lt__(lhs, rhs):
    return special_value_less_than(lhs, rhs)
  def __le__(lhs, rhs):
    return rhs >= lhs
  def __gt__(lhs, rhs):
    return special_value_less_than(rhs, lhs)
  def __ge__(lhs, rhs):
    return negate_predicate(special_value_less_than(lhs, rhs))
  def __eq__(lhs, rhs):
    return special_value_equal(lhs, rhs)
  def __ne__(lhs, rhs):
    return negate_predicate(special_value_equal(lhs, rhs))

def FP_SpecialValue_get_c_cst(self):
    prefix = self.get_base_precision().get_ml_support_prefix()
    suffix = "." + self.get_base_precision().get_union_field_suffix()
    return prefix + self.ml_support_name + suffix

def FP_SpecialValue_init(self, precision):
    self.precision = precision

def FP_SpecialValue_get_str(self):
    return "%s" % (self.ml_support_name)


class FP_MathSpecialValue(FP_SpecialValue):
    # define if the quiet bit (field MSB) is set for qNaN (True)
    # or for sNaN (False)
    QUIET_BIT_SET_FOR_QNAN = True
    def get_c_cst(self):
        return self.ml_support_name

def negate_predicate(predicate):
    """ inverse the given predicate result
        (True -> False, False -> True, Unordered -> Unordered) """
    if predicate is Unordered:
        return Unordered
    else:
        return not predicate

class UnorderedPredicate:
    """ Unordered comparison predicate """
    def __str__(self):
        return "Unordered"
    def __repr__(self):
        return "Unordered"
    def __bool__(self):
        raise NotImplementedError

Unordered = UnorderedPredicate()

def number_less_than_special_value(lhs, rhs):
    if is_nan(rhs):
        return Unordered
    elif is_minus_infty(rhs):
        return True
    elif is_plus_infty(rhs):
        return False
    elif is_zero(rhs):
        return sollya.SollyaObject.__lt__(sollya.SollyaObject(lhs), sollya.SollyaObject(0))
    elif is_number(rhs):
        return sollya.SollyaObject.__lt__(sollya.SollyaObject(lhs), sollya.SollyaObject(rhs))
    else:
        raise NotImplementedError

def special_value_less_than(lhs, rhs):
    """ Implementation of inequality predicate lhs < rhs

        :param lhs: left hand side operand
        :type lhs: FP_SpecialValue (or NumericValue)
        :param rhs: right hand side operand
        :type rhs: FP_SpecialValue (or NumericValue)
        :return: predicate value (True, False or Unordered)
        """
    if is_nan(lhs) or is_nan(rhs):
        return Unordered
    elif is_minus_infty(lhs):
        if is_minus_infty(rhs):
            return False
        else:
            return True
    elif is_zero(lhs):
        if is_zero(rhs):
            return False
        elif is_plus_infty(rhs):
            return True
        elif is_minus_infty(rhs):
            return False
        elif is_number(rhs):
            return 0 < rhs
        else:
            raise NotImplementedError
    elif is_plus_infty(lhs):
        return False
    elif is_numeric_value(lhs):
        if is_numeric_value(rhs):
            return lhs < rhs
        elif is_minus_infty(rhs):
            return False
        elif is_plus_infty(rhs):
            return True
        elif is_zero(rhs):
            return lhs < 0
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def special_value_equal(lhs, rhs):
    """ Implementation of equality predicate lhs == rhs

        :param lhs: left hand side operand
        :type lhs: FP_SpecialValue (or NumericValue)
        :param rhs: right hand side operand
        :type rhs: FP_SpecialValue (or NumericValue)
        :return: predicate value (True, False or Unordered)
        """
    if is_plus_infty(lhs):
        return is_plus_infty(rhs)
    elif is_minus_infty(lhs):
        return is_minus_infty(rhs)
    elif is_zero(lhs):
        # zeros compare equal regardless of their sign
        return is_zero(rhs)
    elif is_nan(lhs):
        return False
    elif is_numeric_value(lhs):
        if is_special_value(rhs):
            return False
        else:
            return sollya.SollyaObject.__eq__(sollya.SollyaObject(lhs), sollya.SollyaObject(rhs))
    else:
        raise NotImplementedError




def special_value_add(lhs, rhs):
    """ Addition between special values or between special values and
        numbers """
    if is_nan(lhs) or is_nan(rhs):
        return FP_QNaN(lhs.precision)
    elif (is_plus_infty(lhs) and is_minus_infty(rhs)) or \
         (is_minus_infty(lhs) and is_plus_infty(rhs)):
        return FP_QNaN(lhs.precision)
    elif is_infty(lhs) and is_infty(rhs):
        # non-symmetrical infty case have already been excluded
        return lhs
    elif is_zero(lhs) and is_zero(rhs):
        # TODO: ignore rounding mode
        if is_minus_zero(lhs) or is_minus_zero(rhs):
            if FP_SpecialValue.rounding_mode is None:
                Log.report(Log.Warning, "undefined rounding mode during +/- 0 + +/- 0 operation in special_value_add")
                return lhs
            # -0 + -0 is always -0, regardless of rounding mode
            elif (is_minus_zero(lhs) and is_minus_zero(rhs)) or FP_SpecialValue.rounding_mode is sollya.RD:
                return FP_MinusZero(lhs.precision)
            else:
                return FP_PlusZero(lhs.precision)
        else:
            # -0 + -0 has been excluded previously
            return FP_PlusZero(lhs.precision)
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
         (is_minus_infty(lhs) and is_minus_infty(rhs)):
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
    exp = int(self.get_base_precision().get_nanorinf_exp_field())
    return exp << self.get_base_precision().get_field_size()
  def __str__(self):
    return "+inf"
  def __neg__(self):
    return FP_MinusInfty(self.precision)

class FP_MinusInfty(FP_MathSpecialValue):
  ml_support_name = "-INFINITY" # _sv_MinusInfty"
  def get_integer_coding(self):
    exp = int(self.get_base_precision().get_nanorinf_exp_field())
    sign = 1
    field_size = self.get_base_precision().get_field_size()
    exp_size = self.get_base_precision().get_exponent_size()
    return ((sign << exp_size) | exp) << field_size
  def __str__(self):
    return "-inf"
  def __neg__(self):
    return FP_PlusInfty(self.precision)


# TODO: management of omega induces extra cost for any non omega value
#      should be optimized
class FP_NumericSpecialValue(FP_SpecialValue):
    def __le__(self, y):
        return self.get_value() <= (y)
    def __lt__(self, y):
        return self.get_value() < (y)
    def __ge__(self, y):
        return self.get_value() >= (y)
    def __gt__(self, y):
        return self.get_value() > (y)
    def __eq__(self, y):
        return self.get_value() == (y)
    def __ne__(self, y):
        return self.get_value() != (y)
    def __abs__(self):
        """ Absolute implementation for numerical value """
        return abs(self.get_value())

def FP_PlusOmega(precision):
    """ alias for positive omega value """
    return NumericValue(precision.get_base_format().get_omega())
def FP_MinusOmega(precision):
    """ alias for positive omega value """
    return NumericValue(-precision.get_base_format().get_omega())


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
    field_size = self.get_base_precision().get_field_size()
    exp_size = self.get_base_precision().get_exponent_size()
    return sign << (exp_size + field_size)
  def __str__(self):
    return "-0"
  def __neg__(self):
    return FP_PlusZero(self.precision)

class FP_QNaN(FP_MathSpecialValue):
  """ Floating-point quiet NaN """
  ml_support_name = "NAN"
  ml_nan_field = (lambda self, field_size: (1, (int(S2**(field_size - 1) - 1))))
  def get_integer_coding(self):
    exp = int(self.get_base_precision().get_nanorinf_exp_field())
    field_size = self.get_base_precision().get_field_size()
    exp_size = self.get_base_precision().get_exponent_size()
    ## field MSB is set according to FP_MathSpecialValue.QUIET_BIT_SET_FOR_QNAN
    quiet_bit = (1 << (field_size - 1)) if FP_MathSpecialValue.QUIET_BIT_SET_FOR_QNAN else 0
    sign, mant = FP_QNaN.ml_nan_field(self, field_size)
    mant |= quiet_bit
    return mant | (((sign << exp_size) | exp) << field_size)
  def get_c_cst(self):
    if self.precision.get_bit_size() == 32:
        return "__builtin_nanf(\"\")"
    elif self.precision.get_bit_size() == 64:
        return "__builtin_nan(\"\")"
    else:
        super().get_c_cst()
  def __str__(self):
    return "qNaN"
  def _sollya_(self):
    return SOLLYA_NAN

class FP_SNaN(FP_SpecialValue):
  """ Floating-point signaling NaN """
  ml_support_name = "_sv_SNaN"
  def get_integer_coding(self):
    exp = int(self.get_base_precision().get_nanorinf_exp_field())
    sign = 1
    field_size = self.get_base_precision().get_field_size()
    exp_size = self.get_base_precision().get_exponent_size()
    ## field MSB is set according to FP_MathSpecialValue.QUIET_BIT_SET_FOR_QNAN
    quiet_bit = (1 << (field_size - 1)) if not FP_MathSpecialValue.QUIET_BIT_SET_FOR_QNAN else 0
    mant = int(S2**(field_size - 1) - 1) | quiet_bit
    return mant | (((sign << exp_size) | exp) << field_size)
  def __str__(self):
    return "sNaN"
  def _sollya_(self):
    return SOLLYA_NAN

def is_qnan(value):
    """ testing if a value is an instance of a quiet NaN """
    return isinstance(value, FP_QNaN)
def is_snan(value):
    """ testing if a value is an instance of a signaling NaN """
    return isinstance(value, FP_SNaN)
def is_nan(value):
    """ testing if a value is an instance of a NaN """
    return is_qnan(value) or is_snan(value)

def is_plus_zero(value):
    return isinstance(value, FP_PlusZero)
def is_minus_zero(value):
    return isinstance(value, FP_MinusZero)
def is_zero(value):
    return is_plus_zero(value) or is_minus_zero(value) or (is_numeric_value(value) and value == 0)
def is_plus_infty(value):
    return isinstance(value, FP_PlusInfty)
def is_minus_infty(value):
    return isinstance(value, FP_MinusInfty)
def is_infty(value):
    return is_plus_infty(value) or is_minus_infty(value)
def is_number(value):
    """ Only partially valid """
    return not isinstance(value, FP_SpecialValue)
def is_sv_omega(value):
    assert False, "omega is not a special value anymore (but an alias to a numerical value"
    raise NotImplementedError


def is_numeric_value(vx):
    """ predicate testing if vx is a numeric value (including omega) """
    return not(FP_SpecialValue.is_special_value(vx))

from metalibm_core.core.ml_formats import *

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
        FP_PlusOmega(PRECISION),
        FP_MinusOmega(PRECISION),
    ]
    op_map = collections.OrderedDict([
        ("+", operator.__add__),
        ("-", operator.__sub__),
        ("*", operator.__mul__),
        ("<", operator.__lt__),
        ("<=", operator.__le__),
        ("==", operator.__eq__),
    ])
    for op in op_map:
        for lhs in value_list:
            for rhs in value_list:
                print( "{} {} {} = ".format(lhs, op, rhs))
                print("{}".format(op_map[op](lhs, rhs)))


    print(
        "FP_PlusOmega(ML_Binary32) > 2 = {}".format(
            FP_PlusOmega(ML_Binary32) > 2
        )
    )
    print(
        "FP_MinusOmega(ML_Binary32) > 2 = {}".format(
            FP_MinusOmega(ML_Binary32) > 2
        )
    )
    print(
        "FP_MinusOmega(ML_Binary32) > FP_PlusOmega(ML_Binary32) = {}".format(
            FP_MinusOmega(ML_Binary32) > FP_PlusOmega(ML_Binary32)
        )
    )
    TEST_CASE = [
        (FP_PlusOmega(PRECISION), "<", FP_SNaN(PRECISION)),
        (FP_PlusOmega(PRECISION), "<=", FP_SNaN(PRECISION)),
        (FP_PlusInfty(PRECISION), "==", FP_PlusInfty(PRECISION)),
        (FP_PlusInfty(PRECISION), "==", FP_QNaN(PRECISION)),
        (FP_PlusZero(PRECISION), "==", FP_MinusZero(PRECISION)),
    ]
    for lhs, op, rhs in TEST_CASE:
        print( "{} {} {} = ".format(lhs, op, rhs))
        print("{}".format(op_map[op](lhs, rhs)))


