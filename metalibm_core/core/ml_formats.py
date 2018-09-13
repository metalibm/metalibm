# -*- coding: utf-8 -*-

## @package ml_formats
#  Metalibm Formats node precision

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

###############################################################################
# created:          Dec 23rd, 2013
# last-modified:    Mar  8th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import operator
import re

import sollya
from metalibm_core.utility.log_report import Log
from metalibm_core.code_generation.code_constant import *
from metalibm_core.core.special_values import (
    FP_SpecialValue, FP_PlusInfty, FP_MinusInfty, FP_QNaN, FP_SNaN,
    FP_PlusZero, FP_MinusZero, NumericValue
)


S2 = sollya.SollyaObject(2)


## \defgroup ml_formats ml_formats
#  @{

# numerical floating-point constants
ml_nan   = sollya.parse("nan")
ml_infty = sollya.parse("infty")

def get_sollya_from_long(v):
  result = sollya.SollyaObject(0)
  power  = sollya.SollyaObject(1)
  base = S2**16
  while v:
    v, r = divmod(v, int(base))
    result += int(r) * power
    power *= base
  return result



## Ancestor class for Metalibm's format classes
class ML_Format(object):
    """ parent to every Metalibm's format class """
    def __init__(self, name = None, display_format = None):
      self.name = {} if name is None else name
      self.display_format = {} if display_format is None else display_format

    ## return format name
    def get_name(self, language = C_Code):
        if language in self.name:
            return self.name[language]
        else: return self.name[C_Code]

    ## return source code name for the format
    def get_code_name(self, language = C_Code):
        return self.get_name(language = language)

    def get_match_format(self):
        return self

    def get_base_format(self):
        return self
    def get_support_format(self):
        return self

    def get_display_format(self, language = C_Code):
        if language in self.display_format:
            return self.display_format[language]
        elif C_Code in self.display_format:
            return self.display_format[C_Code]
        else:
            return "ERROR_FORMAT"

    ## return the format's bit-size
    def get_bit_size(self):
        """ <abstract> return the bit size of the format (if it exists) """
        print(self) # Exception ML_NotImplemented print
        raise NotImplementedError

    def is_cst_decl_required(self):
        return False

    ## return the C code for args initialization
    def generate_initialization(self, *args, **kwords):
      return None

    ## return the C code for value assignation to var
    # @param var variable assigned
    # @param value value being assigned
    # @param final boolean flag indicating if this assignation is the last in an assignation list
    def generate_assignation(self, var, value, final = True, language = C_Code):
      final_symbol = ";\n" if final else ""
      return "%s = %s" % (var, value)

    # return the format maximal value
    def get_max_value(self):
      raise NotImplementedError

    def is_vector_format(self):
      return False


## class for floating-point exception
class ML_FloatingPointException: pass

## class for type of floating-point exceptions
class ML_FloatingPointException_Type(ML_Format):
  ## dummy placeholder to generate C constant for FP exception (should raise error)
  def get_cst(self, value, language = C_Code):
    return "NONE"
  def is_cst_decl_required(self):
    return False
  def get_match_format(self):
    return self

## ML object for floating-point exception type
ML_FPE_Type = ML_FloatingPointException_Type()

## ML object for floating-point underflow exception
ML_FPE_Underflow    = ML_FloatingPointException()
## ML object for floating-point overflow exception
ML_FPE_Overflow     = ML_FloatingPointException()
## ML object for floatingè-point inexact exception
ML_FPE_Inexact      = ML_FloatingPointException()
## ML object for floating-point invalid exception
ML_FPE_Invalid      = ML_FloatingPointException()
## ML object for floating-point divide by zero exception
ML_FPE_DivideByZero = ML_FloatingPointException()

## format attribute wrapper 
#  extend a base format with custom attributes
class FormatAttributeWrapper(ML_Format):
	def __init__(self, base_format, attribute_list):
		self.base_format = base_format
		self.attribute_list = attribute_list

	def get_base_format(self):
		return self.base_format.get_base_format()
	def get_support_format(self):
		return self.base_format.get_support_format()
	def get_match_format(self):
		return self.base_format.get_match_format()
	def is_vector_format(self):
		return self.base_format.is_vector_format()
	def get_bit_size(self):
		return self.base_format.get_bit_size()
	def get_display_format(self, language = C_Code):
		return self.base_format.get_display_format(language)
	def get_name(self, language = C_Code):
		str_list = self.attribute_list + [self.base_format.get_name(language = language)]
		return " ".join(str_list)
	def __str__(self):
		return self.get_name(C_Code)

## Class of rounding mode type
class ML_FloatingPoint_RoundingMode_Type(ML_Format):
    name_map = {None: "ml_rnd_mode_t", C_Code: "ml_rnd_mode_t", OpenCL_Code: "ml_rnd_mode_t"}
    def get_c_name(self):
        return "ml_rnd_mode_t"

    def get_name(self, language = C_Code):
      return ML_FloatingPoint_RoundingMode_Type.name_map[language]

## Class of floating-point rounding mode
class ML_FloatingPoint_RoundingMode(object):
    pass

## ML type object for rounding modes
ML_FPRM_Type = ML_FloatingPoint_RoundingMode_Type()

## ML object for rounding to nearest mode
ML_RoundToNearest        = ML_FloatingPoint_RoundingMode()
## ML object for rounding toward zero mode
ML_RoundTowardZero       = ML_FloatingPoint_RoundingMode()
## ML object for rouding towards plus infinity mode
ML_RoundTowardPlusInfty  = ML_FloatingPoint_RoundingMode()
## ML object for rounding forwards minus infinity
ML_RoundTowardMinusInfty = ML_FloatingPoint_RoundingMode()
## ML object for current global rounding mode
ML_GlobalRoundMode       = ML_FloatingPoint_RoundingMode()


## Ancestor class for abstract format
class ML_AbstractFormat(ML_Format):
    def __init__(self, c_name):
        ML_Format.__init__(self)
        self.name[C_Code] = c_name

    def __str__(self):
        return self.name[C_Code]

    ## return the gappa constant corresponding to parameter
    #  @param cst_value constant value being translated
    def get_gappa_cst(self, cst_value):
        """ C code for constante cst_value """
        sollya.settings.display = sollya.hexadecimal
        if isinstance(cst_value, int):
            return str(float(cst_value))
        else:
            return str(cst_value)

    def is_cst_decl_required(self):
      return False

## ML object for exact format (no rounding involved)
ML_Exact = ML_AbstractFormat("ML_Exact")




## Ancestor class for instanciated formats
class ML_InstanciatedFormat(ML_Format): pass

## Ancestor class for all Floating-point formats
class ML_FP_Format(ML_Format):
    """ parent to every Metalibm's floating-point class """
    pass
    @staticmethod
    def is_fp_format(precision):
        """ generic predicate to test whether or not precision
            is a floating-point format """
        return isinstance(precision, ML_FP_Format)

## Ancestor class for standard (as defined in IEEE-754) floating-point formats
class ML_Std_FP_Format(ML_FP_Format):
    """ standard floating-point format base class """

    def __init__(self, bit_size, exponent_size, field_size, c_suffix, c_name, ml_support_prefix, c_display_format, sollya_object, union_field_suffix = None):
        ML_Format.__init__(self)
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format


        self.bit_size = bit_size
        self.exponent_size = exponent_size
        self.field_size = field_size
        self.c_suffix = c_suffix
        self.ml_support_prefix = ml_support_prefix
        self.sollya_object = sollya_object
        ## suffix used when selecting format in a support library union
        self.union_field_suffix = union_field_suffix

    def get_ml_support_prefix(self):
        return self.ml_support_prefix
    def get_union_field_suffix(self):
        return self.union_field_suffix

    ## return the sollya object encoding the format precision
    def get_sollya_object(self):
      return self.sollya_object

    ## round the sollya object @p value to the sollya precision
    #  equivalent to @p self
    def round_sollya_object(self, value, round_mode = sollya.RN):
      return sollya.round(value, self.get_sollya_object(), round_mode)

    def __str__(self):
        return self.name[C_Code]

    def get_name(self, language = C_Code):
        return self.name[C_Code]

    def get_bias(self):
        return - 2**(self.get_exponent_size() - 1) + 1

    def get_emax(self):
        return 2**self.get_exponent_size() - 2 + self.get_bias()

    def get_special_value_coding(self, sv, language = C_Code):
        """ Generate integer coding for a special value number
            in self format """
        assert FP_SpecialValue.is_special_value(sv)
        return sv.get_integer_coding()

    ## return the integer coding of @p value
    #  @param value numeric value to be converted
    #  @return value encoding (as an integer number)
    def get_integer_coding(self, value, language = C_Code):
        if FP_SpecialValue.is_special_value(value):
            return self.get_special_value_coding(value, language)
        elif value == ml_infty:
            return self.get_special_value_coding(FP_PlusInfty(self), language)
        elif value == -ml_infty:
            return self.get_special_value_coding(FP_MinusInfty(self), language)
        else:
            pre_value = value
            value = sollya.round(value, self.get_sollya_object(), sollya.RN)
            # FIXME: managing negative zero
            sign = int(1 if value < 0 else 0)
            value = abs(value)
            if value == 0.0:
              Log.report(Log.Warning, "+0.0 forced during get_integer_coding conversion")
              exp_biased = 0
              mant = 0
            else:
              try: 
                exp        = int(sollya.floor(sollya.log2(value)))
              except ValueError as e:
                Log.report(Log.Error, "unable to compute int(sollya.floor(sollya.log2({}))), pre_value={}", value, pre_value, error=e)
              exp_biased = int(exp - self.get_bias())
              if exp < self.get_emin_normal():
                exp_biased = 0
                mant = int((value / S2**self.get_emin_subnormal()))
              else:
                mant = int((value / S2**exp - 1.0) * (S2**self.get_field_size()))
            return mant | (exp_biased << self.get_field_size()) | (sign << (self.get_field_size() + self.get_exponent_size()))

    def get_value_from_integer_coding(self, value, base=10):
        """ Convert a value binary encoded following IEEE-754 standard
            to its floating-point numerical (or special) counterpart """
        value = int(value, base)
        exponent_field = ((value >> self.get_field_size()) & (2**self.get_exponent_size() - 1)) 
        is_subnormal = (exponent_field == 0)
        mantissa = value & (2**self.get_field_size() - 1)
        sign_bit = value >> (self.get_field_size() + self.get_exponent_size())
        if exponent_field == self.get_nanorinf_exp_field():
            if mantissa == 0 and sign_bit:
                return FP_MinusInfty(self)
            elif mantissa == 0 and not(sign_bit):
                return FP_PlusInfty(self)
            else:
                # NaN value
                quiet_bit = mantissa >> (self.get_field_size() - 1)
                if quiet_bit:
                    return FP_QNaN(self)
                else:
                    return FP_SNaN(self)
        elif exponent_field == 0 and mantissa == 0:
            if sign_bit:
                return FP_MinusZero(self)
            else:
                return FP_PlusZero(self)
        else:
            assert exponent_field != self.get_nanorinf_exp_field()
            exponent = exponent_field + self.get_bias() + (1 if is_subnormal else 0)
            sign = -1.0 if sign_bit != 0 else 1.0
            mantissa_value = mantissa
            implicit_digit = 0.0 if is_subnormal else 1.0
            return NumericValue(sign * S2**int(exponent) * (implicit_digit + mantissa_value * S2**-self.get_field_size()))

    # @return<SollyaObject> the format omega value, the maximal normal value
    def get_omega(self):
        return S2**self.get_emax() * (2 - S2**-self.get_field_size())

    # @return<SollyaObject> the format maximal value
    def get_max_value(self):
        return self.get_omega()

    def get_min_normal_value(self):
        """ return the minimal normal number in @p self format """
        return S2**self.get_emin_normal()

    ## return the exponent field corresponding to 
    #  a special value (inf or NaN)
    def get_nanorinf_exp_field(self):
        return S2**self.get_exponent_size() - 1

    ## Return the minimal exponent for a normal number
    def get_emin_normal(self):
        return 1 + self.get_bias()

    ## Return the minimal exponent for a subnormal number
    def get_emin_subnormal(self):
        return 1 - (self.get_field_size()) + self.get_bias()

    ## Return the display (for debug message) associated
    #  to format @p self
    def get_display_format(self, language = C_Code):
        return self.display_format[language]

    def get_bit_size(self):
        """ return the format bit size """
        return self.bit_size

    def get_zero_exponent_value(self):
        return 0

    def get_special_exponent_value(self):
        return 2**self.get_exponent_size() - 1

    def get_exponent_size(self):
        return self.exponent_size

    def get_exponent_interval(self):
        low_bound  = self.get_emin_normal()
        high_bound = self.get_emax()
        return sollya.Interval(low_bound, high_bound)

    ## return the size of the mantissa bitfield (excluding implicit bit(s))
    def get_field_size(self):
        return self.field_size

    ## Return the complete mantissa size (including implicit bit(s))
    def get_mantissa_size(self):
        return self.field_size + 1

    def get_cst(self, cst_value, language = C_Code):
      """Return how a constant of value cst_value should be written in the
      language language for this meta-format.
      """
      if language is C_Code:
        return self.get_c_cst(cst_value)
      elif language is Gappa_Code:
        return self.get_gappa_cst(cst_value)
      else:
        # default case
        return self.get_c_cst(cst_value)

    def get_c_cst(self, cst_value):
        """ C code for constant cst_value """
        if isinstance(cst_value, FP_SpecialValue):
            return cst_value.get_c_cst()
        else:
            sollya.settings.display = sollya.hexadecimal
            if cst_value == sollya.SollyaObject(0):
                conv_result = "0.0" + self.c_suffix
            if cst_value == ml_infty:
                conv_result = "INFINITY"
            elif cst_value == ml_nan:
                conv_result = "NAN"
            elif isinstance(cst_value, int):
                conv_result = str(float(cst_value)) + self.c_suffix
            else:
              if isinstance(cst_value, sollya.SollyaObject):
                conv_result  = str(self.round_sollya_object(cst_value)) + self.c_suffix
              else:
                conv_result  = str(cst_value) + self.c_suffix
            if conv_result == "0f":
                conv_result = "0.0f"
            return conv_result

    def get_precision(self):
        """ return the bit-size of the mantissa """
        return self.get_field_size()

    def get_gappa_cst(self, cst_value):
        """ C code for constante cst_value """
        if isinstance(cst_value, FP_SpecialValue):
            return cst_value.get_gappa_cst()
        else:
            sollya.settings.display = sollya.hexadecimal
            if isinstance(cst_value, int):
                return str(float(cst_value))
            else:
                return str(cst_value)

    def get_integer_format(self):
        int_precision = {
                ML_Binary16: ML_Int16,
                ML_Binary32: ML_Int32,
                ML_Binary64: ML_Int64,
                ML_Binary80: None,
                }
        return int_precision[self]

    def get_unsigned_integer_format(self):
        uint_precision = {
                ML_Binary16: ML_UInt16,
                ML_Binary32: ML_UInt32,
                ML_Binary64: ML_UInt64,
                ML_Binary80: None,
                }
        return uint_precision[self]


def is_std_float(precision):
    return isinstance(precision, ML_Std_FP_Format)

## Generic constructor for Metalibm formats
class ML_FormatConstructor(ML_Format):
    """ Generic constructor for Metalibm formats """
    ## Object constructor 
    #  @param bit_size size of the format (in bits)
    #  @param c_name name of the format in the C language
    #  @param c_display_format string format to display @p self format value
    #  @param get_c_cst function self, value -> Node to generate
    #         constant value associated with @p self format
    def __init__(self, bit_size, c_name, c_display_format, get_c_cst):
        ML_Format.__init__(self)
        self.bit_size = bit_size
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format
        self.get_cst_map = {C_Code: get_c_cst}

    ## generate a constant value with numerical value @p value
    #  in language @p language
    def get_cst(self, value, language = C_Code):
        return self.get_cst_map[language](self, value)

    def __str__(self):
        return self.name[C_Code]

    ## Return the format size (in bits)
    def get_bit_size(self):
        return self.bit_size

## a virtual format is a format which is internal to Metalibm
#  representation and relies on an other non-virtual format
#  for support in generated code
class VirtualFormat(ML_Format):
  def __init__(self,
               base_format = None,
               support_format = None,
               get_cst = lambda self, value, language:
               self.base_format.get_cst(value, language),
               cst_decl_required = False
        ):
    ML_Format.__init__(self)
    self.support_format = support_format
    self.base_format    = base_format
    self.internal_get_cst = get_cst
    # is constant declaration required
    self.cst_decl_required = cst_decl_required

  def get_cst(self, cst_value, language = C_Code):
    return self.internal_get_cst(self, cst_value, language)

  def __str__(self):
    return "{}/{}".format(str(self.base_format), self.support_format)

  ## return name for the format
  def get_name(self, language = C_Code):
    raise NotImplementedError
    return self.base_format.get_name(language = language)

  ## return source code name for the format
  def get_code_name(self, language = C_Code):
    code_name = self.support_format.get_name(language = language)
    return code_name

  def set_support_format(self, _format):
    self.support_format = _format

  def get_match_format(self):
    return self.base_format

  def get_base_format(self):
    return self.base_format

  def get_support_format(self):
    return self.support_format
  def get_signed(self):
    return self.get_base_format().get_signed()

  def get_bit_size(self):
    return self.get_base_format().get_bit_size()
  def is_cst_decl_required(self):
    return self.cst_decl_required

  def is_vector_format(self):
      return False


def get_virtual_cst(prec, value, language):
    """ constant get for virtual format """
    return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language)
    )

## Virtual format with no match forwarding
class VirtualFormatNoForward(VirtualFormat):
    def get_match_format(self):
        return self

class VirtualFormatNoBase(VirtualFormat):
    """ Virtual format class which does not point towards a distinct
        base format """
    def get_match_format(self):
        return self
    def get_base_format(self):
        return self
    def get_vector_format(self):
        return False


## Ancestor to fixed-point format
class ML_Fixed_Format(ML_Format):
    """ parent to every Metalibm's fixed-point class """
    def __init__(self, align = 0):
      ML_Format.__init__(self)
      # self.support_format must be an integer format
      # used to contain the fixed-point value

      # offset between the support LSB and the actual value LSB
      self.support_right_align = align

    def get_match_format(self):
      return self
    def get_base_format(self):
      return self

    def get_name(self, language = C_Code):
      return ML_Format.get_name(self, language = language)
    def get_code_name(self, language = C_Code):
      return ML_Format.get_code_name(self, language = language)

    def set_support_right_align(self, align):
      self.support_right_align = align

    def get_support_right_align(self):
      return self.support_right_align


class ML_Base_FixedPoint_Format(ML_Fixed_Format, VirtualFormatNoBase):
    """ base class for standard integer format """
    def __init__(self, integer_size, frac_size, signed = True, support_format = None, align = 0):
        """ standard fixed-point format object initialization function """
        ML_Fixed_Format.__init__(self, align)
        VirtualFormatNoBase.__init__(self, support_format = support_format)

        self.integer_size = integer_size
        self.frac_size = frac_size
        self.signed = signed


    ## @return size (in bits) of the integer part of @p self formats
    #          may be negative to indicate a right shift of the fractionnal
    #          part
    def get_integer_size(self):
        return self.integer_size

    def get_c_bit_size(self):
        return self.c_bit_size


    @staticmethod
    def match(format_str):
        """ returns None if format_str does not match the class pattern
            or a re.match if it does """
        return re.match("(?P<name>F[US])(?P<integer>-?[\d]+)\.(?P<frac>-?[\d]+)",format_str)

    ## @return size (in bits) of the fractional part of
    #          @p self formats
    #          may be negative to indicate a left shift of the integer part
    def get_frac_size(self):
        return self.frac_size

    def get_precision(self):
        """ return the number of digits after the point """
        return self.frac_size

    ## @return boolean signed/unsigned property
    def get_signed(self):
        return self.signed

    ## return the maximal possible value for the format
    def get_max_value(self):
        offset = -1 if self.get_signed() else 0
        max_code_exp = self.get_integer_size() + self.get_frac_size()
        code_value = S2**(max_code_exp + offset) - 1
        return code_value * S2**-self.get_frac_size()

    ## @p round the numerical value @p value to
    #  @p self fixed-point format while applying
    #  @p round_mode to determine rounding direction
    #  @return rounded value (SollyaObject)
    def round_sollya_object(self, value, round_mode=sollya.RN):
        rnd_function = {
            sollya.RN: sollya.nearestint,
            sollya.RD: sollya.floor,
            sollya.RU: sollya.ceil,
            sollya.RZ: lambda x: sollya.floor(x) if x > 0 \
                       else sollya.ceil(x)
        }[round_mode]
        scale_factor = S2**self.get_frac_size()
        return rnd_function(scale_factor * value) / scale_factor

    ## return the minimal possible value for the format
    def get_min_value(self):
        if not self.get_signed():
            return 0
        else:
            max_code_exp = self.get_integer_size() + self.get_frac_size()
            code_value = S2**(max_code_exp - 1) 
            return - (code_value * S2**-self.get_frac_size())

    ## if value exceeds formats then
    def truncate(self, value):
        descaled_value = value * S2**self.get_frac_size()
        masked_value = int(descaled_value) & int(S2**self.get_bit_size() - 1)
        scaled_value = masked_value * S2**-self.get_frac_size()
        if scaled_value > self.get_max_value():
            scaled_value -= S2**self.get_integer_size() 
        return scaled_value

    def __str__(self):
        if self.signed:
          return "FS%d.%d" % (self.integer_size, self.frac_size)
        else:
          return "FU%d.%d" % (self.integer_size, self.frac_size)

    def get_bit_size(self):
        return self.integer_size + self.frac_size

    def get_cst(self, cst_value, language = C_Code):
        if language is C_Code:
            return self.get_c_cst(cst_value)
        elif language is Gappa_Code:
            return self.get_gappa_cst(cst_value)
        else:
            return self.get_c_cst(cst_value)

    def saturate(self, value):
        """ Saturate value to stay within:
            [self.get_min_value(), self.get_max_value()] """
        if value > self.get_max_value():
            return self.get_max_value()
        elif value < self.get_min_value():
            return self.get_min_value()
        else:
            return value

    def get_integer_coding(self, value, language=C_Code):
      if value > self.get_max_value() or  value < self.get_min_value():
        Log.report(Log.Error, "value {} is out of format {} range [{}; {}]", value, self, self.get_min_value(), self.get_max_value())
      if value < 0:
        if not self.signed:
            Log.report(Log.Error, "negative value encountered {} while converting for an unsigned precision: {}".format(value, self))
        encoded_value = (~int(abs(value) * S2**self.frac_size) + 1) % 2**self.get_bit_size()
        return encoded_value
      else:
        encoded_value = int(value * S2**self.frac_size)
        return encoded_value

    def get_c_cst(self, cst_value):
        """ C-language constant generation """
        try:
          encoded_value = int(cst_value * S2**self.frac_size)
        except (ValueError, TypeError) as e:
          print(e, cst_value, self.frac_size)
          Log.report(Log.Error, "Error during constant conversion to sollya object")
          
        return ("" if self.signed else "U") + "INT" + str(self.c_bit_size) + "_C(" + str(encoded_value) + ")"

    def get_gappa_cst(self, cst_value):
        """ Gappa-language constant generation """
        return str(cst_value)


class ML_Base_SW_FixedPoint_Format(ML_Base_FixedPoint_Format):
    """ Base Fixed-Point format for software implementation,
        try to infer the required size of C-format to support
        this format """
    def __init__(self, integer_size, frac_size, signed=True, support_format=None, align=0):
        ML_Base_FixedPoint_Format.__init__(
            self,
            integer_size,
            frac_size,
            signed,
            support_format
        )
        # guess the minimal bit_size required in the c repesentation
        bit_size = integer_size + frac_size
        if bit_size < 1 or bit_size > 128:
            Log.report(Log.Warning, "unsupported bit_size {} in ML_Base_SW_FixedPoint_Format".format(bit_size))
        else:
            possible_c_bit_sizes = [8, 16, 32, 64, 128]
            self.c_bit_size = next(n for n in possible_c_bit_sizes if n >= bit_size)
            c_name = ("" if self.signed else "u") + "int" + str(self.c_bit_size) + "_t"
            c_display_format = "%\"PRIx" + str(self.c_bit_size) + "\""
            self.name[C_Code] = c_name
            self.display_format[C_Code] = c_display_format


## Ancestor to standard (meaning integers)  fixed-point format
class ML_Standard_FixedPoint_Format(ML_Base_SW_FixedPoint_Format):
  def __init__(self, integer_size, frac_size, signed = True):
    ML_Base_SW_FixedPoint_Format.__init__(self, integer_size, frac_size, signed = signed, support_format = self, align = 0)

  ## use 0 as the LSB weight to round in sollya
  def get_sollya_object(self):
    return sollya.SollyaObject(0)

    ## round the sollya object @p value to the sollya precision
    #  equivalent to @p self
  def round_sollya_object(self, value, round_mode = sollya.RN):
    # TBD: support other rounding mode
    return sollya.nearestint(value)

  def __repr__(self):
      return self.name[C_Code]

  def __str__(self):
    return self.name[C_Code]

class ML_Custom_FixedPoint_Format(ML_Base_SW_FixedPoint_Format):
    """ Custom fixed-point format class """
    def __eq__(self, other):
        """ equality predicate for custom fixed-point format object """
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

    def __ne__(self, other):
        """ unequality predicate for custom fixed-point format object """
        return not (self == other)


    @staticmethod
    def parse_from_match(format_match):
        """ Parse the description of a class format and generates
            the format object """
        assert not format_match is None
        name = format_match.group("name")
        int_size = int(format_match.group("integer"))
        frac_size = int(format_match.group("frac"))
        is_signed = (name == "FS")
        return ML_Custom_FixedPoint_Format(int_size, frac_size, signed=is_signed)

    ## parse a string describing a ML_Custom_FixedPoint_Format object
    #  @param format_str string describing the format object
    #  @return the format instance converted from the string
    @staticmethod
    def parse_from_string(format_str):
        format_match = ML_Custom_FixedPoint_Format.match(format_str)
        return ML_Custom_FixedPoint_Format.parse_from_match(format_match)


# Standard binary floating-point format declarations
## IEEE binary32 (fp32) single precision floating-point format
ML_Binary32 = ML_Std_FP_Format(32, 8, 23, "f", "float", "fp32", "%a", sollya.binary32, union_field_suffix = "f")
## IEEE binary64 (fp64) double precision floating-point format
ML_Binary64 = ML_Std_FP_Format(64, 11, 52, "", "double", "fp64", "%la", sollya.binary64, union_field_suffix = "d")
ML_Binary80 = ML_Std_FP_Format(80, 15, 64, "L", "long double", "fp80", "%la", sollya.binary80)
## IEEE binary16 (fp16) half precision floating-point format
ML_Binary16 = ML_Std_FP_Format(16, 5, 10, "__ERROR__", "half", "fp16", "%a", sollya.binary16)


# Standard integer format declarations
ML_Int8    = ML_Standard_FixedPoint_Format(8, 0, True)
ML_UInt8   = ML_Standard_FixedPoint_Format(8, 0, False)

ML_Int16    = ML_Standard_FixedPoint_Format(16, 0, True)
ML_UInt16   = ML_Standard_FixedPoint_Format(16, 0, False)

ML_Int32    = ML_Standard_FixedPoint_Format(32, 0, True)
ML_UInt32   = ML_Standard_FixedPoint_Format(32, 0, False)

ML_Int64    = ML_Standard_FixedPoint_Format(64, 0, True)
ML_UInt64   = ML_Standard_FixedPoint_Format(64, 0, False)

ML_Int128    = ML_Standard_FixedPoint_Format(128, 0, True)
ML_UInt128   = ML_Standard_FixedPoint_Format(128, 0, False)

ML_Int256    = ML_Standard_FixedPoint_Format(256, 0, True)

def bool_get_c_cst(self, cst_value):
  if cst_value:
    return "ML_TRUE"
  else:
    return "ML_FALSE"

class ML_Bool_Format(object):
    """ abstract Boolean format """
    pass


class ML_BoolClass(ML_FormatConstructor, ML_Bool_Format):
  def __str__(self):
    return "ML_Bool"

ML_Bool      = ML_BoolClass(32, "int", "%d", bool_get_c_cst)

## virtual parent to string formats
class ML_String_Format(ML_Format):
    """ abstract String format """
    pass
class ML_StringClass(ML_String_Format):
    """ Metalibm character string class """
    def __init__(self, c_name, c_display_format, get_c_cst):
        ML_Format.__init__(self)
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format
        self.get_cst_map = {C_Code: get_c_cst}

    def get_cst(self, value, language = C_Code):
        return self.get_cst_map[language](self, value)

    def __str__(self):
        return "ML_String"

## Metalibm string format
ML_String = ML_StringClass("char*", "%s", lambda self, s: "\"{}\"".format(s)) 

## Predicate checking if @p precision is a standard integer format
def is_std_integer_format(precision):
	return isinstance(precision, ML_Standard_FixedPoint_Format) or \
           isinstance(precision.get_base_format(), ML_Standard_FixedPoint_Format) and \
           not precision.is_vector_format()
  #return precision in [ ML_Int8, ML_UInt8, ML_Int16, ML_UInt16,
  #                      ML_Int32, ML_UInt32, ML_Int64, ML_UInt64,
  #                      ML_Int128, ML_UInt128 ]

def is_std_signed_integer_format(precision):
	return is_std_integer_format(precision) and \
	       (precision.get_base_format().get_signed() or \
            precision.get_signed())
  #return precision in [ ML_Int8, ML_Int16, ML_Int32, ML_Int64, ML_Int128 ]

def is_std_unsigned_integer_format(precision):
	return is_std_integer_format(precision) and \
	       ((not precision.get_base_format().get_signed()) or \
            (not precision.get_signed()))
  #return precision in [ ML_UInt8, ML_UInt16, ML_UInt32, ML_UInt64, ML_UInt128 ]

def is_table_index_format(precision):
    """ Predicate to test if <precision> can be used as table index format """
    return isinstance(precision, ML_Standard_FixedPoint_Format) or \
           isinstance(precision.get_match_format(), ML_Standard_FixedPoint_Format) and \
           not precision.is_vector_format()

def get_std_integer_support_format(precision):
  """ return the ML's integer format to contains
      the fixed-point format precision """
  assert(isinstance(precision, ML_Fixed_Format))
  format_map = {
    # signed
    True: {
      8:  ML_Int8,
      16: ML_Int16,
      32: ML_Int32,
      64: ML_Int64,
      128: ML_Int128,
    },
    # unsigned
    False: {
      8: ML_UInt8,
      16: ML_UInt16,
      32: ML_UInt32,
      64: ML_UInt64,
      128: ML_UInt128,
    },
  }
  return format_map[precision.get_signed()][precision.get_c_bit_size()]


## functor for abstract format construction
def AbstractFormat_Builder(name, inheritance):
    field_map = {
        "name": name,
        "__str__": lambda self: self.name[C_Code],
    }
    return type(name, (ML_AbstractFormat,) + inheritance, field_map)

class ML_IntegerClass(ML_AbstractFormat, ML_Fixed_Format): pass
class ML_FloatClass(ML_AbstractFormat, ML_FP_Format): pass
class ML_AbstractBoolClass(ML_AbstractFormat, ML_Bool_Format): pass

# abstract formats singleton
ML_Integer          = ML_IntegerClass("ML_Integer") #AbstractFormat_Builder("ML_Integer",  (ML_Fixed_Format,))("ML_Integer")
ML_Float            = ML_FloatClass("ML_Float") #AbstractFormat_Builder("ML_Float",    (ML_FP_Format,))("ML_Float")
ML_AbstractBool     = ML_AbstractBoolClass("ML_AbstractBool")#AbstractFormat_Builder("MLAbstractBool",     (ML_Bool_Format,))("ML_AbstractBool")



###############################################################################
#                     COMPOUND FORMAT
###############################################################################

class ML_Compound_Format(ML_Format):
    def __init__(self, c_name, c_field_list, field_format_list, ml_support_prefix, c_display_format, sollya_object):
        ML_Format.__init__(self)
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format

        self.ml_support_prefix = ml_support_prefix
        self.sollya_object = sollya_object
        self.c_field_list = c_field_list
        self.field_format_list = field_format_list

    def __str__(self):
        return self.name[C_Code]

    ## return the sollya object encoding the format precision
    def get_sollya_object(self):
      return self.sollya_object

    ## forces constant declaration during code generation
    def is_cst_decl_required(self):
        return True

    def get_cst(self, cst_value, language = C_Code):
        tmp_cst = cst_value
        field_str_list = []
        for field_name, field_format in zip(self.c_field_list, self.field_format_list):
            # FIXME, round is only valid for double_double or triple_double stype format
            field_value = sollya.round(tmp_cst, field_format.sollya_object, RN)
            tmp_cst = cst_value - field_value
            field_str_list.append(".%s = %s" % (field_name, field_format.get_c_cst(field_value)))
        return "{%s}" % (", ".join(field_str_list))

    def get_gappa_cst(self, cst_value):
        """ Constant generation in Gappa-language """
        return str(cst_value)



class ML_Compound_FP_Format(ML_Compound_Format, ML_FP_Format):
  pass
class ML_Compound_Integer_Format(ML_Compound_Format, ML_Fixed_Format):
  pass

# compound binary floating-point format declaration
ML_DoubleDouble = ML_Compound_FP_Format("ml_dd_t", ["hi", "lo"],
                                        [ML_Binary64, ML_Binary64],
                                        "", "",
                                        sollya.doubledouble)
ML_TripleDouble = ML_Compound_FP_Format("ml_td_t", ["hi", "me", "lo"],
                                        [ML_Binary64, ML_Binary64,
                                            ML_Binary64],
                                        "", "",
                                        sollya.tripledouble)
ML_SingleSingle = ML_Compound_FP_Format("ml_ds_t", ["hi", "lo"],
                                        [ML_Binary32, ML_Binary32],
                                        "", "",
                                        2*ML_Binary32.get_mantissa_size() + 1)
###############################################################################
#                     VECTOR FORMAT
###############################################################################

## common ancestor to every vector format
class ML_VectorFormat(ML_Format):
  def __init__(self, scalar_format, vector_size, c_name):
    ML_Format.__init__(self, name = {C_Code: c_name})
    self.scalar_format = scalar_format
    self.vector_size   = vector_size

  def is_vector_format(self):
    return True

  def get_bit_size(self):
    return self.vector_size * self.scalar_format.get_bit_size()

  def __str__(self):
	  return self.get_code_name(language = C_Code)

  def get_scalar_format(self):
    return self.scalar_format
  def set_scalar_format(self, new_scalar_format):
    self.scalar_format = new_scalar_format

  def get_vector_size(self):
    return self.vector_size
  def set_vector_size(self, new_vector_size):
    self.vector_size = new_vector_size

  def get_name(self, language = C_Code):
    try:
      return ML_Format.get_name(self, language)
    except KeyError:
      return self.get_scalar_format().get_name(language)

## Generic class for Metalibm support library vector format
class ML_CompoundVectorFormat(ML_VectorFormat, ML_Compound_Format):
  def __init__(self, c_format_name, opencl_format_name, vector_size, scalar_format, sollya_precision = None, cst_callback = None):
    ML_VectorFormat.__init__(self, scalar_format, vector_size, c_format_name)
    ML_Compound_Format.__init__(self, c_format_name, ["_[%d]" % i for i in range(vector_size)], [scalar_format for i in range(vector_size)], "", "", sollya_precision)
    # registering OpenCL-C format name
    self.name[OpenCL_Code] = opencl_format_name
    self.cst_callback = cst_callback

  def get_cst_default(self, cst_value, language = C_Code):
    elt_value_list = [self.scalar_format.get_cst(cst_value[i], language = language) for i in range(self.vector_size)]
    if language is C_Code:
      return "{._ = {%s}}" % (", ".join(elt_value_list))
    elif language is OpenCL_Code:
      return "(%s)(%s)" % (self.get_name(language = OpenCL_Code), (", ".join(elt_value_list)))
    else:
      Log.report(Log.Error, "unsupported language in ML_CompoundVectorFormat.get_cst: %s" % (language))
      
  def get_cst(self, cst_value, language = C_Code):
    if self.cst_callback is None:
      return self.get_cst_default(cst_value, language)
    else:
      return self.cst_callback(self, cst_value, language)
  

class ML_IntegerVectorFormat(ML_CompoundVectorFormat, ML_Fixed_Format):
  pass

class ML_FloatingPointVectorFormat(ML_CompoundVectorFormat, ML_FP_Format):
  pass

## helper function to generate a vector format
#  @param format_name string name of the result format
#  @param vector_size integer number of element in the vector
#  @param scalar_format ML_Format object, format of a vector's element
#  @param sollya_precision pythonsollya object, sollya precision to be used for computation
#  @param compound_constructor ML_Compound_Format child class used to build the result format
#  @param cst_callback function (self, value, language) -> str, used to generate constant value code
def vector_format_builder(c_format_name, opencl_format_name, vector_size,
                          scalar_format, sollya_precision=None,
                          compound_constructor=ML_FloatingPointVectorFormat, cst_callback=None):
  return compound_constructor(c_format_name, opencl_format_name, vector_size,
                              scalar_format, sollya_precision, cst_callback)

v2float32 = vector_format_builder("ml_float2_t", "float2", 2, ML_Binary32)
v3float32 = vector_format_builder("ml_float3_t", "float3", 3, ML_Binary32)
v4float32 = vector_format_builder("ml_float4_t", "float4", 4, ML_Binary32)
v8float32 = vector_format_builder("ml_float8_t", "float8", 8, ML_Binary32)

v2float64 = vector_format_builder("ml_double2_t", "double2", 2, ML_Binary64)
v3float64 = vector_format_builder("ml_double3_t", "double3", 3, ML_Binary64)
v4float64 = vector_format_builder("ml_double4_t", "double4", 4, ML_Binary64)
v8float64 = vector_format_builder("ml_double8_t", "double8", 8, ML_Binary64)

v2bool  = vector_format_builder("ml_bool2_t", "int2", 2, ML_Bool, compound_constructor = ML_IntegerVectorFormat)
v3bool  = vector_format_builder("ml_bool3_t", "int3", 3, ML_Bool, compound_constructor = ML_IntegerVectorFormat)
v4bool  = vector_format_builder("ml_bool4_t", "int4", 4, ML_Bool, compound_constructor = ML_IntegerVectorFormat)
v8bool  = vector_format_builder("ml_bool8_t", "int8", 8, ML_Bool, compound_constructor = ML_IntegerVectorFormat)

v2int32  = vector_format_builder("ml_int2_t", "int2", 2, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v3int32  = vector_format_builder("ml_int3_t", "int3", 3, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v4int32  = vector_format_builder("ml_int4_t", "int4", 4, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v8int32  = vector_format_builder("ml_int8_t", "int8", 8, ML_Int32, compound_constructor = ML_IntegerVectorFormat)

v2uint32 = vector_format_builder("ml_uint2_t", "uint2", 2, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v3uint32 = vector_format_builder("ml_uint3_t", "uint3", 3, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v4uint32 = vector_format_builder("ml_uint4_t", "uint4", 4, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v8uint32 = vector_format_builder("ml_uint8_t", "uint8", 8, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)

v2int64  = vector_format_builder("ml_long2_t", "long2", 2, ML_Int64, compound_constructor = ML_IntegerVectorFormat)
v3int64  = vector_format_builder("ml_long3_t", "long3", 3, ML_Int64, compound_constructor = ML_IntegerVectorFormat)
v4int64  = vector_format_builder("ml_long4_t", "long4", 4, ML_Int64, compound_constructor = ML_IntegerVectorFormat)
v8int64  = vector_format_builder("ml_long8_t", "long8", 8, ML_Int64, compound_constructor = ML_IntegerVectorFormat)

v2uint64 = vector_format_builder("ml_ulong2_t", "ulong2", 2, ML_UInt64, compound_constructor = ML_IntegerVectorFormat)
v3uint64 = vector_format_builder("ml_ulong3_t", "ulong3", 3, ML_UInt64, compound_constructor = ML_IntegerVectorFormat)
v4uint64 = vector_format_builder("ml_ulong4_t", "ulong4", 4, ML_UInt64, compound_constructor = ML_IntegerVectorFormat)
v8uint64 = vector_format_builder("ml_ulong8_t", "ulong8", 8, ML_UInt64, compound_constructor = ML_IntegerVectorFormat)


###############################################################################
#                         GENERIC, NON NUMERICAL FORMATS
###############################################################################

ML_Void = ML_FormatConstructor(0, "void", "ERROR", lambda _: None)


class FP_Context(object):
    """ Floating-Point context """
    def __init__(self, rounding_mode = ML_GlobalRoundMode, silent = None):
        self.rounding_mode       = rounding_mode
        self.init_ev_value       = None
        self.init_rnd_mode_value = None
        self.silent              = silent

    def get_rounding_mode(self):
        return self.rounding_mode

    def get_silent(self):
        return self.silent

class FunctionFormat(object):
    """ format for function object """
    pass


def merge_abstract_format(*args):
    """ return the most generic abstract format
        to unify args formats """
    has_float = False
    has_integer = False
    has_bool = False
    for arg_type in args:
        if isinstance(arg_type, ML_FP_Format): has_float = True
        if isinstance(arg_type, ML_Fixed_Format): has_integer = True
        if isinstance(arg_type, ML_Bool_Format): has_bool = True

    if has_float: return ML_Float
    if has_integer: return ML_Integer
    if has_bool: return ML_AbstractBool
    else:
        print([str(arg) for arg in args])
        Log.report(Log.Error, "unknown formats while merging abstract format tuple")

## @}
# end of metalibm's Doxygen ml_formats group

