# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Dec 23rd, 2013
# last-modified:    Oct  6th, 2015
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import sollya
from ..utility.log_report import Log
from ..code_generation.code_constant import *
import re


S2 = sollya.SollyaObject(2)

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

## class for floating-point exception
class ML_FloatingPointException: pass

## class for type of floating-point exceptions
class ML_FloatingPointException_Type(object):
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
        else:
            return self.display_format[C_Code]

    ## return the format's bit-size
    def get_bit_size(self):
        """ <abstract> return the bit size of the format (if it exists) """
        print self # Exception ML_NotImplemented print
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

## ML object for exact format (no rounding involved)
ML_Exact = ML_AbstractFormat("ML_Exact")


## functor for abstract format construction
def AbstractFormat_Builder(name, inheritance):
    field_map = {
        "name": name,
        "__str__": lambda self: self.name[C_Code],
    }
    return type(name, (ML_AbstractFormat,) + inheritance, field_map)


## Ancestor class for instanciated formats
class ML_InstanciatedFormat(ML_Format): pass

## Ancestor class for all Floating-point formats
class ML_FP_Format(ML_Format):
    """ parent to every Metalibm's floating-point class """
    pass

## Ancestor class for standard (as defined in IEEE-754) floating-point formats
class ML_Std_FP_Format(ML_FP_Format):
    """ standard floating-point format base class """

    def __init__(self, bit_size, exponent_size, field_size, c_suffix, c_name, ml_support_prefix, c_display_format, sollya_object):
        ML_Format.__init__(self)
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format


        self.bit_size = bit_size
        self.exponent_size = exponent_size
        self.field_size = field_size
        self.c_suffix = c_suffix
        self.ml_support_prefix = ml_support_prefix
        self.sollya_object = sollya_object

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

    ## return the integer coding of @p value
    #  @param value numeric value to be converted
    #  @return value encoding (as an integer number)
    def get_integer_coding(self, value, language = C_Code):
        # FIXME: manage subnormal and special values
        value = sollya.round(value, self.get_sollya_object(), sollya.RN)
        # FIXME: managing negative zero
        sign = int(1 if value < 0 else 0)
        value = abs(value)
        exp   = int(sollya.floor(sollya.log2(value)))
        exp_biased = int(exp - self.get_bias())
        mant = int((value / S2**exp - 1.0) / (S2**-self.get_field_size()))
        return mant | (exp_biased << self.get_field_size()) | (sign << (self.get_field_size() + self.get_exponent_size()))

    def get_value_from_integer_coding(self, value, base = 10):
      value = int(value, base)
      mantissa = value & (2**self.get_field_size() - 1)
      exponent = ((value >> self.get_field_size()) & (2**self.get_exponent_size() - 1)) + self.get_bias()
      sign_bit = value >> (self.get_field_size() + self.get_exponent_size())
      sign = -1.0 if sign_bit != 0 else 1.0
      mantissa_value = mantissa
      return sign * S2**int(exponent) * (1.0 + mantissa_value * S2**-self.get_field_size())

    # @return<SollyaObject> the format omega value, the maximal normal value
    def get_omega(self):
        return S2**self.get_emax() * (2 - S2**-self.get_field_size())

    # @return<SollyaObject> the format maximal value
    def get_max_value(self):
        return self.get_omega()

    def get_emin_normal(self):
        return 1 + self.get_bias()

    def get_emin_subnormal(self):
        return 1 - (self.get_field_size() + 1) + self.get_bias()

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

    def get_field_size(self):
        return self.field_size

    ## Return the complete mantissa size
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


class ML_FormatConstructor(ML_Format):
    def __init__(self, bit_size, c_name, c_display_format, get_c_cst):
        ML_Format.__init__(self)
        self.bit_size = bit_size
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format
        self.get_cst_map = {C_Code: get_c_cst}

    def get_cst(self, value, language = C_Code):
        return self.get_cst_map[language](self, value)

    def __str__(self):
        return self.name[C_Code]

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
               self.base_format.get_cst(value, language)
        ):
    ML_Format.__init__(self)
    self.support_format = support_format
    self.base_format    = base_format
    self.internal_get_cst = get_cst

  def get_cst(self, cst_value, language = C_Code):
    return self.internal_get_cst(self, cst_value, language)

  ## return name for the format
  def get_name(self, language = C_Code):
    return self.base_format.get_name(language = language)

  ## return source code name for the format
  def get_code_name(self, language = C_Code):
    return self.support_format.get_name(language = language)

  def set_support_format(self, _format):
    self.support_format = _format

  def get_match_format(self):
    return self.base_format

  def get_base_format(self):
    return self.base_format

  def get_support_format(self):
    return self.support_format


## Ancestor to fixed-point format
class ML_Fixed_Format(VirtualFormat):
    """ parent to every Metalibm's fixed-point class """
    def __init__(self, support_format = None, align = 0):
      VirtualFormat.__init__(self, support_format = support_format)
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


class ML_Base_FixedPoint_Format(ML_Fixed_Format):
    """ base class for standard integer format """
    def __init__(self, integer_size, frac_size, signed = True, support_format = None, align = 0):
        """ standard fixed-point format object initialization function """
        ML_Fixed_Format.__init__(self, support_format, align)

        self.integer_size = integer_size
        self.frac_size = frac_size
        self.signed = signed

        # guess the minimal bit_size required in the c repesentation
        bit_size = integer_size + frac_size
        if bit_size < 1 or bit_size > 128:
            raise ValueError("integer_size+frac_size must be between 1 and 128 (is "+str(bit_size)+")")
        possible_c_bit_sizes = [8, 16, 32, 64, 128]
        self.c_bit_size = next(n for n in possible_c_bit_sizes if n >= bit_size)
        c_name = ("" if self.signed else "u") + "int" + str(self.c_bit_size) + "_t"
        c_display_format = "%\"PRIx" + str(self.c_bit_size) + "\""
        self.name[C_Code] = c_name
        self.display_format[C_Code] = c_display_format

    def get_integer_size(self):
        return self.integer_size

    def get_c_bit_size(self):
        return self.c_bit_size

    def get_frac_size(self):
        return self.frac_size

    def get_precision(self):
        """ return the number of digits after the point """
        return self.frac_size

    def get_signed(self):
        return self.signed

    def __str__(self):
        if self.frac_size == 0:
          return self.name[C_Code]
        elif self.signed:
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

    def get_c_cst(self, cst_value):
        """ C-language constant generation """
        try:
          encoded_value = int(cst_value * sollya.S2**self.frac_size)
        except ValueError as e:
          print e, cst_value, self.frac_size
          Log.report(Log.Error, "Error during constant conversion to sollya object")
          
        return ("" if self.signed else "U") + "INT" + str(self.c_bit_size) + "_C(" + str(encoded_value) + ")"

    def get_gappa_cst(self, cst_value):
        """ Gappa-language constant generation """
        return str(cst_value)



## Ancestor to standard (meaning integers)  fixed-point format
class ML_Standard_FixedPoint_Format(ML_Base_FixedPoint_Format):
  def __init__(self, integer_size, frac_size, signed = True):
    ML_Base_FixedPoint_Format.__init__(self, integer_size, frac_size, signed = signed, support_format = self, align = 0)

  ## use 0 as the LSB weight to round in sollya
  def get_sollya_object(self):
    return sollya.SollyaObject(0)

    ## round the sollya object @p value to the sollya precision
    #  equivalent to @p self
  def round_sollya_object(self, value, round_mode = sollya.RN):
    # TBD: support other rounding mode
    return sollya.nearestint(value)

class ML_Custom_FixedPoint_Format(ML_Base_FixedPoint_Format):
    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not (self == other)

    ## parse a string describing a ML_Custom_FixedPoint_Format object
    #  @param format_str string describing the format object
    #  @return the format instance converted from the string
    @staticmethod
    def parse_from_string(format_str):
      format_match = re.match("(?P<name>F[US])\((?P<integer>-?[\d]+),(?P<frac>-?[\d]+)\)",format_str)
      if format_match is None:
        return None
      else:
        name = format_match.group("name")
        signed = (name == "FS")
        if not name in ["FS", "FU"]:
          return None
        return ML_Custom_FixedPoint_Format(int(format_match.group("integer")), int(format_match.group("frac")), signed = signed)

class ML_Bool_Format(object):
    """ abstract Boolean format """
    pass

# Standard binary floating-point format declarations
ML_Binary32 = ML_Std_FP_Format(32, 8, 23, "f", "float", "fp32", "%a", sollya.binary32)
ML_Binary64 = ML_Std_FP_Format(64, 11, 52, "", "double", "fp64", "%la", sollya.binary64)
ML_Binary80 = ML_Std_FP_Format(80, 15, 64, "L", "long double", "fp80", "%la", sollya.binary80)
# Half precision format
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


def bool_get_c_cst(self, cst_value):
  if cst_value:
    return "ML_TRUE"
  else:
    return "ML_FALSE"

class ML_BoolClass(ML_FormatConstructor, ML_Bool_Format):
  def __str__(self):
    return "ML_Bool"

ML_Bool      = ML_BoolClass(32, "int", "%d", bool_get_c_cst)


def is_std_integer_format(precision):
  return precision in [ ML_Int8, ML_UInt8, ML_Int16, ML_UInt16,
                        ML_Int32, ML_UInt32, ML_Int64, ML_UInt64,
                        ML_Int128, ML_UInt128 ]

def is_std_signed_integer_format(precision):
  return precision in [ ML_Int8, ML_Int16, ML_Int32, ML_Int64, ML_Int128 ]

def is_std_unsigned_integer_format(precision):
  return precision in [ ML_UInt8, ML_UInt16, ML_UInt32, ML_UInt64, ML_UInt128 ]

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



# abstract formats
ML_Integer          = AbstractFormat_Builder("ML_Integer",  (ML_Fixed_Format,))("ML_Integer")
ML_Float            = AbstractFormat_Builder("ML_Float",    (ML_FP_Format,))("ML_Float")
ML_AbstractBool     = AbstractFormat_Builder("MLAbstractBool",     (ML_Bool_Format,))("ML_AbstractBool")



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



class ML_Compound_FP_Format(ML_Compound_Format, ML_FP_Format):
  pass
class ML_Compound_Integer_Format(ML_Compound_Format, ML_Fixed_Format):
  pass

# compound binary floating-point format declaration
ML_DoubleDouble = ML_Compound_FP_Format("ml_dd_t", ["hi", "lo"], [ML_Binary64, ML_Binary64], "", "", sollya.doubledouble)
ML_TripleDouble = ML_Compound_FP_Format("ml_td_t", ["hi", "me", "lo"], [ML_Binary64, ML_Binary64, ML_Binary64], "", "", sollya.tripledouble)

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
    return "VEC_%s[%d]" % (self.scalar_format, self.vector_size)

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
  def __init__(self, c_format_name, opencl_format_name, vector_size, scalar_format, sollya_precision = None):
    ML_VectorFormat.__init__(self, scalar_format, vector_size, c_format_name)
    ML_Compound_Format.__init__(self, c_format_name, ["_[%d]" % i for i in xrange(vector_size)], [scalar_format for i in xrange(vector_size)], "", "", sollya_precision)
    # registering OpenCL-C format name
    self.name[OpenCL_Code] = opencl_format_name


  def get_cst(self, cst_value, language = C_Code):
    elt_value_list = [self.scalar_format.get_cst(cst_value[i], language = language) for i in xrange(self.vector_size)]
    if language is C_Code:
      return "{._ = {%s}}" % (", ".join(elt_value_list))
    elif language is OpenCL_Code:
      return "(%s)(%s)" % (self.get_name(language = OpenCL_Code), (", ".join(elt_value_list)))
    else:
      Log.report(Log.Error, "unsupported language in ML_CompoundVectorFormat.get_cst: %s" % (language))


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
def vector_format_builder(c_format_name, opencl_format_name, vector_size,
                          scalar_format, sollya_precision = None,
                          compound_constructor = ML_FloatingPointVectorFormat):
  return compound_constructor(c_format_name, opencl_format_name, vector_size,
                              scalar_format, sollya_precision)

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

v2int32  = vector_format_builder("ml_int2_t", "int2", 2,  ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v3int32  = vector_format_builder("ml_int3_t", "int3", 3,  ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v4int32  = vector_format_builder("ml_int4_t", "int4", 4, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
v8int32  = vector_format_builder("ml_int8_t", "int8", 8, ML_Int32, compound_constructor = ML_IntegerVectorFormat)

v2uint32 = vector_format_builder("ml_uint2_t", "uint2", 2, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v3uint32 = vector_format_builder("ml_uint3_t", "uint3", 3, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v4uint32 = vector_format_builder("ml_uint4_t", "uint4", 4, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
v8uint32 = vector_format_builder("ml_uint8_t", "uint8", 8, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)


###############################################################################
#                         GENERIC, NON NUMERICAL FORMATS
###############################################################################

ML_Void = ML_FormatConstructor(0, "void", "ERROR", lambda _: None)

###############################################################################
#                     FLOATING-POINT SPECIAL VALUES
###############################################################################
class FP_SpecialValue(object):
    """ parent to all floating-point constants """
    suffix_table = {
        ML_Binary32: ".f",
        ML_Binary64: ".d",
    }
    support_prefix = {
        ML_Binary32: "fp32",
        ML_Binary64: "fp64",
    }

def FP_SpecialValue_get_c_cst(self):
    prefix = self.support_prefix[self.precision]
    suffix = self.suffix_table[self.precision]
    return prefix + self.ml_support_name + suffix

def FP_SpecialValue_init(self, precision):
    self.precision = precision

def FP_SpecialValue_get_str(self):
    return "%s" % (self.ml_support_name)

def FP_SpecialValueBuilder(special_value):
    attr_map = {
        "ml_support_name": special_value,
        "__str__": FP_SpecialValue_get_str,
        "get_precision": lambda self: self.precision,
        "__init__": FP_SpecialValue_init,
        "get_c_cst": FP_SpecialValue_get_c_cst
    }
    return type(special_value, (FP_SpecialValue,), attr_map)

## Special value class builder for floatingg-point special values
#  using lib math (libm) macros and constant
def FP_MathSpecialValueBuilder(special_value):
    attr_map = {
        "ml_support_name": special_value,
        "__str__": FP_SpecialValue_get_str,
        "get_precision": lambda self: self.precision,
        "__init__": FP_SpecialValue_init,
        "get_c_cst": lambda self: self.ml_support_name
    }
    return type(special_value, (FP_SpecialValue,), attr_map)

#class FP_PlusInfty(FP_SpecialValueBuilder("_sv_PlusInfty")):
#    pass
class FP_PlusInfty(FP_MathSpecialValueBuilder("INFINITY")):
    pass
class FP_MinusInfty(FP_SpecialValueBuilder("_sv_MinusInfty")):
    pass
class FP_PlusOmega(FP_SpecialValueBuilder("_sv_PlusOmega")):
    pass
class FP_MinusOmega(FP_SpecialValueBuilder("_sv_MinusOmega")):
    pass
class FP_PlusZero(FP_SpecialValueBuilder("_sv_PlusZero")):
    pass
class FP_MinusZero(FP_SpecialValueBuilder("_sv_MinusZero")):
    pass
class FP_QNaN(FP_SpecialValueBuilder("_sv_QNaN")):
    pass
class FP_SNaN(FP_SpecialValueBuilder("_sv_SNaN")):
    pass


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

