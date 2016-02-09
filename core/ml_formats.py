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

from pythonsollya import *
from ..utility.common import ML_NotImplemented
import re

## Class of rounding mode type
class ML_FloatingPoint_RoundingMode_Type:
    def get_c_name(self):
        return "ml_rnd_mode_t"

## Class of floating-point rounding mode
class ML_FloatingPoint_RoundingMode:
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

## class for floating-point exception
class ML_FloatingPointException: pass

## class for type of floating-point exceptions
class ML_FloatingPointException_Type: 
  ## dummy placeholder to generate C constant for FP exception (should raise error) 
  def get_c_cst(self, value):
    return "NONE"
  def is_cst_decl_required(self):
    return False

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

    ## return the format's bit-size 
    def get_bit_size(self):
        """ <abstract> return the bit size of the format (if it exists) """
        print self # Exception ML_NotImplemented print
        raise ML_NotImplemented()

    def is_cst_decl_required(self):
        return False

    ## return the C code for args initialization
    def generate_c_initialization(self, *args):
      return None

    ## return the C code for value assignation to var
    # @param var variable assigned
    # @param value value being assigned
    # @param final boolean flag indicating if this assignation is the last in an assignation list
    def generate_c_assignation(self, var, value, final = True):
      final_symbol = ";\n" if final else ""
      return "%s = %s" % (var, value)


## Ancestor class for abstract format
class ML_AbstractFormat(ML_Format): 
    def __init__(self, c_name): 
        self.c_name = c_name

    def __str__(self):
        return self.c_name

    ## return the gappa constant corresponding to parameter
    #  @param cst_value constant value being translated
    def get_gappa_cst(self, cst_value):
        """ C code for constante cst_value """
        display(hexadecimal)
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
        "__str__": lambda self: self.name,
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
        self.bit_size = bit_size
        self.exponent_size = exponent_size
        self.field_size = field_size
        self.c_suffix = c_suffix
        self.c_name = c_name
        self.ml_support_prefix = ml_support_prefix
        self.sollya_object = sollya_object
        self.c_display_format = c_display_format

    ## return the sollya object encoding the format precision
    def get_sollya_object(self):
      return self.sollya_object

    def __str__(self):
        return self.c_name

    def get_c_name(self):
        return self.c_name

    def get_bias(self):
        return - 2**(self.get_exponent_size() - 1) + 1

    def get_emax(self):
        return 2**self.get_exponent_size() - 2 + self.get_bias()

    def get_emin_normal(self):
        return 1 + self.get_bias()

    def get_emin_subnormal(self):
        return 1 - (self.get_field_size() + 1) + self.get_bias()

    def get_c_display_format(self):
        return self.c_display_format

    def get_bit_size(self):
        """ return the format bit size """ 
        return self.bit_size

    def get_exponent_size(self):
        return self.exponent_size

    def get_exponent_interval(self):
        low_bound  = self.get_emin_normal()
        high_bound = self.get_emax()
        return Interval(low_bound, high_bound)

    def get_field_size(self):
        return self.field_size

    def get_c_cst(self, cst_value):
        """ C code for constante cst_value """
        if isinstance(cst_value, FP_SpecialValue): 
            return cst_value.get_c_cst()
        else:
            display(hexadecimal)
            if cst_value == 0:
                conv_result = "0.0" + self.c_suffix
            elif isinstance(cst_value, int):
                conv_result = str(float(cst_value)) + self.c_suffix
            else:
                conv_result  = str(cst_value) + self.c_suffix
            return conv_result

    def get_precision(self):
        """ return the bit-size of the mantissa """
        return self.get_field_size()

    def get_gappa_cst(self, cst_value):
        """ C code for constante cst_value """
        if isinstance(cst_value, FP_SpecialValue): 
            return cst_value.get_gappa_cst()
        else:
            display(hexadecimal)
            if isinstance(cst_value, int):
                return str(float(cst_value)) 
            else:
                return str(cst_value) 


class ML_FormatConstructor(ML_Format):
    def __init__(self, bit_size, c_name, c_display_format, get_c_cst):
        self.bit_size = bit_size
        self.c_name = c_name
        self.c_display_format = c_display_format
        self.get_c_cst = get_c_cst

    def __str__(self):
        return self.c_name

    def get_c_name(self):
        return self.c_name

    def get_c_display_format(self):
        return self.c_display_format

    def get_bit_size(self):
        return self.bit_size

## Ancestor to fixed-point format
class ML_Fixed_Format(ML_Format):
    """ parent to every Metalibm's fixed-point class """
    def __init__(self, support_format = None, align = 0):
      # integer format used to contain the fixed-point value 
      self.support_format = support_format

      # offset between the support LSB and the actual value LSB 
      self.support_right_align = align

    def set_support_format(self, _format):
      self.support_format = _format

    def get_support_format(self):
      return self.support_format

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
        self.c_name = ("" if self.signed else "u") + "int" + str(self.c_bit_size) + "_t"
        self.c_display_format = "%\"PRIx" + str(self.c_bit_size) + "\""

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
          return self.c_name
        elif self.signed:
          return "FS%d.%d" % (self.integer_size, self.frac_size)
        else:
          return "FU%d.%d" % (self.integer_size, self.frac_size)

    def get_c_name(self):
        return self.c_name

    def get_c_display_format(self):
        return self.c_display_format

    def get_bit_size(self):
        return self.integer_size + self.frac_size

    def get_c_cst(self, cst_value):
        """ C-language constant generation """
        encoded_value = int(cst_value * S2**self.frac_size)
        return ("" if self.signed else "U") + "INT" + str(self.c_bit_size) + "_C(" + str(encoded_value) + ")"

    def get_gappa_cst(self, cst_value):
        """ Gappa-language constant generation """
        return str(cst_value)

      

## Ancestor to standard (meaning integers)  fixed-point format
class ML_Standard_FixedPoint_Format(ML_Base_FixedPoint_Format):
  def __init__(self, integer_size, frac_size, signed = True):
    ML_Base_FixedPoint_Format.__init__(self, integer_size, frac_size, signed = signed, support_format = self, align = 0)

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
ML_Binary32 = ML_Std_FP_Format(32, 8, 23, "f", "float", "fp32", "%a", binary32)
ML_Binary64 = ML_Std_FP_Format(64, 11, 52, "", "double", "fp64", "%la", binary64)
ML_Binary80 = ML_Std_FP_Format(80, 15, 64, "L", "long double", "fp80", "%la", binary80)


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
  return precision in [ML_Int8,ML_UInt8,ML_Int16,ML_UInt16,ML_Int32,ML_UInt32,ML_Int64,ML_UInt64,ML_Int128,ML_UInt128]

def get_std_integer_support_format(precision):
  """ return the ML's integer format to contains
      the fixed-point format precision """
  assert(isinstance(precision, ML_Fixed_Format))
  format_map = {
    # signed
    True: {
      8: ML_Int8,
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
#                     COMPOUNT FORMAT
###############################################################################

class ML_Compound_Format(ML_Format):
    def __init__(self, c_name, c_field_list, field_format_list, ml_support_prefix, c_display_format, sollya_object):
        self.c_name = c_name
        self.ml_support_prefix = ml_support_prefix
        self.sollya_object = sollya_object
        self.c_display_format = c_display_format
        self.c_display_format = "undefined"
        self.c_field_list = c_field_list
        self.field_format_list = field_format_list

    def get_c_name(self):
        return self.c_name

    ## forces constant declaration during code generation
    def is_cst_decl_required(self):
        return True

    def get_c_cst(self, cst_value):
        tmp_cst = cst_value
        field_str_list = []
        for field_name, field_format in zip(self.c_field_list, self.field_format_list):
            # FIXME, round is only valid for double_double or triple_double stype format
            field_value = round(tmp_cst, field_format.sollya_object, RN)
            tmp_cst = cst_value - field_value
            field_str_list.append(".%s = %s" % (field_name, field_format.get_c_cst(field_value)))
        return "{%s}" % (", ".join(field_str_list))


    def get_c_display_format(self):
        return self.c_display_format

class ML_Compound_FP_Format(ML_Compound_Format, ML_FP_Format):
  pass
class ML_Compound_Integer_Format(ML_Compound_Format, ML_Fixed_Format):
  pass

# compound binary floating-point format declaration
ML_DoubleDouble = ML_Compound_FP_Format("ml_dd_t", ["hi", "lo"], [ML_Binary64, ML_Binary64], "", "", doubledouble)
ML_TripleDouble = ML_Compound_FP_Format("ml_td_t", ["hi", "me", "lo"], [ML_Binary64, ML_Binary64, ML_Binary64], "", "", tripledouble)

###############################################################################
#                     VECTOR FORMAT
###############################################################################

## common ancestor to every vector format
class ML_VectorFormat: 
  def __init__(self, scalar_format, vector_size):
    self.scalar_format = scalar_format
    self.vector_size   = vector_size 

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

class ML_CompoundVectorFormat(ML_VectorFormat, ML_Compound_Format):
  def __init__(self, format_name, vector_size, scalar_format, sollya_precision = None):
    ML_VectorFormat.__init__(self, scalar_format, vector_size)
    ML_Compound_Format.__init__(self, format_name, ["_[%d]" % i for i in xrange(vector_size)], [scalar_format for i in xrange(vector_size)], "", "", sollya_precision)


  def get_c_cst(self, cst_value):
    tmp_cst = cst_value
    field_str_list = []
    elt_value_list = [self.scalar_format.get_c_cst(cst_value[i]) for i in xrange(self.vector_size)]
    return "{._ = {%s}}" % (", ".join(elt_value_list))


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
def vector_format_builder(format_name, vector_size, scalar_format, sollya_precision = None, compound_constructor = ML_FloatingPointVectorFormat):
  return compound_constructor(format_name, vector_size, scalar_format, sollya_precision)

ML_Float2 = vector_format_builder("ml_float2_t", 2, ML_Binary32)
ML_Float4 = vector_format_builder("ml_float4_t", 4, ML_Binary32)
ML_Float8 = vector_format_builder("ml_float8_t", 8, ML_Binary32)

ML_Double2 = vector_format_builder("ml_double2_t", 2, ML_Binary64)
ML_Double4 = vector_format_builder("ml_double4_t", 4, ML_Binary64)
ML_Double8 = vector_format_builder("ml_double8_t", 8, ML_Binary64)

ML_Bool2  = vector_format_builder("ml_bool2_t", 2, ML_Bool, compound_constructor = ML_IntegerVectorFormat)
ML_Bool4  = vector_format_builder("ml_bool4_t", 4, ML_Bool, compound_constructor = ML_IntegerVectorFormat)
ML_Bool8  = vector_format_builder("ml_bool8_t", 8, ML_Bool, compound_constructor = ML_IntegerVectorFormat)

ML_Int2  = vector_format_builder("ml_int2_t", 2, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
ML_Int4  = vector_format_builder("ml_int4_t", 4, ML_Int32, compound_constructor = ML_IntegerVectorFormat)
ML_Int8  = vector_format_builder("ml_int8_t", 8, ML_Int32, compound_constructor = ML_IntegerVectorFormat)

ML_UInt2 = vector_format_builder("ml_uint2_t", 2, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
ML_UInt4 = vector_format_builder("ml_uint4_t", 4, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)
ML_UInt8 = vector_format_builder("ml_uint8_t", 8, ML_UInt32, compound_constructor = ML_IntegerVectorFormat)

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

class FP_PlusInfty(FP_SpecialValueBuilder("_sv_PlusInfty")): 
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


class FP_Context:
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

class ML_FunctionPrecision:
    pass

class ML_Faithful(ML_FunctionPrecision):
    pass

class ML_CorrectlyRounded(ML_FunctionPrecision):
    pass

class ML_DegradedAccuracy(ML_FunctionPrecision):
  def __init__(self, goal):
    self.goal = goal

  ## return the absolute or relative goal assocaited
  #  with the accuracy object
  def get_goal(self):
    return self.goal

class ML_DegradedAccuracyAbsolute(ML_DegradedAccuracy):
  """ absolute error accuracy """
  def __init__(self, absolute_goal):
    ML_DegradedAccuracy.__init__(self, absolute_goal)

  def __str__(self):
    return "ML_DegradedAccuracyAbsolute(%s)" % self.goal

class ML_DegradedAccuracyRelative(ML_DegradedAccuracy):
  """ relative error accuracy """
  def __init__(self, relative_goal):
    ML_DegradedAccuracy.__init__(self, relative_goal)

  def __str__(self):
    return "ML_DegradedAccuracyRelative(%s)" % self.goal


## degraded absolute accuracy alias
def daa(*args, **kwords):
    return ML_DegradedAccuracyAbsolute(*args, **kwords)
## degraded relative accuracy alias
def dar(*args, **kwords):
    return ML_DegradedAccuracyRelative(*args, **kwords)
