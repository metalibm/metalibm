# -*- coding: utf-8 -*-

""" HDL description specific formats (for RTL and signals) """

###############################################################################
# This file is part of New Metalibm tool
# Copyrights Nicolas Brunie (2016)
# All rights reserved
# created:          Nov 20th, 2016
# last-modified:    Nov 20th, 2016
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
# description: Declaration of Node formats for hardware designs
###############################################################################

import sollya

from .ml_formats import ML_Format, ML_Base_FixedPoint_Format, ML_Fixed_Format
from ..code_generation.code_constant import VHDL_Code

## Helper constant: 2 as a sollya object
S2 = sollya.SollyaObject(2)

class StdLogicDirection:
  class Downwards: 
    @staticmethod
    def get_descriptor(low, high):
      return "%d downto %d" % (high, low)
  class Upwards: 
    @staticmethod
    def get_descriptor(low, high):
      return "%d to %d" % (low, high)

## Computes the negation of the positive @p value on 
#  @p size bits
#  Fails if value exceeds the largest representable
#  number of @p size - 1 bits
def get_2scomplement_neg(value, size):
  value = int(abs(value))
  assert value < (S2**(size-1) - 1)
  return (~value+1)

def generic_get_vhdl_cst(value, bit_size):
  #return "\"%s\"" % bin(value)[2:].zfill(self.bit_size)
  value = int(value)
  value &= int(2**bit_size - 1)
  assert bit_size > 0
  assert value <= (2**bit_size - 1)
  if bit_size % 4 == 0:
    return "X\"%s\"" % hex(value)[2:].replace("L","").zfill(bit_size / 4)
  else:
    return "\"%s\"" % bin(value)[2:].replace("L","").zfill(bit_size)


class RTL_FixedPointFormat(ML_Base_FixedPoint_Format):
  def __init__(self, integer_size, frac_size, signed = True, support_format = None, align = 0):
    ML_Fixed_Format.__init__(self, support_format, align)
    self.integer_size = integer_size
    self.frac_size    = frac_size
    self.signed       = signed
    name = ("" if self.signed else "U") + "INT" + str(self.get_bit_size()) 
    self.name[VHDL_Code] = name

  def get_vhdl_cst(self, cst_value):
    return generic_get_vhdl_cst(cst_value * S2**self.get_frac_size(), self.get_bit_size())

  def get_name(self, language = VHDL_Code):
    return self.support_format.get_name(language)
  def get_code_name(self, language = VHDL_Code):
    return self.support_format.get_code_name(language)

  def is_cst_decl_required(self):
    return True

  def get_cst(self, cst_value, language = VHDL_Code):
    if language is VHDL_Code:
      return self.get_vhdl_cst(cst_value)
    else:
      raise NotImplementedError

## Format class for multiple bit signals
class ML_StdLogicVectorFormat(ML_Format):
  """ Format class for multiple bit signals """
  def __init__(self, bit_size, offset = 0, direction = StdLogicDirection.Downwards):
    assert bit_size > 0
    bit_size = int(bit_size)
    offset   = int(offset)
    ML_Format.__init__(self)
    self.bit_size = bit_size
    self.name[VHDL_Code] = "std_logic_vector({direction_descriptor})".format(direction_descriptor = direction.get_descriptor(offset, offset + self.bit_size - 1))
    self.display_format[VHDL_Code] = "%s"

  def __str__(self):
    return self.name[VHDL_Code]
  def get_name(self, language = VHDL_Code):
    return self.name[language]

  def get_cst(self, cst_value, language = VHDL_Code):
    if language is VHDL_Code:
      return self.get_vhdl_cst(cst_value)
    else:
      # default case
      return self.get_vhdl_cst(cst_value)
  def get_bit_size(self):
    return self.bit_size

  def get_integer_coding(self, value, language = VHDL_Code):
    return int(value)

  def get_vhdl_cst(self, value):
    return generic_get_vhdl_cst(value, self.bit_size)

  def is_cst_decl_required(self):
    return True

## Class of single bit value format
class ML_StdLogicClass(ML_Format):
  """ class of single bit value signals """
  def __init__(self):
    ML_Format.__init__(self)
    self.bit_size = 1
    self.name[VHDL_Code] = "std_logic"
    self.display_format[VHDL_Code] = "%s"

  def __str__(self):
    return self.name[VHDL_Code]
  def get_name(self, language = VHDL_Code):
    return self.name[language]
  def get_cst(self, value, language = VHDL_Code):
    return "'%d'" % value
  def get_bit_size(self):
    return self.bit_size
  def get_support_format(self):
    return self

## std_logic type singleton
ML_StdLogic = ML_StdLogicClass()


## Helper to build RTL fixed-point formats
def fixed_point(int_size, frac_size, signed = True):
    new_precision = RTL_FixedPointFormat(
        int_size, frac_size,
        signed = signed,
        support_format = ML_StdLogicVectorFormat(int_size + frac_size)
    )
    return new_precision

## Test whether @p precision is a fixed-point format
#  @return boolean value
def is_fixed_point(precision):
    return isinstance(precision, ML_Base_FixedPoint_Format)

