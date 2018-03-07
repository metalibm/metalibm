# -*- coding: utf-8 -*-

""" HDL description specific formats (for RTL and signals) """

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
# created:          Nov 20th, 2016
# last-modified:    Mar  8th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
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
    return "X\"%s\"" % hex(value)[2:].replace("L","").zfill(bit_size // 4)
  else:
    return "\"%s\"" % bin(value)[2:].replace("L","").zfill(bit_size)


class RTL_FixedPointFormat(ML_Base_FixedPoint_Format):
  def __init__(self, integer_size, frac_size, signed = True, support_format = None, align = 0):
    ML_Base_FixedPoint_Format.__init__(self, integer_size, frac_size, signed, support_format = support_format, align = align)
    name = ("" if self.signed else "U") + "INT" + str(self.get_bit_size()) 
    self.name[VHDL_Code] = name

  def get_vhdl_cst(self, cst_value):
    return generic_get_vhdl_cst(cst_value * S2**self.get_frac_size(), self.get_bit_size())

  def get_name(self, language = VHDL_Code):
    return self.support_format.get_name(language)
  def get_code_name(self, language = VHDL_Code):
    return self.support_format.get_code_name(language)

  def is_cst_decl_required(self):
    return False

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
    return False

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
def fixed_point(int_size, frac_size, signed = True, support_format = None):
    """ Generate a fixed-point format """
    support_format = support_format or ML_StdLogicVectorFormat(int_size + frac_size) 
    new_precision = RTL_FixedPointFormat(
        int_size, frac_size,
        signed = signed,
        support_format = support_format
    )
    return new_precision

## Test whether @p precision is a fixed-point format
#  @return boolean value
def is_fixed_point(precision):
    return isinstance(precision, ML_Base_FixedPoint_Format)

