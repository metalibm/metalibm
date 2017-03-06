# -*- coding: utf-8 -*-

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

from .ml_formats import ML_Format
from ..code_generation.code_constant import *


class StdLogicDirection:
  class Downwards: 
    @staticmethod
    def get_descriptor(low, high):
      return "%d downto %d" % (high, low)
  class Upwards: 
    @staticmethod
    def get_descriptor(low, high):
      return "%d to %d" % (low, high)

def get_2scomplement_neg(value, size):
  value = int(abs(value))
  assert value < (S2**(size-1) - 1)
  return (~value+1)

class ML_StdLogicVectorFormat(ML_Format):
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

  def get_cst(self, cst_value, language = C_Code):
    if language is VHDL_Code:
      return self.get_vhdl_cst(cst_value)
    else:
      # default case
      return self.get_vhdl_cst(cst_value)
  def get_bit_size(self):
    return self.bit_size

  def get_vhdl_cst(self, value):
    #return "\"%s\"" % bin(value)[2:].zfill(self.bit_size)
    value = int(value)
    value &= int(2**self.bit_size - 1)
    assert self.bit_size > 0
    if self.bit_size % 4 == 0:
      return "X\"%s\"" % hex(value)[2:].replace("L","").zfill(self.bit_size / 4)
    else:
      return "\"%s\"" % bin(value)[2:].replace("L","").zfill(self.bit_size)

class ML_StdLogicClass(ML_Format):
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

# std_logic type singleton
ML_StdLogic = ML_StdLogicClass()


