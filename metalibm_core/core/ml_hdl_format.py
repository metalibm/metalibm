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

class ML_StdLogicVectorFormat(ML_Format):
  def __init__(self, bit_size, offset = 0, direction = StdLogicDirection.Downwards):
    ML_Format.__init__(self)
    self.bit_size = bit_size
    self.name[VHDL_Code] = "std_logic_vector({direction_descriptor})".format(direction_descriptor = direction.get_descriptor(offset, offset + self.bit_size - 1))

  def __str__(self):
    return self.name[VHDL_Code]
  def get_name(self, language = VHDL_Code):
    return self.name[language]

class ML_StdLogicClass(ML_Format):
  def __init__(self):
    ML_Format.__init__(self)
    self.bit_size = 1
    self.name[VHDL_Code] = "std_logic"
  def __str__(self):
    return self.name[VHDL_Code]
  def get_name(self, language = VHDL_Code):
    return self.name[language]

# std_logic type singleton
ML_StdLogic = ML_StdLogicClass()


