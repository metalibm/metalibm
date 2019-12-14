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

###############################################################################
# created:          Jun  5th, 2015
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from .ml_formats import *


class ML_AdvancedFormat(ML_FP_Format):
  """ Complex format with initialization and set function """
  def __init__(self, c_name, init_function, set_function, bit_size = 53):
    self.name = {C_Code: c_name}
    self.init_function = init_function
    self.set_function = set_function
    self.bit_size = bit_size

  def get_c_name(self):
    return self.name[C_Code]

  def generate_c_initialization(self, symbol, symbol_object):
    return self.init_function(self, symbol, symbol_object)

  def generate_c_assignation(self, var, result, final = True):
    final_symbol = ";\n" if final else ""
    return self.set_function(self, var, result) + final_symbol

  def get_bit_size(self):
    return self.bit_size

  def __str__(self):
    return self.name[C_Code]


class ML_Pointer_Format(ML_Format):
  """ wrapper for address/pointer format """
  def __init__(self, data_precision):
    self.data_precision = data_precision

  def get_name(self, language = C_Code):
    return "%s*" % self.get_data_precision().get_name(language)

  def get_data_precision(self):
    return self.data_precision


  def __eq__(self, other):
      """ equality predicate for custom fixed-point format object """
      return (is_pointer_format(other)) and (self.data_precision == other.data_precision)

  def __ne__(self, other):
      """ unequality predicate for custom fixed-point format object """
      return not (self == other)

def is_pointer_format(_format):
  """ boolean test to check whether _format is a pointer _format """
  return isinstance(_format, ML_Pointer_Format)


# TODO : finish mpfr

mpfr_init = lambda self, symbol, symbol_object: "mpfr_init2(%s, %d)" % (symbol, self.get_bit_size())
mpfr_set = lambda self, var, result: "mpfr_set(%s, %s, MPFR_RNDN)" % (var.get(), result.get())
ML_Mpfr_t = ML_AdvancedFormat("mpfr_t", mpfr_init, mpfr_set)

# definition of standard pointer types
ML_Binary32_p = ML_Pointer_Format(ML_Binary32)
ML_Binary64_p = ML_Pointer_Format(ML_Binary64)

ML_Int32_p    = ML_Pointer_Format(ML_Int32)
ML_Int64_p    = ML_Pointer_Format(ML_Int64)
