# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Jun  5th, 2015
# last-modified:    Jun  9th, 2015
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import *
from .ml_formats import *


class ML_Complex_Format(ML_FP_Format):
  """ Complex format with initizalition and set function """
  def __init__(self, c_name, init_function, set_function, bit_size = 53):
    self.c_name = c_name
    self.init_function = init_function
    self.set_function = set_function
    self.bit_size = bit_size

  def get_c_name(self):
    return self.c_name

  def generate_c_initialization(self, symbol, symbol_object):
    return self.init_function(self, symbol, symbol_object)

  def generate_c_assignation(self, var, result, final = True):
    final_symbol = ";\n" if final else ""
    return self.set_function(self, var, result) + final_symbol

  def get_bit_size(self):
    return self.bit_size

  def __str__(self):
    return self.c_name


class ML_Pointer_Format(ML_Format):
  """ wrapper for address/pointer format """
  def __init__(self, data_precision):
    self.data_precision = data_precision

  def get_c_name(self):
    return "%s*" % self.get_data_precision().get_c_name()

  def get_data_precision(self):
    return self.data_precision

def is_pointer(_format):
  """ boolean test to check whether _format is a pointer _format """
  return isinstance(_format, ML_Pointer_Format)


# TODO : finish mpfr

mpfr_init = lambda self, symbol, symbol_object: "mpfr_init2(%s, %d)" % (symbol, self.get_bit_size())
mpfr_set = lambda self, var, result: "mpfr_set(%s, %s, MPFR_RNDN)" % (var.get(), result.get())
ML_Mpfr_t = ML_Complex_Format("mpfr_t", mpfr_init, mpfr_set)

# definition of standard pointer types
ML_Binary32_p = ML_Pointer_Format(ML_Binary32)
ML_Binary64_p = ML_Pointer_Format(ML_Binary64)

ML_Int32_p    = ML_Pointer_Format(ML_Int32)
ML_Int64_p    = ML_Pointer_Format(ML_Int64)
