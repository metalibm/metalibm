# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Jun  5th, 2016
# last-modified:    Jun  5th, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import *
from .ml_formats import ML_FP_Format


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

  def get_bit_size(self):
    return self.bit_size

  def __str__(self):
    return self.c_name

# TODO : finish mpfr

mpfr_init = lambda self, symbol, symbol_object: "mpfr_init2(%s, %d)" % (symbol, self.get_bit_size())
mpfr_set = lambda *args: ""
ML_Mpfr_t = ML_Complex_Format("mpfr_t", mpfr_init, mpfr_set)
