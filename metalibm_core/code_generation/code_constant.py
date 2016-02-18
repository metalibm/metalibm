# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014-2016)
# All rights reserved
# created:          Jul  3rd, 2014
# last-modified:    Feb 22nd, 2016
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

## Standard C class Code
class C_Code(object):
  def __str__(self):
    return "C_Code"

class Gappa_Code:
  def __str__(self):
    return "Gappa_Code"

## OpenCL-C class Code
class OpenCL_Code:
  def __str__(self):
    return "OpenCL_Code"
