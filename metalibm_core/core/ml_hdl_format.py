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

from .ml_formats import (
    ML_Format, ML_Base_FixedPoint_Format, ML_Fixed_Format,
    VirtualFormat, get_virtual_cst,
    ML_StringClass, DisplayFormat,
)
from ..code_generation.code_constant import VHDL_Code

from ..utility.log_report import Log

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

def generic_get_vhdl_cst(value, bit_size, is_std_logic=False):
    """
        :param is_std_logic: indicates whether or not fixed-point support format
            is a single-bit std_logic (rather than std_logic_vector) which implies
            particular value string generation
        :type is_std_logic: bool

    """
    try:
        value = int(value)
        value &= int(2**bit_size - 1)
    except TypeError:
        Log.report(Log.Error, "unsupported value={}/bit_size={} in generic_get_vhdl_cst".format(value, bit_size), error=TypeError)
    assert bit_size > 0
    assert value <= (2**bit_size - 1)
    if is_std_logic:
        assert bit_size == 1
        if bit_size != 1:
            Log.report(Log.Error, "bit_size must be 1 (not {}) for generic_get_vhdl_cst is_std_logic=True)", bit_size)
        return "'%s'" % bin(value)[2:].replace("L","")
    elif bit_size % 4 == 0:
        return "X\"%s\"" % hex(value)[2:].replace("L","").zfill(bit_size // 4)
    else:
        return "\"%s\"" % bin(value)[2:].replace("L","").zfill(bit_size)

class ML_UnevaluatedFormat:
    """ generic virtual class for unevaluated format.
        Unevaluated format is a parameterized format whose parameters can be left
        partially unevaluated during declaration """

    def evaluate(self, node_value_solver):
        """ function to compute final value for node parameters
            returns a fully evaluated node """
        raise NotImplementedError

    def get_c_cst(self, cst_value):
        Log.report(Log.Error, "unevaluated format can not be used for get_c_cst")
        raise NotImplementedError

class UndefinedFixedPointFormat(ML_Base_FixedPoint_Format, ML_UnevaluatedFormat):
    """ class for undefined fixed-point format: the corresponding
        object is a fixed-point format whose integer and fractional size
        have not been determined yet (they can be unevaluated expressions) """
    def __init__(self, integer_size, frac_size, signed=True, support_format=None, align=0):
        ML_Base_FixedPoint_Format.__init__(self, integer_size, frac_size, signed, support_format=support_format, align=align)
        name = "unevaluated_fixed_point"
        self.name[VHDL_Code] = name

    def evaluate(self, node_value_solver):
        int_size = node_value_solver(self.integer_size)
        frac_size = node_value_solver(self.frac_size)
        return fixed_point(int_size, frac_size, signed=self.signed, support_format=self.support_format) 

    def get_c_cst(self, cst_value):
        return ML_UnevaluatedFormat.get_c_cst(self, cst_value)

    def __str__(self):
        if self.signed:
          return "FS<undef>"
        else:
          return "FU<undef>"

class RTL_FixedPointFormat(ML_Base_FixedPoint_Format):
    def __init__(self, integer_size, frac_size, signed = True, support_format = None, align = 0):
        ML_Base_FixedPoint_Format.__init__(self, integer_size, frac_size, signed, support_format = support_format, align = align)
        name = ("" if self.signed else "U") + "INT" + str(self.get_bit_size()) 
        self.name[VHDL_Code] = name

    def get_vhdl_cst(self, cst_value):
        is_std_logic = (self.support_format == ML_StdLogic)
        return generic_get_vhdl_cst(cst_value * S2**self.get_frac_size(), self.get_bit_size(), is_std_logic=is_std_logic)

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

    @staticmethod
    def parse_from_match(format_match):
        """ Parse the description of a class format and generates
            the format object """
        assert not format_match is None
        name = format_match.group("name")
        int_size = int(format_match.group("integer"))
        frac_size = int(format_match.group("frac"))
        is_signed = (name == "FS")
        return fixed_point(int_size, frac_size, signed=is_signed)


def HdlVirtualFormat(base_precision):
    """ Build a VirtualFormat to wrap @p base_precision """
    return VirtualFormat(
        base_format=base_precision,
        support_format=ML_StdLogicVectorFormat(base_precision.get_bit_size()),
        get_cst=get_virtual_cst
    )

class HDL_LowLevelFormat(ML_Format):
  format_prefix = "undefined_prefix"
  """ Format class for multiple bit signals """
  def __init__(self, bit_size, offset = 0, direction = StdLogicDirection.Downwards):
    ML_Format.__init__(self)
    try:
        assert bit_size > 0
        bit_size = int(bit_size)
        offset   = int(offset)
    except TypeError:
        # bit_size, offset are in incompatible format, we switch to lazy
        # resolution
        bit_size = None
        offset = None
        self.name[VHDL_Code] = "UNSOLVED_FORMAT"
        self.bit_size = None
        self.resolved = False
    else:
        self.bit_size = bit_size
        self.name[VHDL_Code] = "{format_prefix}({direction_descriptor})".format(
            format_prefix=self.format_prefix,
            direction_descriptor = direction.get_descriptor(offset, offset + self.bit_size - 1))
        self.resolved = True
    self.direction = direction
    self.offset = offset
    self.display_format[VHDL_Code] = "%s"

  def __str__(self):
    return self.name[VHDL_Code]
  def __repr__(self):
    return self.name[VHDL_Code]
  def get_name(self, language=VHDL_Code):
    language = VHDL_Code if language is None else language
    return self.name[language]
  def get_code_name(self, language=VHDL_Code):
    return self.get_name(language)

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
  def __eq__(self, format2):
    return isinstance(format2, ML_StdLogicVectorFormat) and self.bit_size == format2.bit_size and self.offset == format2.offset
  def __hash__(self):
    return ML_Format.__hash__(self)



## Format class for multiple bit signals
class ML_StdLogicVectorFormat(HDL_LowLevelFormat):
    """ classic std_logic_vector format """
    format_prefix = "std_logic_vector"
    pass

class HDL_NumericVectorFormat(HDL_LowLevelFormat):
    pass

class HDL_UnsignedVectorFormat(HDL_NumericVectorFormat):
    """ std_logic_arith unsigned format """
    format_prefix = "unsigned"

class HDL_SignedVectorFormat(HDL_NumericVectorFormat):
    """ std_logic_arith unsigned format """
    format_prefix = "signed"

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
  def get_integer_coding(self, value, language=VHDL_Code):
    return int(value)

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

def lazy_fixed_point(int_size, frac_size, signed=True, support_format=None):
    """ Generate a lazy fixed-point format with unevaluated integer and fractionnal sizes """
    new_precision = UndefinedFixedPointFormat(
        int_size, frac_size,
        signed=signed,
        support_format=support_format
    )
    return new_precision

## Test whether @p precision is a fixed-point format
#  @return boolean value
def is_fixed_point(precision):
    return isinstance(precision, ML_Base_FixedPoint_Format)

def is_unevaluated_format(precision):
    return isinstance(precision, ML_UnevaluatedFormat)

def get_unsigned_precision(precision):
    """ convert a sign agnostic precision (std_logic_vector)
        to its unsigned counterpart (same size, same offset, same
        direction """
    if isinstance(precision, HDL_UnsignedVectorFormat):
        return precision
    elif isinstance(precision, ML_StdLogicVectorFormat):
        return HDL_UnsignedVectorFormat(precision.bit_size, precision.offset, precision.direction)
    else:
        raise NotImplementedError
        
def get_signed_precision(precision):
    """ convert a sign agnostic precision (std_logic_vector)
        to its signed counterpart (same size, same offset, same
        direction """
    if isinstance(precision, HDL_SignedVectorFormat):
        return precision
    elif isinstance(precision, ML_StdLogicVectorFormat):
        return HDL_SignedVectorFormat(precision.bit_size, precision.offset, precision.direction)
    else:
        raise NotImplementedError

def get_numeric_precision(precision, is_signed):
    """ convert a sign agnostic precision (std_logic_vector)
        to its signed/unsigned counterpart (same size, same offset, same
        direction """
    if is_signed:
        return get_signed_precision(precision)
    else:
        return get_unsigned_precision(precision)

## 
HDL_FILE = ML_StringClass("file", DisplayFormat("%s"), lambda self, s: "\"{}\"".format(s)) 

HDL_LINE = ML_StringClass("line",  DisplayFormat("%s"), lambda self, s: "\"{}\"".format(s)) 
