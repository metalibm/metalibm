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
# created:          Mar 20th, 2014
# last-modified:    Mar  8th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

## @package ml_table
#  Metalibm Table (numerical array)

from sollya import Interval

from .ml_operations import (
    ML_LeafNode, BitLogicAnd, BitLogicRightShift, TypeCast, Constant
)
from .attributes import Attributes, attr_init
from .ml_formats import (
    ML_Int32, ML_Int64, ML_UInt32, ML_UInt64, ML_Format, ML_FP_Format)
from .ml_complex_formats import (
    ML_Pointer_Format,
)
from ..code_generation.code_constant import *
from ..code_generation.code_configuration import CodeConfiguration
from .special_values import is_numeric_value

from ..utility.source_info import SourceInfo

## \defgroup ml_table ml_table
#  @{

## create a multi-dimension array
#  @param dimensions tuple of size in each dimensions
#         the number of dimensions is the tuple length
#  @return a python multi-dimensional list whose dimensions
#          matches @p dimensions
def create_multi_dim_array(dimensions, init_data = None):
    """ create a multi dimension array """
    if len(dimensions) == 1:
        if init_data != None:
            return [init_data[i] for i in range(dimensions[0])]
        else:
            return [None for i in range(dimensions[0])]
    else:
        dim = dimensions[0]
        if init_data != None:
            return [create_multi_dim_array(dimensions[1:], init_data[i]) for i in range(dim)]
        else:
            return [create_multi_dim_array(dimensions[1:]) for i in range(dim)]


## return the C encoding of the array @table whose dimension tuple is @p dimensions
#  and data's format is @p storage_precision
def get_table_content(table, dimensions, storage_precision, language=C_Code, max_len=80):
    row_suffix = "\n" + CodeConfiguration.tab
    if len(dimensions) == 1:
        code = ""
        new_line = "{"
        for value in table:
            value_str = storage_precision.get_cst(value, language=language)
            if len(new_line) + len(value_str) > max_len:
                if new_line[0] == "{":
                    # if we are at the first line and an actual line break
                    # occurs then we insert a CR after the opening {
                    new_line = "{" + row_suffix + new_line[1:]
                code += new_line + row_suffix
                new_line = ""
            new_line += " " + value_str +","
        # removing terminal'','
        if new_line[-1] == ",":
            new_line = new_line[:-1]
        if new_line[0] == "{":
            # no line break => closing '}' on the same line
            code += new_line + "}"
        else:
            # at least one line break => closing '}' on the next line
            code += new_line + "\n" + "}"
        return code
    else:
        code = "{" + row_suffix
        code += ("," + row_suffix).join(get_table_content(line, dimensions[1:], storage_precision, language = language) for line in table)
        code += "\n}"
        return code

## Format for Table object (storage precision)
class ML_TableFormat(ML_Format):
  def __init__(self, storage_precision, dimensions):
    self.storage_precision = storage_precision
    # uplet of dimensions 
    self.dimensions = dimensions

  ## Generate code name for the table format
  def get_code_name(self, language=None):
    storage_code_name = self.storage_precision.get_code_name(language)
    return "{}*".format(storage_code_name)

  def get_storage_precision(self):
    return self.storage_precision
  def get_dimensions(self):
    return self.dimensions
  def get_match_format(self):
    return self #.storage_precision.get_match_format(), self.get_dimensions()


class ML_Table(ML_LeafNode):
    """ Metalibm Table object """
    ## string used in get_str
    str_name = "Table"
    ## ML_Table constructor 
    #  @param empty indicates whether the table should be initialized empty
    def __init__(self, empty = False, storage_precision = None, **kwords): 
        self.attributes = Attributes(**kwords)
        dimensions = attr_init(kwords, "dimensions", [])
        #storage_precision = attr_init(kwords, "storage_precision", None)
        init_data = attr_init(kwords, "init_data", None)

        self.table = create_multi_dim_array(dimensions, init_data = init_data)
        self.dimensions = dimensions
        self.storage_precision = storage_precision

        self.empty = empty

        self.index = -1

    def __setitem__(self, key, value):
        self.table[key] = value

    def __getitem__(self, key):
        return self.table[key]

    def is_empty(self):
        return self.empty

    def get_storage_precision(self):
        return self.storage_precision

    def get_precision(self):
        return self.get_storage_precision()

    def get_tag(self):
        return self.attributes.get_tag()

    ## return @p self data content
    def get_data(self):
        return self.table

    def get_subset_interval(self, index_function, range_set):
        # init bound values
        low_bound  = None
        high_bound = None
        # going through the selected valued list
        # to build the range interval
        for indexes in range_set:
          value = index_function(self, indexes)
          if not is_numeric_value(value):
            low_bound = value
            high_bound = value
          else:
            if low_bound is None or low_bound > value: low_bound = value
            if high_bound is None or high_bound < value: high_bound = value
        return Interval(low_bound, high_bound)

    ## return the interval containing each value of the Table
    def get_interval(self):
        def build_range_set(dimensions, prefix = []):
          if len(dimensions) == 1:
            return [(prefix + [i]) for i in range(dimensions[0])]
          else:
            return sum([build_range_set(dimensions[1:], prefix = prefix + [i]) for i in range(dimensions[0])], [])

        def get_rec_index(table, range_tuple):
          if len(range_tuple) == 1:
            return table[range_tuple[0]]
          else:
            return get_rec_index(table[range_tuple[0]], range_tuple[1:])

        def get_index(self, range_tuple):
          return get_rec_index(self.table, range_tuple)
          
        return self.get_subset_interval(get_index, build_range_set(self.dimensions))


    def get_definition(self, table_name, final = ";", language = C_Code):
        precision_name = self.get_storage_precision().get_name(language = language)
        return "%s %s[%s]" % (precision_name, table_name, "][".join([str(dim) for dim in self.dimensions]))

    def get_content_init(self, language = C_Code):
        return get_table_content(self.table, self.dimensions, self.get_storage_precision(), language = language)

    def get_str(
            self, depth = None, display_precision = False,
            tab_level = 0, memoization_map = {}, display_attribute = False,
            display_id = False, custom_callback = lambda optree: ""
        ):
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        precision_str = "" if not display_precision else "[%s]" % str(self.get_storage_precision())
        custom_str = custom_callback(self)
        return CodeConfiguration.tab * tab_level + custom_str + "%s[%s]%s%s%s\n" % (
            self.str_name,
            "][".join([str(dim) for dim in self.dimensions]),
            precision_str, id_str, attribute_str
        )

    def copy(self, copy_map = None):
        copy_map = {} if copy_map is None else copy_map
        if self in copy_map:
            return copy_map[self]
        else:
            kwords = self.attributes.__dict__.copy()
            kwords.update({
                'dimensions' : self.dimensions,
                'storage_precision' : self.storage_precision,
                'init_data': self.table
                })
            new_copy = self.__class__(**kwords)
            copy_map[self] = new_copy
            return new_copy


def generic_mantissa_msb_index_fct(index_size, variable):
    """ Build an indexing node for a table assuming index
        must be build from floating-point input variable
        and index is expected to be the index_size-th most significant
        bits of variable's mantissa """
    assert ML_FP_Format.is_fp_format(variable.get_precision())
    # determining integer format whose size matches variable's floating-point format
    int_precision = {32: ML_Int32, 64: ML_Int64}[variable.get_precision().get_bit_size()]
    # building an index mask from the index_size
    index_mask   = Constant(2**index_size - 1, precision = int_precision)
    shift_amount = Constant(
        variable.get_precision().get_field_size() - index_size, precision=int_precision
    )
    return BitLogicAnd(
        BitLogicRightShift(
            TypeCast(variable, precision=int_precision),
            shift_amount, precision=int_precision
        ),
        index_mask, precision=int_precision)


## New Table class
#  This class uses ML_TableFormat as precision for the
#  table object
class ML_NewTable(ML_Table):
  ## string used in get_str
  str_name = "NewTable"
  ## ML_NewTable constructor
  def __init__(self, dimensions = (16,), storage_precision = None, empty = False, **kw):
    ML_Table.__init__(self, dimensions = dimensions, storage_precision = storage_precision, empty = empty, **kw)
    self.precision = ML_TableFormat(storage_precision, dimensions)

  def get_precision(self):
    return self.precision

  def get_precision_as_pointer_format(self):
    return ML_Pointer_Format(self.precision.storage_precision)


## Table specifically used to described an approximation i.e a seed-like
#  instruction.
#  This class has source-info information to retrieve declaration location
class ML_ApproxTable(ML_NewTable):
    str_name = "ApproxTable"
    def __init__(self, index_size=7, index_function=None, **kwords):
        ML_NewTable.__init__(self, **kwords)
        self.index_size = index_size
        index_function = index_function or (lambda variable: generic_mantissa_msb_index_fct(index_size, variable))
        self.index_function = index_function
        self.sourceinfo = SourceInfo.retrieve_source_info(0)

    def get_source_info(self):
        return self.sourceinfo

    def get_index_function(self):
        """ <index_function> getter """
        return self.index_function

## @} 
# end of metalibm's Doxygen ml_table group
