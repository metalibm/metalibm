# -*- coding: utf-8 -*-
###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014)
# All rights reserved
# created:          Mar 20th, 2014
# last-modified:    Nov 17th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import Interval

from ml_operations import ML_LeafNode, BitLogicAnd, BitLogicRightShift, TypeCast, Constant
from attributes import Attributes, attr_init
from ml_formats import ML_Int32, ML_Int64, ML_UInt32, ML_UInt64

def create_multi_dim_array(dimensions, init_data = None):
    """ create a multi dimension array """
    if len(dimensions) == 1:
        if init_data != None:
            return [init_data[i] for i in xrange(dimensions[0])]
        else:
            return [None for i in xrange(dimensions[0])]
    else:
        dim = dimensions[0]
        if init_data != None:
            return [create_multi_dim_array(dimensions[1:], init_data[i]) for i in xrange(dim)]
        else:
            return [create_multi_dim_array(dimensions[1:]) for i in xrange(dim)]


def get_table_c_content(table, dimensions, storage_precision):
    if len(dimensions) == 1:
        return "{" + ", ".join([storage_precision.get_c_cst(value) for value in table]) + "}"
    else:
        code = "{\n  "
        code += ",\n  ".join(get_table_c_content(line, dimensions[1:], storage_precision) for line in table)
        code += "\n}"
        return code

class ML_Table(ML_LeafNode):
    """ Metalibm Table object """
    def __init__(self, **kwords): 
        self.attributes = Attributes(**kwords)
        dimensions = attr_init(kwords, "dimensions", [])
        storage_precision = attr_init(kwords, "storage_precision", None)
        init_data = attr_init(kwords, "init_data", None)

        self.table = create_multi_dim_array(dimensions, init_data = init_data)
        self.dimensions = dimensions
        self.storage_precision = storage_precision

    def __setitem__(self, key, value):
        self.table[key] = value

    def __getitem__(self, key):
        return self.table[key]

    def get_storage_precision(self):
        return self.storage_precision

    def get_precision(self):
        return self.get_storage_precision()

    def get_tag(self):
        return self.attributes.get_tag()

    def get_subset_interval(self, index_function, range_set):
        # init bound values
        low_bound  = None
        high_bound = None
        # going through the selected valued list
        # to build the range interval
        for indexes in range_set: 
          value = index_function(self, indexes)
          if low_bound is None or low_bound > value: low_bound = value
          if high_bound is None or high_bound < value: high_bound = value
        return Interval(low_bound, high_bound)


    def get_c_definition(self, table_name, final = ";"):
        precision_c_name = self.get_storage_precision().get_c_name()
        return "%s %s[%s]" % (precision_c_name, table_name, "][".join([str(dim) for dim in self.dimensions]))

    def get_c_content_init(self):
        return get_table_c_content(self.table, self.dimensions, self.get_storage_precision())

    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}, display_attribute = False, display_id = False):
        id_str     = ("[id=%x]" % id(self)) if display_id else ""
        attribute_str = "" if not display_attribute else self.attributes.get_str(tab_level = tab_level)
        precision_str = "" if not display_precision else "[%s]" % str(self.get_storage_precision())
        return "  " * tab_level + "Table[%s]%s%s%s\n" % ("][".join([str(dim) for dim in self.dimensions]), precision_str, id_str, attribute_str)
        


def generic_index_function(index_size, variable):
    inter_precision = {32: ML_Int32, 64: ML_Int64}[variable.get_precision().get_bit_size()]

    index_mask   = Constant(2**index_size - 1, precision = inter_precision)
    shift_amount = Constant(variable.get_precision().get_field_size() - index_size, precision = ML_UInt32) 

    return BitLogicAnd(BitLogicRightShift(TypeCast(variable, precision = inter_precision), shift_amount, precision = inter_precision), index_mask, precision = inter_precision) 

        
class ML_ApproxTable(ML_Table):
    def __init__(self, **kwords):
        ML_Table.__init__(self, **kwords)
        index_size = attr_init(kwords, "index_size", 7)
        self.index_size = index_size
        index_function = attr_init(kwords, "index_function", lambda variable: generic_index_function(index_size, variable))
        self.index_function = index_function

    def get_index_function(self):
        """ <index_function> getter """
        return self.index_function



