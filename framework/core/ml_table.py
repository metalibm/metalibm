# -*- coding: utf-8 -*-
###############################################################################
# This file is part of KFG
# Copyright (2013)
# All rights reserved
# created:          Mar 20th, 2014
# last-modified:    Mar 20th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from ml_operations import ML_LeafNode
from attributes import Attributes, attr_init

def create_multi_dim_array(dimensions):
    """ create a multi dimension array """
    if len(dimensions) == 1:
        return [None] * dimensions[0]
    else:
        dim = dimensions[0]
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

        self.table = create_multi_dim_array(dimensions)
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


    def get_c_definition(self, table_name, final = ";"):
        precision_c_name = self.get_storage_precision().get_c_name()
        return "%s %s[%s]" % (precision_c_name, table_name, "][".join([str(dim) for dim in self.dimensions]))


    def get_c_content_init(self):
        return get_table_c_content(self.table, self.dimensions, self.get_storage_precision())


    def get_str(self, depth = None, display_precision = False, tab_level = 0, memoization_map = {}):
        precision_str = "" if not display_precision else "[%s]" % str(self.get_storage_precision())
        return "  " * tab_level + "Table[%s]%s\n" % ("][".join([str(dim) for dim in self.dimensions]), precision_str)
        
        
