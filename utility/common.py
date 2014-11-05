# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 23rd, 2013
# last-modified:    Oct 30th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


import sys

class ML_NotImplemented(Exception):
    """ not-implemented exception """
    pass


class Callable:
    """ wrapper for static class function member """ 
    def __init__(self, anycallable):
        self.__call__ = anycallable


def zip_index(list_):
    return zip(list_, range(len(list_)))

def tupelize(single_elt):
    return (single_elt,) 


def test_flag_option(flag_name, flag_value, default_value):
    return flag_value if flag_name in sys.argv else default_value

def extract_option_value(option_name, default_value, help_map = None, help_str = "", processing = lambda x: x):
    if help_map != None: 
        help_map[option_name] = "[%s] %s" % (default_value, help_str)
    return processing(sys.argv[sys.argv.index(option_name)+1] if option_name in sys.argv else default_value)

def extract_option_list_value(option_name, default_value):
    return sys.argv[sys.arg.index(option_name+1)].split(",") if option_name in sys.argv else default_value
