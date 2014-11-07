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


