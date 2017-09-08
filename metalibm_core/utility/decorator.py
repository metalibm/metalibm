# -*- coding: utf-8 -*-

""" Metalibm function decorator module """

###############################################################################
# This file is part of the New Metalibm tool
# Copyright (2017-)
# All rights reserved
# created:          Aug 21st, 2017
# last-modified:    Aug 21st, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################

def safe(operation):
    """ function decorator to forward None value without raising Exception 

        Args:
            operation (function *args-> result)
        Return:
            function *args -> result
    """
    return lambda *args: None if None in args else operation(*args)


