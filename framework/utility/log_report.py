# -*- coding: utf-8 -*-

###############################################################################
# This file is part of KFG
# Copyright (2013)
# All rights reserved
# created:          Dec 27th, 2013
# last-modified:    Dec 27th, 2013
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from common import Callable

class Log:
    """ log report class """

    class LogLevel(object): 
        """ log level builder """
        def __init__(self, level_name):
            self.name = level_name

    # log levels definition
    Warning = LogLevel("Warning")
    Info = LogLevel("Info")
    Error = LogLevel("Error")
    Debug = LogLevel("Debug")

    def report(level, msg):
        """ report log message """
        print "%s: %s" % (level.name, msg)
        if level is Log.Error:
            raise Exception()

    report = Callable(report)



