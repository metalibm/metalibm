# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Dec 27th, 2013
# last-modified:    Dec 27th, 2013
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


import sys
from common import Callable

class Log:
    """ log report class """
    log_stream = None

    class LogLevel(object): 
        """ log level builder """
        def __init__(self, level_name):
            self.name = level_name

    # log levels definition
    Warning = LogLevel("Warning")
    Info = LogLevel("Info")
    Error = LogLevel("Error")
    Debug = LogLevel("Debug")

    def report(level, msg, eol = "\n"):
        """ report log message """
        if Log.log_stream:
            Log.log_stream.write(msg + eol)
        else:
            print "%s: %s" % (level.name, msg)
        if level is Log.Error:
            sys.exit(1)
            # raise Exception()

    def set_log_stream(log_stream):
        Log.log_stream = log_stream

    set_log_stream = Callable(set_log_stream)
    report         = Callable(report)



