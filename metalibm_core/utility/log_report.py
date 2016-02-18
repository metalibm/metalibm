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

class Log:
    """ log report class """
    log_stream    = None
    dump_stdout   = False
    exit_on_error = True

    @staticmethod
    def set_dump_stdout(new_dump_stdout):
      Log.dump_stdout = new_dump_stdout
      

    class LogLevel(object): 
        """ log level builder """
        def __init__(self, level_name):
            self.name = level_name

    # log levels definition
    Warning = LogLevel("Warning")
    Info = LogLevel("Info")
    Error = LogLevel("Error")
    Debug = LogLevel("Debug")
    Verbose = LogLevel("Verbose")

    # list of enabled log levels
    enabled_levels = [Warning, Info, Error]

    @staticmethod
    def report(level, msg, eol = "\n"):
        """ report log message """
        if Log.log_stream:
            Log.log_stream.write(msg + eol)
            if Log.dump_stdout: 
              print "%s: %s" % (level.name, msg)
        elif level in Log.enabled_levels:
            print "%s: %s" % (level.name, msg)
        if level is Log.Error:
            if Log.exit_on_error:
              sys.exit(1)
            else:
              raise Exception()

    ## enable display of the specific log level
    #  @param level log-level to be enabled
    @staticmethod
    def enable_level(level):
      Log.enabled_levels.append(level)

    ## disable display of the specific log level
    #  @param level log-level to be disabled
    @staticmethod
    def disable_level(level):
      Log.enabled_levels.remove(level)

    @staticmethod
    def set_log_stream(log_stream):
        Log.log_stream = log_stream

