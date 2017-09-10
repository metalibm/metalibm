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
import pdb

class Log(object):
    """ log report class """
    log_stream     = None
    dump_stdout    = False
    ## abort execution when an Error level message is reported
    exit_on_error  = True
    ## Tribber PDB break when an Error level message is reported
    break_on_error = False

    @staticmethod
    def set_dump_stdout(new_dump_stdout):
      Log.dump_stdout = new_dump_stdout

    @staticmethod
    def set_break_on_error(value):
      print "setting break on error ", value
      Log.break_on_error = value

    class LogLevel(object):
        """ log level builder """
        def __init__(self, level_name, sub_level=None):
            self.name = level_name
            self.sub_level = sub_level
    class LogLevelFilter(LogLevel):
        """ filtering log message """
        def match(self, tested_level):
            if not tested_level.name == self.name:
                return False
            elif self.sub_level is None or self.sub_level == tested_level.name:
                return True
            else:
                return False

    # log levels definition
    Warning = LogLevel("Warning")
    Info    = LogLevel("Info")
    Error   = LogLevel("Error")
    Debug   = LogLevel("Debug")
    Verbose = LogLevel("Verbose")

    # list of enabled log levels
    enabled_levels = [
        LogLevelFilter(Warning),
        LogLevelFilter(Error)
    ]

    @staticmethod
    def filter_log_level(filter_list, log_level):
        """ Test if log_level matches one of the filters listed in
            filter_list """
        for log_filter in filter_list:
            if log_filter.match(log_level):
                return True
        return False

    @staticmethod
    def report(level, msg, eol = "\n"):
        """ report log message """
        if Log.log_stream:
            Log.log_stream.write(msg + eol)
            if Log.dump_stdout:
              print "%s: %s" % (level.name, msg)
        elif Log.filter_log_level(Log.enabled_levels, level):
            print "%s: %s" % (level.name, msg)
        if level is Log.Error:
            if Log.break_on_error:
              pdb.set_trace()
              raise Exception()
            elif Log.exit_on_error:
              sys.exit(1)
            else:
              raise Exception()

    ## enable display of the specific log level
    #  @param level log-level to be enabled
    @staticmethod
    def enable_level(level, sub_level=None):
      Log.enabled_levels.append(LogLevelFilter(level,sub_level))

    ## disable display of the specific log level
    #  @param level log-level to be disabled
    @staticmethod
    def disable_level(level):
        raise NotImplementedError
        # does not match new LogLevelFilter mechanisms
        # Log.enabled_levels.remove(level)

    @staticmethod
    def set_log_stream(log_stream):
        Log.log_stream = log_stream

