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
# created:          Dec 27th, 2013
# last-modified:    Mar  7th, 2018
#
# Author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


import sys
import pdb

class Log(object):
    """ log report class """
    log_stream     = None
    dump_stdout    = False
    ## disable display of Exception backtrace when Log.Error level message
    #  is display
    exit_on_error  = False
    ## Tribber PDB break when an Error level message is reported
    break_on_error = False

    @staticmethod
    def set_dump_stdout(new_dump_stdout):
      Log.dump_stdout = new_dump_stdout

    @staticmethod
    def set_break_on_error(value):
      print("setting break on error ", value)
      Log.break_on_error = value

    class LogLevel(object):
        """ log level builder """
        def __init__(self, level_name, sub_level=None):
            self.name = level_name
            self.sub_level = sub_level
    class LogLevelFilter(LogLevel):
        """ filtering log message """
        def match(self, tested_level):
            if tested_level.name != self.name:
                return False
            elif self.sub_level is None or self.sub_level == tested_level.sub_level:
                return True
            else:
                return False

    # log levels definition
    Warning = LogLevelFilter("Warning")
    Info    = LogLevelFilter("Info")
    Error   = LogLevelFilter("Error")
    Debug   = LogLevelFilter("Debug")
    Verbose = LogLevelFilter("Verbose")

    # list of enabled log levels
    enabled_levels = [
        # Warning,
        Error,
        # LogLevelFilter("Info", "passes")
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
    def is_level_enabled(level):
        return Log.filter_log_level(Log.enabled_levels, level)

    @staticmethod
    def report_custom(level, msg, eol="\n", error=None):
        return Log.report(level, msg + eol, error=error)

    @staticmethod
    def report(level, msg, *args, **kw):
        error = kw.pop("error", None)
        """ report log message """
        if Log.filter_log_level(Log.enabled_levels, level):
            if Log.log_stream:
                Log.log_stream.write(msg.format(*args, **kw))
            if Log.dump_stdout:
                print("%s: %s" % (level.name, msg.format(*args, **kw)))
        if level is Log.Error:
            if Log.break_on_error:
              pdb.set_trace()
              raise error or Exception(msg.format(*args, **kw))
            elif Log.exit_on_error:
              sys.exit(1)
            else:
              raise error or Exception(msg.format(*args, **kw))

    ## enable display of the specific log level
    #  @param level log-level to be enabled
    @staticmethod
    def enable_level(level, sub_level=None):
      Log.enabled_levels.append(Log.LogLevelFilter(level,sub_level))

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

