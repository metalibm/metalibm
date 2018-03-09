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

import sys
from .log_report import Log

def test_flag_option(flag_name, flag_value, default_value, parse_arg = None, help_map = None, help_str = ""):
    if help_map != None: 
        help_map[flag_name] = "[yes=%s|no=%s] %s" % (flag_value, default_value, help_str)

    if flag_name in sys.argv and parse_arg: 
        parse_arg.append(sys.argv.index(flag_name)) 
    return flag_value if flag_name in sys.argv else default_value

def extract_option_value(option_name, default_value, help_map = None, help_str = "", processing = lambda x: x, parse_arg = None):
    if help_map != None: 
        help_map[option_name] = "[%s] %s" % (default_value, help_str)
    if option_name in  sys.argv:
        option_index = sys.argv.index(option_name)
        if option_index + 1 >= len(sys.argv):
            Log.report(Log.Error, "missing value for option argument: %s" % option_name)
        elif parse_arg:
            parse_arg.append(option_index)
            parse_arg.append(option_index+1)
    return processing(sys.argv[sys.argv.index(option_name)+1] if option_name in sys.argv else default_value)

def extract_option_list_value(option_name, default_value):
    return sys.argv[sys.arg.index(option_name+1)].split(",") if option_name in sys.argv else default_value
