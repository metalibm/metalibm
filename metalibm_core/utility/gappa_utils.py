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
# created:          Apr  7th, 2014
# last-modified:    Mar  7th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import re
import subprocess
import sys

import sollya

def parse_gappa_interval(interval_value):
    # search for middle ","
    end_index = len(interval_value)
    tmp_str = re.sub("[ \[\]]", lambda _: "", interval_value)
    while "{" in tmp_str:
        start = tmp_str.index("{")
        end = tmp_str.index("}")
        tmp_str = tmp_str[:start] + tmp_str[end+1:]
    v0, v1 = tmp_str.split(",")
    return sollya.Interval(sollya.parse(v0), sollya.parse(v1))


def execute_gappa_script_extract(gappa_code, gappa_filename = "gappa_tmp.g"):
    result = {}
    gappa_stream = open(gappa_filename, "w")
    gappa_stream.write(gappa_code)
    gappa_stream.close()
    gappa_cmd = "gappa {}".format(gappa_filename)
    cmd_result = subprocess.check_output(
        gappa_cmd, stderr=subprocess.STDOUT, shell=True)
    if sys.version_info >= (3, 0):
        gappa_result = str(cmd_result, 'utf-8')
    else:
        gappa_result = str(cmd_result)
    start_result_index = gappa_result.index("Results")
    for result_line in gappa_result[start_result_index:].splitlines()[1:]:
        if not " in " in result_line: continue
        result_split = result_line.split(" in ")
        var = result_split[0].replace(" ", "")
        interval_value = result_split[1].replace(" ", "")
        result[var] = parse_gappa_interval(interval_value)
    return result

DISABLE_GAPPA = False

## Check if gappa binary is available in the execution environement
def is_gappa_installed():
    if DISABLE_GAPPA:
        return False
    """ check if gappa is present on the execution environement """
    dev_null = open("/dev/null", "w")
    gappa_test = subprocess.call("gappa --help 2> /dev/null", shell=True)
    return (gappa_test == 0)
