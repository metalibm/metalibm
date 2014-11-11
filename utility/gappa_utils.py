# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013)
# All rights reserved
# created:          Apr  7th, 2014
# last-modified:    Apr  7th, 2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from pythonsollya import *
import commands
import re
import subprocess

def parse_gappa_interval(interval_value):
    # search for middle ","
    end_index = len(interval_value)
    tmp_str = re.sub("[ \[\]]", lambda _: "", interval_value)
    while "{" in tmp_str:
        start = tmp_str.index("{")
        end = tmp_str.index("}")
        tmp_str = tmp_str[:start] + tmp_str[end+1:]
    v0, v1 = tmp_str.split(",")
    return Interval(SollyaObject(v0), SollyaObject(v1))


def execute_gappa_script_extract(gappa_code, gappa_filename = "gappa_tmp.g"):
    result = {}
    gappa_stream = open(gappa_filename, "w")
    gappa_stream.write(gappa_code)
    gappa_stream.close()
    gappa_result = commands.getoutput("gappa %s" % gappa_filename)
    start_result_index = gappa_result.index("Results")
    for result_line in gappa_result[start_result_index:].splitlines()[1:]:
        if not " in " in result_line: continue
        result_split = result_line.split(" in ")
        var = result_split[0].replace(" ", "")
        interval_value = result_split[1].replace(" ", "")
        result[var] = parse_gappa_interval(interval_value)
    return result
    

def is_gappa_installed():
	""" check if gappa is present on the execution environement """
	gappa_test = subprocess.call("gappa", shell=True)
	return (gappa_test == 0)
