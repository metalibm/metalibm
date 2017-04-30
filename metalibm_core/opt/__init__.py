# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Metalibm tool
# Copyright (201&)
# All rights reserved
# created:          Apr 29th, 2017
# last-modified:    Apr 29th, 2017
#
# author(s):     Nicolas Brunie (nibrunie@gmail.com)
# desciprition:  Auto-Load module for Metalibm's optimization pass
###############################################################################
import os
import re


def pass_validity_test(pass_name):
    return re.match("p_[\w]+\.py", pass_name) != None

# dynamically search for installed targets
pass_dirname = os.path.dirname(os.path.realpath(__file__))

pass_list = [possible_pass for possible_pass in os.listdir(pass_dirname) if pass_validity_test(possible_pass)]
    
__all__ = pass_list

# listing submodule

if __name__ == "__main__":
    print "pass_list: ", pass_list
