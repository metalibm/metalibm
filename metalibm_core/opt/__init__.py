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
# created:          Apr 29th, 2017
# last-modified:    Mar  7th, 2018
#
# author(s):     Nicolas Brunie (nbrunie@kalray.eu)
# desciprition:  Auto-Load module for Metalibm's optimization pass
###############################################################################
import os
import re


## check if @p pass_name is a valid filename
#  for a path description file
def pass_validity_test(pass_name):
  return re.match("p_[\w]+\.py$", pass_name) != None

## build the pass module name from the filename
#  @pass_name
def get_module_name(pass_name):
	return pass_name.replace(".py", "") 

# dynamically search for installed targets
pass_dirname = os.path.dirname(os.path.realpath(__file__))

pass_list = [get_module_name(possible_pass) for possible_pass in os.listdir(pass_dirname) if pass_validity_test(possible_pass)]
    
__all__ = pass_list

# listing submodule

if __name__ == "__main__":
    print("pass_list: ", pass_list)
