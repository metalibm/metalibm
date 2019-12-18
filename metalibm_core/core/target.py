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
# created:          Sep 23rd,  2014
# last-modified:    Mar  7th,  2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################



class TargetRegister(object):
    """ target register """
    target_map = {}

    @staticmethod
    def get_target_name_list():
        for target_name in TargetRegister.target_map:
            print(target_name)

    @staticmethod
    def get_target_by_name(target_name):
        return TargetRegister.target_map[target_name]

    @staticmethod
    def register_new_target(target_name, target_build_function):
        TargetRegister.target_map[target_name] = target_build_function

    @staticmethod
    def METALIBM_TARGET_REGISTER(target_class):
        """ decorator to automate target class registerting """
        TargetRegister.register_new_target(target_class.target_name, lambda _: target_class)
        return target_class
