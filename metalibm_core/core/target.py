# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2014)
# All rights reserved
# created:          Sep 23rd,  2014
# last-modified:    Sep 23rd,  2014
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

class TargetRegister(object):
    target_map = {}

    @staticmethod
    def get_target_name_list():
        for target_name in TargetRegister.target_map:
            print target_name

    @staticmethod
    def get_target_by_name(target_name):
        return TargetRegister.target_map[target_name]

    @staticmethod
    def register_new_target(target_name, target_build_function):
        TargetRegister.target_map[target_name] = target_build_function
