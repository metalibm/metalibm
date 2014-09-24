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

from ..utility.common import Callable

class TargetRegister:
    target_map = {}

    def get_target_name_list():
        for target_name in TargetRegister.target_map:
            print target_name

    def get_target_by_name(target_name):
        return TargetRegister.target_map[target_name]
            

    def register_new_target(target_name, target_build_function):
        TargetRegister.target_map[target_name] = target_build_function

    get_target_name_list = Callable(get_target_name_list)
    get_target_by_name   = Callable(get_target_by_name)
    register_new_target  = Callable(register_new_target)
