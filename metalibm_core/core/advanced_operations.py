# -*- coding: utf-8 -*-

## @package advanced_operations
#  Metalibm Description Language advanced Operations

###############################################################################
# This file is part of the new Metalibm tool
# Copyright (2017-)
# All rights reserved
# created:          Aug  9th, 2017
# last-modified:    Aug  9th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################

from metalibm_core.core.ml_operations import (
    ArithmeticOperationConstructor, SpecifierOperation, empty_range
)


class FixedPointPosition(
    ArithmeticOperationConstructor("FixedPointPosition",
    range_function = empty_range)
    ):
    """ Dynamic FixedPointPosition  evaluator node
        convert to a constant during code generation, once input
        format has been determined """
    class FromMSBToLSB:
        """ align position to the input most significant bit downward """
        pass
    class FromLSBToLSB:
        """ align position to the input least significant bit upward """
        pass
    class FromPointToLSB:
        """ align position to the input fixed-point upward """
        pass
    def __init__(self, op, position, align = FromLSBToLSB, **kwords):
        self.__class__.__base__.__init__(self, op, position, **kwords)
        self.align = align

    def get_align(self):
        return self.align

    def finish_copy(self, new_copy, copy_map = None):
        new_copy.align = self.align
