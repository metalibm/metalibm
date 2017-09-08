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
        """ offset is given from MSB downward.
            The node returns the index of position (MSB - offset) from LSB
        """
        pass
    class FromLSBToLSB:
        """ offset is given from LSB upward.
            The node returns the position of index (LSB + offset) from LSB
            (i.e result = offset)
        """
        pass
    class FromPointToLSB:
        """ The offset is given from point position upward.
            The node returns the position of index (point + offset) from LSB
        """
        pass
    class FromPointToMSB:
        """ The offset is given from point position upward.
            The node returns the position of (point + offset) from MSB.
            The result is expected to be negative
        """
        pass
    def __init__(self, op, position, align = FromLSBToLSB, **kwords):
        self.__class__.__base__.__init__(self, op, position, **kwords)
        self.align = align

    def get_align(self):
        return self.align

    def finish_copy(self, new_copy, copy_map = None):
        new_copy.align = self.align
