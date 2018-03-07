# -*- coding: utf-8 -*-

## @package advanced_operations
#  Metalibm Description Language advanced Operations

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

###############################################################################
# This file is part of the new Metalibm tool
# created:          Aug  9th, 2017
# last-modified:    Mar  8th, 2018
#
# author(s): Nicolas Brunie (nbrunie@kalray.eu)
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
