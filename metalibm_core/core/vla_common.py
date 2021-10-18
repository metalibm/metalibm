# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:              Oct  3rd, 2021
# last-modified:        Oct  3rd, 2021
#
# author(s):    Nicolas Brunie
# desciprition: Vector Length Agnostic definitions
###############################################################################

from metalibm_core.core.ml_operations import (
    GeneralOperation, ML_ArithmeticOperation, SpecifierOperation, Statement)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_Format, ML_Int32, ML_UInt32, ML_UInt64, ML_Int64)

class VLAType(ML_Format):
    """ wrapper for vector length agnostic types """
    def __init__(self, baseFormat, groupSize=1):
        self.baseFormat = baseFormat
        self.groupSize = groupSize

    @staticmethod
    def isVLAType(t):
        return isinstance(t, VLAType)

class VLAOperation(SpecifierOperation, ML_ArithmeticOperation):
    arity = None
    def __init__(self, *args, specifier=None, **kw):
        ML_ArithmeticOperation.__init__(self, *args, **kw)
        self.specifier = specifier

    def finish_copy(self, new_copy, copy_map = {}):
        new_copy.specifier = self.specifier

    @property
    def name(self):
        return "VLAOperation.{}".format(self.specifier)


class VLAGetLength(ML_ArithmeticOperation):
    name = "VLAGetLength"
    arity = 1


VLA_FORMAT_MAP = {(eltType, lmul): VLAType(eltType, lmul) 
                   for lmul in [1, 2, 4, 8] 
                   for eltType in [ML_Binary32, ML_Binary64, ML_Int32, ML_Int64, ML_UInt32, ML_UInt64]}

VLA_Binary32_l1 = VLA_FORMAT_MAP[(ML_Binary32, 1)]


class VLAOp(ML_ArithmeticOperation):
    """ wrapper for VLA operation """
    arity = None
    def __init__(self, vlaOp, vl, **kw):
        self.vlaOp = vlaOp
        self.vl = vl
        GeneralOperation.__init__(self, *tuple(vlaOp.get_inputs()), vl, **kw)

    def get_codegen_key(self):
        return self.vlaOp.__class__

class VLABlock(GeneralOperation):
    """ Block with same agnostic vector length """
    arity = 2

