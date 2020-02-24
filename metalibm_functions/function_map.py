# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          Feb 22nd, 2020
# last-modified:    Feb 22nd, 2020
###############################################################################

# TODO/FIXME: implement cleaner way to register and list meta-functions
from metalibm_functions.ml_exp import ML_Exponential
from metalibm_functions.ml_tanh import ML_HyperbolicTangent
from metalibm_functions.ml_sqrt import MetalibmSqrt

# dict of (str) -> tuple(ctor, dict(ML_Format -> str))
# the first level key is the function name
# the first value of value tuple is the meta-function constructor
# the second value of the value tuple is a dict which associates to a ML_Format
# the corresponding libm function
FUNCTION_MAP = {
    "exp": ML_Exponential,
    "tanh": ML_HyperbolicTangent,
    "sqrt": MetalibmSqrt,
}
