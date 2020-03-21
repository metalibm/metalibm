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

import sollya
import bigfloat

# TODO/FIXME: implement cleaner way to register and list meta-functions
import metalibm_functions.ml_log10
import metalibm_functions.ml_log1p
import metalibm_functions.ml_log2
import metalibm_functions.ml_log
import metalibm_functions.ml_exp
import metalibm_functions.ml_expm1
import metalibm_functions.ml_exp2
import metalibm_functions.ml_cbrt
import metalibm_functions.ml_sqrt
import metalibm_functions.ml_isqrt
import metalibm_functions.ml_cosh
import metalibm_functions.ml_sinh
import metalibm_functions.ml_sincos
import metalibm_functions.ml_atan
import metalibm_functions.ml_tanh
import metalibm_functions.ml_div
import metalibm_functions.generic_log
import metalibm_functions.erf
import metalibm_functions.ml_acos
import metalibm_functions.fmod
import metalibm_functions.ml_atan

from sollya_extra_functions import cbrt


S2 = sollya.SollyaObject(2)

# dict of (str) -> tuple(ctor, dict(ML_Format -> str))
# the first level key is the function name
# the first value of value tuple is the meta-function constructor
# the second value of the value tuple is a dict which associates to a ML_Format
# the corresponding libm function
FUNCTION_MAP = {
    "exp": (metalibm_functions.ml_exp.ML_Exponential, {}, sollya.exp),
    "tanh": (metalibm_functions.ml_tanh.ML_HyperbolicTangent, {}, sollya.tanh),
    "sqrt": (metalibm_functions.ml_sqrt.MetalibmSqrt, {}, sollya.sqrt),
    "log": (metalibm_functions.generic_log.ML_GenericLog, {"basis": sollya.exp(1)}, sollya.log),
    "log2": (metalibm_functions.generic_log.ML_GenericLog, {"basis": 2}, sollya.log2),
    "log10": (metalibm_functions.generic_log.ML_GenericLog, {"basis": 10}, sollya.log10),
    "exp2": (metalibm_functions.ml_exp2.ML_Exp2, {}, (lambda x: S2**x)),

    "div": (metalibm_functions.ml_div.ML_Division, {}, (lambda x,y: x / y)),
    "cbrt": (metalibm_functions.ml_cbrt.ML_Cbrt, {}, cbrt),

    "cosh": (metalibm_functions.ml_cosh.ML_HyperbolicCosine, {}, sollya.cosh),
    "sinh": (metalibm_functions.ml_sinh.ML_HyperbolicSine, {}, sollya.sinh),

    "cos": (metalibm_functions.ml_sincos.ML_SinCos, {"sin_output": False}, sollya.cos),
    "sin": (metalibm_functions.ml_sincos.ML_SinCos, {"sin_output": True}, sollya.sin),
    "atan": (metalibm_functions.ml_atan.MetaAtan, {}, sollya.atan),

    "erf": (metalibm_functions.erf.ML_Erf, {}, sollya.erf),
    "fmod": (metalibm_functions.fmod.MetaFMOD, {}, bigfloat.fmod),
}
