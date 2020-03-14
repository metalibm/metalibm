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
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
# Instances (see valid/unit_test.py
# 1.  --pre-gen-passes m128_promotion --target x86_avx2
#

import sys

import sollya
from sollya import Interval
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.ml_operations import (
    ReciprocalSeed, Return
)
from metalibm_core.core.ml_formats import ML_Binary32
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.precisions import dar

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_core.targets.intel.x86_processor import X86_AVX2_Processor

from metalibm_core.code_generation.generic_processor import GenericProcessor


class ML_UT_LegalizeReciprocalSeed(ML_Function("ml_ut_legalize_rcp")):
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_legalize_rcp.c",
            "function_name": "ut_legalize_rcp",
            "precision": ML_Binary32,
            "target": GenericProcessor.get_target_instance(),
            "fast_path_extract": True,
            "fuse_fma": True,
            "debug": True,
            "libm_compliant": True,
            "auto_test_range": Interval(S2**-8, S2**8),
            "accuracy": dar(S2**-6),
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        # declaring function input variable
        vx = self.implementation.add_input_variable("x", self.precision)

        approx = ReciprocalSeed(
            vx, precision=self.precision,
            tag = "approx"
        )

        result = approx

        scheme = Return(result, precision=self.precision, debug=debug_multi)


        return scheme

    def numeric_emulate(self, x):
        """ numeric emulation """
        # extracting mantissa from x
        # abs_x = abs(x)
        # mantissa = abs_x / S2**sollya.floor(sollya.log2(abs_x))
        # index = sollya.floor((mantissa - 1.0) * 2**8)
        # result = sollya.round(1/sollya.sqrt(1.0 + index * S2**-8), 9, sollya.RN)
        if x == 0:
            return sollya.parse("infty")
        result = sollya.round(1.0/x, 9, sollya.RN)
        return result


def run_test(args):
  ml_ut_legalize_rcp = ML_UT_LegalizeReciprocalSeed(args)
  ml_ut_legalize_rcp.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_LegalizeReciprocalSeed.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


