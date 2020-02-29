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
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.precisions import (
    ML_CorrectlyRounded, ML_Faithful, ML_DegradedAccuracyRelative, dar,
    ML_DegradedAccuracyAbsolute
)

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)
from metalibm_core.utility.debug_utils import debug_multi


from metalibm_core.code_generation.generic_processor import GenericProcessor


class ML_UT_Accuracies(ML_Function("ml_ut_accuracies")):
    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_accuracies.c",
            "function_name": "ut_accuracies",
            "precision": ML_Binary32,
            "target": GenericProcessor.get_target_instance(),
            "fast_path_extract": True,
            "fuse_fma": True,
            "debug": True,
            "libm_compliant": True,
            "test_range": Interval(S2**-8, S2**8),
            "accuracy": dar(S2**-7),
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        # declaring function input variable
        vx = self.implementation.add_input_variable("x", self.precision)

        approx = Division(1, vx, tag="division")
        if isinstance(self.accuracy, ML_CorrectlyRounded):
            pass
        elif isinstance(self.accuracy, ML_Faithful):
            approx = approx + S2**-self.precision.get_mantissa_size() * approx
        elif isinstance(self.accuracy, ML_DegradedAccuracyRelative):
            approx = approx + approx * self.accuracy.goal * 0.5
        elif isinstance(self.accuracy, ML_DegradedAccuracyAbsolute):
            approx = approx + self.accuracy.goal * 0.5
        else:
            raise NotImplementedError

        result = approx

        scheme = Return(result, precision=self.precision, debug=debug_multi)

        return scheme

    def numeric_emulate(self, x):
        """ numeric emulation """
        result = self.precision.round_sollya_object(1 / x)
        return result


def run_test(args):
  ml_ut_accuracies = ML_UT_Accuracies(args)
  ml_ut_accuracies.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  ARG_TEMPLATE = ML_NewArgTemplate(default_arg=ML_UT_Accuracies.get_default_args())
  args = ARG_TEMPLATE.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


