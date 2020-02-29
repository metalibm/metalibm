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

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.code_constant import C_Code 

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

from metalibm_core.targets.intel.x86_processor import X86_AVX2_Processor


class ML_UT_M128Conversion(ML_Function("ml_ut_m128_conversion")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_m128_conversion.c",
        "function_name": "ut_m128_conversion",
        "precision": ML_Binary32,
        "target": X86_AVX2_Processor.get_target_instance(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True,
        "pre_gen_passes": ["m128_promotion"],
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)

    add_xx = Addition(vx, vx, precision = self.precision)
    mult = Multiplication(add_xx, vx, precision = self.precision)
    cst  = Constant(1.1, precision = self.precision)

    index_size = 4
    table_size = 2**index_size

    table = ML_NewTable(
      dimensions = [table_size],
      storage_precision = self.precision
    )
    for i in range(table_size):
      table[i] = i

    index = NearestInteger(
      vx,
      precision = ML_Int32
    )
    # index = index % table_size = index & (2**index_size - 1)
    index = BitLogicAnd(
      index,
      Constant(2**index_size - 1, precision = ML_Int32),
      precision = ML_Int32
    )

    index = BitLogicRightShift(index, Constant(1, precision=ML_Int32), precision=ML_Int32)

    table_value = TableLoad(table, index, precision = self.precision)

    int_tree = Multiplication(
        index,
        Addition(
            index,
            Constant(7, precision = ML_Int32),
            precision = ML_Int32
        ),
        precision = ML_Int32
    )

    result = Multiplication(
      table_value,
      FusedMultiplyAdd(
        Addition(
            cst,
            Conversion(int_tree, precision = self.precision),
            precision = self.precision
        ),
        mult,
        add_xx,
        specifier = FusedMultiplyAdd.Subtract,
        precision = self.precision
      ),
      precision = self.precision,
      tag="result"
    )

    scheme = Return(result, precision=self.precision, debug=debug_multi)

    # conv_pass = Pass_M128_Promotion(self.processor)
    # new_scheme = conv_pass.execute(scheme)

    return scheme

  def numeric_emulate(self, x):
    index = int(sollya.nearestint(x)) % 16
    table_value = index >> 1
    add_xx = sollya.round(x + x, self.precision.get_sollya_object(), sollya.RN)
    mult   = sollya.round(add_xx * x, self.precision.get_sollya_object(), sollya.RN)
    cst    = sollya.round(1.1, self.precision.get_sollya_object(), sollya.RN)
    return sollya.round(
      table_value * 
        sollya.round(
          sollya.round(cst * mult, self.precision.get_sollya_object(), sollya.RN) - add_xx , self.precision.get_sollya_object(), sollya.RN), self.precision.get_sollya_object(), sollya.RN
    )


def run_test(args):
  ml_ut_m128_conversion = ML_UT_M128Conversion(args)
  ml_ut_m128_conversion.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_M128Conversion.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


