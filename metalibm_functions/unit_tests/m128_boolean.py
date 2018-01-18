# -*- coding: utf-8 -*-

# Instances (see valid/unit_test.py
# 1.  --pre-gen-passes m128_promotion --target x86_avx2
# 

import sys

import sollya
from sollya import S2, Interval

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.code_constant import C_Code 

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

from metalibm_core.targets.intel.x86_processor import X86_AVX2_Processor


class ML_UT_M128Boolean(ML_Function("ml_ut_m128_boolean")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_m128_boolean.c",
        "function_name": "ut_m128_boolean",
        "precision": ML_Binary32,
        "target": X86_AVX2_Processor(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True,
        "pre_gen_pass": ["m128_promotion"], 
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_scheme(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)

    cond = Comparison(vx, Constant(7, precision=self.precision), specifier=Comparison.NotEqual)

    result = Select(cond,
        Constant(0, precision=self.precision),
        Constant(-1, precision=self.precision),
        precision=ML_Int32
    )


    scheme = Return(result, precision=self.precision, debug=debug_multi)

    # conv_pass = Pass_M128_Promotion(self.processor)
    # new_scheme = conv_pass.execute(scheme)

    return scheme

  def numeric_emulate(self, x):
    if x != 7:
        return 0
    else:
        return -1


def run_test(args):
  ml_ut_m128_conversion = ML_UT_M128Boolean(args)
  ml_ut_m128_conversion.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_M128Boolean.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


