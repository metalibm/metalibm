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
from metalibm_core.targets.intel.p_vec_registers import Pass_M128_Promotion




class ML_UT_M128Conversion(ML_Function("ml_ut_m128_conversion")):
  def __init__(self, 
                 arg_template,
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = X86_AVX2_Processor(), 
                 output_file = "ut_m128_conversion.c", 
                 function_name = "ut_m128_conversion"):
    # precision argument extraction
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "ut_m128_conversion",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template
    )

    self.precision = precision


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
    for i in xrange(table_size):
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

    table_value = TableLoad(table, index, precision = self.precision)


    result = Multiplication(
      table_value,
      FusedMultiplyAdd(cst, mult, add_xx, specifier = FusedMultiplyAdd.Subtract, precision = self.precision),
      precision = self.precision
    )

    scheme = Return(result, precision = self.precision)

    # conv_pass = Pass_M128_Promotion(self.processor)
    # new_scheme = conv_pass.execute(scheme)

    return scheme

  def numeric_emulate(self, x):
    index = int(sollya.nearestint(x)) % 16
    table_value = index
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
  arg_template = ML_NewArgTemplate("new_ut_m128_conversion", default_output_file = "new_ut_m128_conversion.c" )
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


