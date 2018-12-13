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
import sys

import sollya

from sollya import (
        Interval, ceil, floor, round, inf, sup, pi, log, atan,
        guessdegree, dirtyinfnorm
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.special_values import FP_QNaN
from metalibm_core.core.precisions import ML_Faithful, ML_CorrectlyRounded
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import ML_NewArgTemplate, ArgDefault
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

# Disabling Sollya's rounding warnings
sollya.roundingwarnings = sollya.off
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on


class ML_Atan(ML_Function("atan")):
  def __init__(self, arg_template=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, arg_template) 

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Atan,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_exp = {
        "output_file": "my_atan.c",
        "function_name": "my_atan",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_exp.update(kw)
    return DefaultArgTemplate(**default_args_exp)
    
    
  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_atan"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self):
    
    def compute_reciprocal(vx):
      inv_seed = ReciprocalSeed(vx, precision = self.precision, tag = "inv_seed", debug = debug_multi)
      nr_1 = 2*inv_seed - vx*inv_seed*inv_seed
      nr_2 = 2*nr_1 - vx*nr_1*nr_1
      nr_3 =2*nr_2 - vx*nr_2*nr_2
      inv_vx = 2*nr_3 - vx*nr_3*nr_3
      
      return inv_vx
      
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    sollya_precision = self.precision.get_sollya_object()
    
    int_precision = {
        ML_Binary32 : ML_Int32,
        ML_Binary64 : ML_Int64
      }[self.precision]
    
    hi_precision = self.precision.get_field_size() - 12
    
    half_pi = round(pi/2, sollya_precision, sollya.RN)
    half_pi_cst = Constant(half_pi, precision = self.precision)
    
    test_sign = Comparison(vx, 0, specifier = Comparison.Less, precision = ML_Bool, debug = debug_multi, tag = "Is_Negative")
    neg_vx = -vx
    
    sign = Variable("sign", precision = self.precision, var_type = Variable.Local)
    abs_vx_std = Variable("abs_vx", precision = self.precision, var_type = Variable.Local)
    red_vx_std = Variable("red_vx", precision = self.precision, var_type = Variable.Local)
    const_index_std = Variable("const_index", precision = int_precision, var_type = Variable.Local)
    
    set_sign = Statement(
        ConditionBlock(test_sign,
          Statement(ReferenceAssign(abs_vx_std, neg_vx), ReferenceAssign(sign, -1)),
          Statement(ReferenceAssign(abs_vx_std, vx), ReferenceAssign(sign, 1))
      ))
      
    if self.precision is ML_Binary32:
      bound = 24
    else:
      bound = 53
      
    test_bound = Comparison(abs_vx_std, S2**bound, specifier = Comparison.GreaterOrEqual, precision = ML_Bool)#, debug = debug_multi, tag ="bound")
    test_bound1 = Comparison(abs_vx_std, 39.0/16.0, specifier = Comparison.GreaterOrEqual, precision = ML_Bool)#, debug = debug_multi, tag ="bound")
    test_bound2 = Comparison(abs_vx_std, 19.0/16.0, specifier = Comparison.GreaterOrEqual, precision = ML_Bool)#, debug = debug_multi, tag ="bound")
    test_bound3 = Comparison(abs_vx_std, 11.0/16.0, specifier = Comparison.GreaterOrEqual, precision = ML_Bool)#, debug = debug_multi, tag ="bound")
    test_bound4 = Comparison(abs_vx_std, 7.0/16.0, specifier = Comparison.GreaterOrEqual, precision = ML_Bool)#, debug = debug_multi, tag ="bound")
    
    
    
    set_bound = Return(sign*half_pi_cst)
    
    set_bound1 = Statement(
      ReferenceAssign(red_vx_std, -compute_reciprocal(abs_vx_std)),
      ReferenceAssign(const_index_std, 3)
    )
    
    set_bound2 = Statement(
      ReferenceAssign(red_vx_std, (abs_vx_std - 1.5)*compute_reciprocal(1 + 1.5*abs_vx_std)),
      ReferenceAssign(const_index_std, 2)
    )
    
    set_bound3 = Statement(
      ReferenceAssign(red_vx_std, (abs_vx_std - 1.0)*compute_reciprocal(abs_vx_std + 1.0)),
      ReferenceAssign(const_index_std, 1)
    )
    
    set_bound4 = Statement(
      ReferenceAssign(red_vx_std, (abs_vx_std - 0.5)*compute_reciprocal(1 + abs_vx_std*0.5)),
      ReferenceAssign(const_index_std, 0)
    )
    
    set_bound5 = Statement(
      ReferenceAssign(red_vx_std, abs_vx_std),
      ReferenceAssign(const_index_std, 4)
    )
    
    
    cons_table = ML_NewTable(dimensions = [5, 2], storage_precision = self.precision, tag = self.uniquify_name("cons_table"))
    coeff_table = ML_NewTable(dimensions = [11], storage_precision = self.precision, tag = self.uniquify_name("coeff_table"))
    
    cons_hi = round(atan(0.5), hi_precision, sollya.RN)
    cons_table[0][0] = cons_hi
    cons_table[0][1] = round(atan(0.5) - cons_hi, sollya_precision, sollya.RN)
    
    cons_hi = round(atan(1.0), hi_precision, sollya.RN)
    cons_table[1][0] = cons_hi
    cons_table[1][1] = round(atan(1.0) - cons_hi, sollya_precision, sollya.RN)
    
    cons_hi = round(atan(1.5), hi_precision, sollya.RN)
    cons_table[2][0] = cons_hi
    cons_table[2][1] = round(atan(1.5) - cons_hi, sollya_precision, sollya.RN)
    
    cons_hi = round(pi/2, hi_precision, sollya.RN)
    cons_table[3][0] = cons_hi
    cons_table[3][1] = round(pi/2 - cons_hi, sollya_precision, sollya.RN)
    
    cons_table[4][0] = 0.0
    cons_table[4][1] = 0.0
    
    coeff_table[0] = round(3.33333333333329318027e-01, sollya_precision, sollya.RN)
    coeff_table[1] = round(-1.99999999998764832476e-01, sollya_precision, sollya.RN)
    coeff_table[2] = round(1.42857142725034663711e-01, sollya_precision, sollya.RN)
    coeff_table[3] = round(-1.11111104054623557880e-01, sollya_precision, sollya.RN)
    coeff_table[4] = round(9.09088713343650656196e-02, sollya_precision, sollya.RN)
    coeff_table[5] = round(-7.69187620504482999495e-02, sollya_precision, sollya.RN)
    coeff_table[6] = round(6.66107313738753120669e-02, sollya_precision, sollya.RN)
    coeff_table[7] = round(-5.83357013379057348645e-02, sollya_precision, sollya.RN)
    coeff_table[8] = round(4.97687799461593236017e-02, sollya_precision, sollya.RN)
    coeff_table[9] = round(-3.65315727442169155270e-02, sollya_precision, sollya.RN)
    coeff_table[10] = round(1.62858201153657823623e-02, sollya_precision, sollya.RN)
    
    red_vx2 = red_vx_std*red_vx_std
    red_vx4 = red_vx2*red_vx2
    a0 = TableLoad(coeff_table, 0, precision = self.precision)
    a1 = TableLoad(coeff_table, 1, precision = self.precision)
    a2 = TableLoad(coeff_table, 2, precision = self.precision)
    a3 = TableLoad(coeff_table, 3, precision = self.precision)
    a4 = TableLoad(coeff_table, 4, precision = self.precision)
    a5 = TableLoad(coeff_table, 5, precision = self.precision)
    a6 = TableLoad(coeff_table, 6, precision = self.precision)
    a7 = TableLoad(coeff_table, 7, precision = self.precision)
    a8 = TableLoad(coeff_table, 8, precision = self.precision)
    a9 = TableLoad(coeff_table, 9, precision = self.precision)
    a10 = TableLoad(coeff_table, 10, precision = self.precision)
    
    poly_even = red_vx2*(a0 + red_vx4*(a2 + red_vx4*(a4 + red_vx4*(a6 + red_vx4*(a8 + red_vx4*a10)))))
    poly_odd = red_vx4*(a1 + red_vx4*(a3 + red_vx4*(a5 + red_vx4*(a7 + red_vx4*a9))))
    
    
    poly_even.set_attributes(tag = "poly_even", debug = debug_multi)
    poly_odd.set_attributes(tag = "poly_odd", debug = debug_multi)
    
    const_load_hi = TableLoad(cons_table, const_index_std, 0, tag = "const_load_hi", debug = debug_multi)
    const_load_lo = TableLoad(cons_table, const_index_std, 1, tag = "const_load_lo", debug = debug_multi)
    
    test_NaN_or_inf = Test(vx, specifier = Test.IsInfOrNaN, tag = "nan_or_inf", likely = False)
    test_nan = Test(vx, specifier = Test.IsNaN, debug = debug_multi, tag = "is_nan_test", likely = False)
    test_positive = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = debug_multi, tag = "inf_sign", likely = False)
                
        
    result = const_load_hi - ((red_vx_std*(poly_even + poly_odd) - const_load_lo) - red_vx_std)
    result.set_attributes(tag = "result", debug = debug_multi)
    
    std_scheme = Statement(
          sign,
          abs_vx_std,
          red_vx_std,
          const_index_std,
          set_sign,
          ConditionBlock(
            test_bound,
            set_bound,
            ConditionBlock(
              test_bound1,
              set_bound1,
              ConditionBlock(
                test_bound2,
                set_bound2,
                ConditionBlock(
                  test_bound3,
                  set_bound3,
                  ConditionBlock(
                    test_bound4,
                    set_bound4,
                    set_bound5
                  )
                )
              )
            )
          ),
          Return(sign*result)
        )
    infty_return = ConditionBlock(test_positive, Return(half_pi_cst), Return(-half_pi_cst))
    non_std_return = ConditionBlock(test_nan, Return(FP_QNaN(self.precision)), infty_return)
    scheme = ConditionBlock(test_NaN_or_inf, Statement(ClearException(), non_std_return), std_scheme)
    return scheme

  def numeric_emulate(self, input_value):
    return atan(input_value)

  standard_test_cases =[[sollya.parse(x)] for x in  ["0x1.107a78p+0", "0x1.9e75a6p+0" ]]



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Atan.get_default_args())
    args = arg_template.arg_extraction()
    ml_atan = ML_Atan(args)
    ml_atan.gen_implementation()
