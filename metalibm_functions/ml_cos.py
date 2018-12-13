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
###############################################################################
import sys
import sollya

from sollya import (
        Interval, ceil, floor, round, inf, sup, log, exp, cos, pi,
        guessdegree, dirtyinfnorm
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.special_values import FP_QNaN, FP_PlusInfty, FP_PlusZero
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed






class ML_Cosine(ML_FunctionBasis)):
  """ Implementation of cosinus function """
  function_name = "ml_cos"
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self,
        args
    )


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Cosine,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_cos = {
        "output_file": "my_cosf.c",
        "function_name": "my_cosf",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_cos.update(kw)
    return DefaultArgTemplate(**default_args_cos)


  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cos"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self): 
    # declaring CodeFunction and retrieving input variable
    vx = Abs(self.implementation.add_input_variable("x", self.precision), tag = "vx") 


    Log.report(Log.Info, "generating implementation scheme")
    if self.debug_flag: 
        Log.report(Log.Info, "debug has been enabled")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    debug_precision = {ML_Binary32: debug_ftox, ML_Binary64: debug_lftolx}[self.precision]


    test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
    test_nan        = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
    test_positive   = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

    test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
    return_snan        = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    # return in case of infinity input
    infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(self.precision)), Return(FP_PlusZero(self.precision))))
    # return in case of specific value input (NaN or inf)
    specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(self.precision))), infty_return)
    # return in case of standard (non-special) input

    sollya_precision = self.precision.get_sollya_object()
    hi_precision = self.precision.get_field_size() - 3


    

    # argument reduction
    frac_pi_index = 3
    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, sollya.RN)
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, sollya.RN)
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, sollya.RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = True)
    fk = Conversion(k, precision = self.precision, tag = "fk")

    inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
    inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

    red_vx_hi = (vx - inv_frac_pi_cst * fk)
    red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
    red_vx_lo_sub = inv_frac_pi_lo_cst * fk
    red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
    vx_d = Conversion(vx, precision = ML_Binary64, tag = "vx_d")
    pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d_hi = (vx_d - inv_frac_pi_cst * fk)
    pre_red_vx_d_hi.set_attributes(tag = "pre_red_vx_d_hi", precision = ML_Binary64, debug = debug_lftolx)
    pre_red_vx_d = pre_red_vx_d_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d.set_attributes(tag = "pre_red_vx_d", debug = debug_lftolx, precision = ML_Binary64)


    modk = Modulo(k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "switch_value", debug = True)

    sel_c = Equal(BitLogicAnd(modk, 2**(frac_pi_index-1)), 2**(frac_pi_index-1))
    red_vx = Select(sel_c, -pre_red_vx, pre_red_vx)
    red_vx.set_attributes(tag = "red_vx", debug = debug_precision, precision = self.precision)

    red_vx_d = Select(sel_c, -pre_red_vx_d, pre_red_vx_d)
    red_vx_d.set_attributes(tag = "red_vx_d", debug = debug_lftolx, precision = ML_Binary64)

    approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

    Log.report(Log.Info, "approx interval: %s\n" % approx_interval)

    error_goal_approx = S2**-self.precision.get_precision()


    Log.report(Log.Info, "building mathematical polynomial")
    poly_degree_vector = [None] * 2**(frac_pi_index+1)



    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

    index_relative = []

    poly_object_vector = [None] * 2**(frac_pi_index+1)
    for i in range(2**(frac_pi_index+1)):
      sub_func = cos(sollya.x+i*pi/S2**frac_pi_index)
      degree = int(sup(guessdegree(sub_func, approx_interval, error_goal_approx))) + 1

      degree_list = range(degree+1)
      a_interval = approx_interval
      if i == 0:
        # ad-hoc, TODO: to be cleaned
        degree = 6
        degree_list = range(0, degree+1, 2)
      elif i % 2**(frac_pi_index) == 2**(frac_pi_index-1):
        # for pi/2 and 3pi/2, an approx to  sin=cos(pi/2+x) 
        # must be generated
        degree_list = range(1, degree+1, 2)

      if i == 3 or i == 5 or i == 7 or i == 9: 
        precision_list =  [sollya.binary64] + [sollya.binary32] *(degree)
      else:
        precision_list = [sollya.binary32] * (degree+1)

      poly_degree_vector[i] = degree 

      constraint = sollya.absolute
      delta = (2**(frac_pi_index - 3))
      centered_i = (i % 2**(frac_pi_index)) - 2**(frac_pi_index-1)
      if centered_i < delta and centered_i > -delta and centered_i != 0:
        constraint = sollya.relative
        index_relative.append(i)
      Log.report(Log.Info, "generating approximation for %d/%d" % (i, 2**(frac_pi_index+1)))
      poly_object_vector[i], _ = Polynomial.build_from_approximation_with_error(sub_func, degree_list, precision_list, a_interval, constraint, error_function = error_function) 


    # unified power map for red_sx^n
    upm = {}
    rel_error_list = []

    poly_scheme_vector = [None] * (2**(frac_pi_index+1))

    for i in range(2**(frac_pi_index+1)):
      poly_object = poly_object_vector[i]
      poly_precision = self.precision
      if i == 3 or i == 5 or i == 7 or i == 9: 
          poly_precision = ML_Binary64
          c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = ML_Binary64)
          c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
          poly_hi = (c0 + c1 * red_vx)
          poly_hi.set_precision(ML_Binary64)
          red_vx_d_2 = red_vx_d * red_vx_d
          poly_scheme = poly_hi + red_vx_d_2 * polynomial_scheme_builder(poly_object.sub_poly(start_index = 2, offset = 2), red_vx, unified_precision = self.precision, power_map_ = upm)
          poly_scheme.set_attributes(unbreakable = True)
      elif i == 4:
          c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = ML_Binary64)
          poly_scheme = c1 * red_vx_d + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)
          poly_scheme.set_precision(ML_Binary64)
      else:
          poly_scheme = polynomial_scheme_builder(poly_object, red_vx, unified_precision = poly_precision, power_map_ = upm)
      #if i == 3:
      #  c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = self.precision)
      #  c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
      #  poly_scheme = (c0 + c1 * red_vx) + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)

      poly_scheme.set_attributes(tag = "poly_cos%dpi%d" % (i, 2**(frac_pi_index)), debug = debug_precision)
      poly_scheme_vector[i] = poly_scheme



      #try:
      if is_gappa_installed() and i == 3:
          opt_scheme = self.opt_engine.optimization_process(poly_scheme, self.precision, copy = True, fuse_fma = self.fuse_fma)

          tag_map = {}
          self.opt_engine.register_nodes_by_tag(opt_scheme, tag_map)

          gappa_vx = Variable("red_vx", precision = self.precision, interval = approx_interval)

          cg_eval_error_copy_map = {
              tag_map["red_vx"]:    gappa_vx, 
              tag_map["red_vx_d"]:  gappa_vx,
          }

          print "opt_scheme"
          print opt_scheme.get_str(depth = None, display_precision = True, memoization_map = {})

          eval_error = self.gappa_engine.get_eval_error_v2(self.opt_engine, opt_scheme, cg_eval_error_copy_map, gappa_filename = "red_arg_%d.g" % i)
          poly_range = cos(approx_interval+i*pi/S2**frac_pi_index)
          rel_error_list.append(eval_error / poly_range)


    #for rel_error in rel_error_list:
    #  print sup(abs(rel_error))

    #return 

    # case 17
    #poly17 = poly_object_vector[17]
    #c0 = Constant(coeff(poly17.get_sollya_object(), 0), precision = self.precision)
    #c1 = Constant(coeff(poly17.get_sollya_object(), 1), precision = self.precision)
    #poly_scheme_vector[17] = FusedMultiplyAdd(c1, red_vx, c0, specifier = FusedMultiplyAdd.Standard) + polynomial_scheme_builder(poly17.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)

    half = 2**frac_pi_index
    sub_half = 2**(frac_pi_index - 1)

    # determine if the reduced input is within the second and third quarter (not first nor fourth)
    # to negate the cosine output
    factor_cond = BitLogicAnd(BitLogicXor(BitLogicRightShift(modk, frac_pi_index), BitLogicRightShift(modk, frac_pi_index-1)), 1, tag = "factor_cond", debug = True)

    CM1 = Constant(-1, precision = self.precision)
    C1  = Constant(1, precision = self.precision)
    factor = Select(factor_cond, CM1, C1, tag = "factor", debug = debug_precision)
    factor2 = Select(Equal(modk, Constant(sub_half)), CM1, C1, tag = "factor2", debug = debug_precision) 


    switch_map = {}
    if 0:
      for i in range(2**(frac_pi_index+1)):
        switch_map[i] = Return(poly_scheme_vector[i])
    else:
      for i in range(2**(frac_pi_index-1)):
        switch_case = (i, half - i)
        #switch_map[i]      = Return(poly_scheme_vector[i])
        #switch_map[half-i] = Return(-poly_scheme_vector[i])
        if i!= 0:
          switch_case = switch_case + (half+i, 2*half-i)
          #switch_map[half+i] = Return(-poly_scheme_vector[i])
          #switch_map[2*half-i] = Return(poly_scheme_vector[i])
        if poly_scheme_vector[i].get_precision() != self.precision:
          poly_result = Conversion(poly_scheme_vector[i], precision = self.precision)
        else:
          poly_result = poly_scheme_vector[i]
        switch_map[switch_case] = Return(factor*poly_result)
      #switch_map[sub_half] = Return(-poly_scheme_vector[sub_half])
      #switch_map[half + sub_half] = Return(poly_scheme_vector[sub_half])
      switch_map[(sub_half, half + sub_half)] = Return(factor2 * poly_scheme_vector[sub_half])


    result = SwitchBlock(modk, switch_map)

    #######################################################################
    #                    LARGE ARGUMENT MANAGEMENT                        #
    #                 (lar: Large Argument Reduction)                     #
    #######################################################################

    # payne and hanek argument reduction for large arguments
    #red_func_name = "payne_hanek_cosfp32" # "payne_hanek_fp32_asm"
    red_func_name = "payne_hanek_fp32_asm"
    payne_hanek_func_op = FunctionOperator(red_func_name, arg_map = {0: FO_Arg(0)}, require_header = ["support_lib/ml_red_arg.h"]) 
    payne_hanek_func   = FunctionObject(red_func_name, [ML_Binary32], ML_Binary64, payne_hanek_func_op)
    payne_hanek_func_op.declare_prototype = payne_hanek_func
    #large_arg_red = FunctionCall(payne_hanek_func, vx)
    large_arg_red = payne_hanek_func(vx)
    red_bound     = S2**20
    
    cond = Abs(vx) >= red_bound
    cond.set_attributes(tag = "cond", likely = False)


    
    lar_neark = NearestInteger(large_arg_red, precision = ML_Int64)
    lar_modk = Modulo(lar_neark, Constant(16, precision = ML_Int64), tag = "lar_modk", debug = True) 
    # Modulo is supposed to be already performed (by payne_hanek_cosfp32)
    #lar_modk = NearestInteger(large_arg_red, precision = ML_Int64)
    pre_lar_red_vx = large_arg_red - Conversion(lar_neark, precision = ML_Binary64)
    pre_lar_red_vx.set_attributes(precision = ML_Binary64, debug = debug_lftolx, tag = "pre_lar_red_vx")
    lar_red_vx = Conversion(pre_lar_red_vx, precision = self.precision, debug = debug_precision, tag = "lar_red_vx")
    lar_red_vx_lo = Conversion(pre_lar_red_vx - Conversion(lar_red_vx, precision = ML_Binary64), precision = self.precision)
    lar_red_vx_lo.set_attributes(tag = "lar_red_vx_lo", precision = self.precision)

    lar_k = 3
    # large arg reduction Universal Power Map
    lar_upm = {}
    lar_switch_map = {}
    approx_interval = Interval(-0.5, 0.5)
    for i in range(2**(lar_k+1)):
      frac_pi = pi / S2**lar_k
      func = cos(frac_pi * i + frac_pi * sollya.x)
      
      degree = 6
      error_mode = sollya.absolute
      if i % 2**(lar_k) == 2**(lar_k-1):
        # close to sin(x) cases
        func = -sin(frac_pi * x) if i == 2**(lar_k-1) else sin(frac_pi * x)
        degree_list = range(0, degree+1, 2)
        precision_list = [sollya.binary32] * len(degree_list)
        poly_object, _ = Polynomial.build_from_approximation_with_error(func/x, degree_list, precision_list, approx_interval, error_mode)
        poly_object = poly_object.sub_poly(offset = -1)
      else:
        degree_list = range(degree+1)
        precision_list = [sollya.binary32] * len(degree_list)
        poly_object, _ = Polynomial.build_from_approximation_with_error(func, degree_list, precision_list, approx_interval, error_mode)

      if i == 3 or i == 5 or i == 7 or i == 9 or i == 11 or i == 13: 
          poly_precision = ML_Binary64
          c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = ML_Binary64)
          c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
          poly_hi = (c0 + c1 * lar_red_vx)
          poly_hi.set_precision(ML_Binary64)
          pre_poly_scheme = poly_hi + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
          pre_poly_scheme.set_attributes(precision = ML_Binary64)
          poly_scheme = Conversion(pre_poly_scheme, precision = self.precision)
      elif i == 4 or i == 12:
        c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
        c3 = Constant(coeff(poly_object.get_sollya_object(), 3), precision = self.precision)
        c5 = Constant(coeff(poly_object.get_sollya_object(), 5), precision = self.precision)
        poly_hi = polynomial_scheme_builder(poly_object.sub_poly(start_index = 3), lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
        poly_hi.set_attributes(tag = "poly_lar_%d_hi" % i, precision = ML_Binary64)
        poly_scheme = Conversion(FusedMultiplyAdd(c1, lar_red_vx, poly_hi, precision = ML_Binary64) + c1 * lar_red_vx_lo, precision = self.precision)
      else:
        poly_scheme = polynomial_scheme_builder(poly_object, lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm)
      # poly_scheme = polynomial_scheme_builder(poly_object, lar_red_vx, unified_precision = self.precision, power_map_ = lar_upm) 
      poly_scheme.set_attributes(tag = "lar_poly_%d" % i, debug = debug_precision)
      lar_switch_map[(i,)] = Return(poly_scheme)
    
    lar_result = SwitchBlock(lar_modk, lar_switch_map)


    # main scheme
    #Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    # scheme = Statement(ConditionBlock(cond, lar_result, result))

    Log.report(Log.Info, "Construction of the initial MDL scheme")
    scheme = Statement(pre_red_vx_d, red_vx_lo_sub, ConditionBlock(cond, lar_result, result))

    return scheme




if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_Cosine.get_default_args())
  # argument extraction 
  args = arg_template.arg_extraction()

  ml_cos = ML_Cosine(args) 

  ml_cos.gen_implementation()
