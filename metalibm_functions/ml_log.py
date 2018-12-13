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

from sollya import Interval, ceil, floor, round, inf, sup, log, exp, log10, guessdegree
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

class ML_Log(ML_Function("ml_log")):
  def __init__(self, args):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log = {
        "output_file": "my_logf.c",
        "function_name": "my_log",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log.update(kw)
    return DefaultArgTemplate(**default_args_log)

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_log"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call



  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.precision) 

    sollya_precision = self.precision.sollya_object

    # constant computation
    invlog2 = round(1/log(2), sollya_precision, sollya.RN)
    invlog2_cst = Constant(invlog2, precision = self.precision)
    #v_log2_hi = round(log(2), 16, sollya.RN) 
    #v_log2_lo = round(log(2) - v_log2_hi, sollya_precision, sollya.RN)

    #log2_hi = Constant(v_log2_hi, precision = self.precision, tag = "log2_hi")
    #log2_lo = Constant(v_log2_lo, precision = self.precision, tag = "log2_lo")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)


    test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
    test_nan = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
    test_positive = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

    test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
    return_snan = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    v_log2_hi = round(log(2), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
    v_log2_lo = round(log(2) - v_log2_hi, self.precision.sollya_object, sollya.RN)
    log2_hi = Constant(v_log2_hi, precision = self.precision, tag = "log2_hi")
    log2_lo = Constant(v_log2_lo, precision = self.precision, tag = "log2_lo")

    vx_exp  = ExponentExtraction(vx, tag = "vx_exp", debug = debug_multi)

    int_precision = self.precision.get_integer_format()

    # table creation
    table_index_size = 7
    log_table = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision, tag = self.uniquify_name("inv_table"))
    log_table[0][0] = 0.0
    log_table[0][1] = 0.0


    # retrieving processor inverse approximation table
    dummy_var = Variable("dummy", precision = self.precision)
    dummy_div_seed = ReciprocalSeed(dummy_var, precision = self.precision)
    inv_approx_table = self.processor.get_recursive_implementation(dummy_div_seed, language = None, table_getter = lambda self: self.approx_table_map)

    integer_precision = {ML_Binary32: ML_UInt32, ML_Binary64: ML_UInt64}[self.precision]


    for i in range(1, 2**table_index_size):
        #inv_value = (1.0 + (self.processor.inv_approx_table[i] / S2**9) + S2**-52) * S2**-1
        inv_value = inv_approx_table[i] # (1.0 + (inv_approx_table[i][0] / S2**9) ) * S2**-1
        value_high = round(log(inv_value), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
        value_low = round(log(inv_value) - value_high, sollya_precision, sollya.RN)
        log_table[i][0] = value_high
        log_table[i][1] = value_low

    def compute_log(_vx, exp_corr_factor = None):
        _vx_mant = MantissaExtraction(_vx, tag = "_vx_mant", debug = debug_multi, precision = self.precision)
        _vx_exp  = ExponentExtraction(_vx, tag = "_vx_exp", debug = debug_multi)

        table_index = BitLogicAnd(
          BitLogicRightShift(
            TypeCast(
              _vx_mant, 
              precision = int_precision, 
              debug = debug_multi
            ), 
            self.precision.get_field_size() - 7, 
            debug = debug_multi
          ), 
          0x7f, 
          tag = "table_index", 
          debug = debug_multi
        ) 

        # argument reduction
        # TODO: detect if single operand inverse seed is supported by the targeted architecture
        pre_arg_red_index = TypeCast(
          BitLogicAnd(
            TypeCast(
              ReciprocalSeed(
                _vx_mant, 
                precision = self.precision, 
                tag = "seed", 
                debug = debug_multi, 
                silent = True
              ), 
              precision = integer_precision
            ), 
            Constant(
              -2, 
              precision = integer_precision
            ), 
            precision = integer_precision
          ), 
          precision = self.precision, 
          tag = "pre_arg_red_index", 
        debug = debug_multi
      )

        arg_red_index = Select(Equal(table_index, 0), 1.0, pre_arg_red_index)

        #_red_vx        = arg_red_index * _vx_mant - 1.0
        _red_vx = FusedMultiplyAdd(arg_red_index, _vx_mant, 1.0, specifier = FusedMultiplyAdd.Subtract)
        _red_vx.set_attributes(tag = "_red_vx", debug = debug_multi)

        inv_err = S2**-7
        red_interval = Interval(1 - inv_err, 1 + inv_err)

        # return in case of standard (non-special) input
        _log_inv_lo = TableLoad(log_table, table_index, 1, tag = "log_inv_lo", debug = debug_multi) 
        _log_inv_hi = TableLoad(log_table, table_index, 0, tag = "log_inv_hi", debug = debug_multi)

        Log.report(Log.Verbose, "building mathematical polynomial")
        approx_interval = Interval(-inv_err, inv_err)
        poly_degree = sup(guessdegree(log(1+sollya.x)/sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        global_poly_object = Polynomial.build_from_approximation(log(1+sollya.x)/sollya.x, poly_degree, [1] + [self.precision]*(poly_degree), approx_interval, sollya.absolute)
        poly_object = global_poly_object.sub_poly(start_index = 1)

        Log.report(Log.Verbose, "generating polynomial evaluation scheme")
        #_poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)
        _poly = PolynomialSchemeEvaluator.generate_estrin_scheme(poly_object, _red_vx, unified_precision = self.precision)

        _poly.set_attributes(tag = "poly", debug = debug_multi)

        corr_exp = Conversion(_vx_exp if exp_corr_factor == None else _vx_exp + exp_corr_factor, precision = self.precision)
        split_red_vx = Split(_red_vx, precision = ML_DoubleDouble, tag = "split_red_vx", debug = debug_multi) 
        red_vx_hi = split_red_vx.hi
        red_vx_lo = split_red_vx.lo

        # result = _red_vx * poly - log_inv_hi - log_inv_lo + _vx_exp * log2_hi + _vx_exp * log2_lo
        pre_result = -_log_inv_hi + (_red_vx + (_red_vx * _poly + (corr_exp * log2_lo - _log_inv_lo)))
        pre_result.set_attributes(tag = "pre_result", debug = debug_multi)
        exact_log2_hi_exp = corr_exp * log2_hi
        exact_log2_hi_exp.set_attributes(tag = "exact_log2_hi_exp", debug = debug_multi)
        cancel_part = (corr_exp * log2_hi - _log_inv_hi)
        cancel_part.set_attributes(tag = "cancel_part", debug = debug_multi)
        sub_part = red_vx_hi + cancel_part
        sub_part.set_attributes(tag = "sub_part", debug = debug_multi)
        #result_one_low_part = (red_vx_hi * _poly + (red_vx_lo + (red_vx_lo * _poly + (corr_exp * log2_lo - _log_inv_lo))))
        result_one_low_part = ((red_vx_lo + (red_vx_lo * _poly + (corr_exp * log2_lo - _log_inv_lo))))
        result_one_low_part.set_attributes(tag = "result_one_low_part", debug = debug_multi)
        _result_one = ((sub_part) + red_vx_hi * _poly) + result_one_low_part 
        return exact_log2_hi_exp + pre_result, _poly, _log_inv_lo, _log_inv_hi, _red_vx, _result_one 

    result, poly, log_inv_lo, log_inv_hi, red_vx, new_result_one = compute_log(vx)
    result.set_attributes(tag = "result", debug = debug_multi)
    new_result_one.set_attributes(tag = "new_result_one", debug = debug_multi)

    neg_input = Comparison(vx, 0, likely = False, specifier = Comparison.Less, debug = debug_multi, tag = "neg_input")
    vx_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debug_multi, tag = "nan_or_inf")
    vx_snan = Test(vx, specifier = Test.IsSignalingNaN, likely = False, debug = debug_multi, tag = "snan")
    vx_inf  = Test(vx, specifier = Test.IsInfty, likely = False, debug = debug_multi, tag = "inf")
    vx_subnormal = Test(vx, specifier = Test.IsSubnormal, likely = False, debug = debug_multi, tag = "vx_subnormal")
    vx_zero = Test(vx, specifier = Test.IsZero, likely = False, debug = debug_multi, tag = "vx_zero")

    exp_mone = Equal(vx_exp, -1, tag = "exp_minus_one", debug = debug_multi, likely = False)
    vx_one = Equal(vx, 1.0, tag = "vx_one", likely = False, debug = debug_multi)

    # exp=-1 case
    Log.report(Log.Verbose, "managing exp=-1 case")

    result2 = (-log_inv_hi - log2_hi) + ((red_vx + poly * red_vx) - log2_lo - log_inv_lo)
    result2.set_attributes(tag = "result2", debug = debug_multi)

    m100 = -100
    S2100 = Constant(S2**100, precision = self.precision)
    result_subnormal, _, _, _, _, _ = compute_log(vx * S2100, exp_corr_factor = m100)

    Log.report(Log.Verbose,  "managing close to 1.0 cases")
    one_err = S2**-7
    approx_interval_one = Interval(-one_err, one_err)
    red_vx_one = vx - 1.0
    poly_degree_one = sup(guessdegree(log(1+sollya.x)/sollya.x, approx_interval_one, S2**-(self.precision.get_field_size()+1))) + 1
    poly_object_one = Polynomial.build_from_approximation(log(1+sollya.x)/sollya.x, poly_degree_one, [self.precision]*(poly_degree_one+1), approx_interval_one, sollya.absolute).sub_poly(start_index = 1)
    poly_one = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object_one, red_vx_one, unified_precision = self.precision)
    poly_one.set_attributes(tag = "poly_one", debug = debug_multi)
    result_one = red_vx_one + red_vx_one * poly_one
    cond_one = (vx < (1+one_err)) & (vx > (1 - one_err))
    cond_one.set_attributes(tag = "cond_one", debug = debug_multi, likely = False)


    # main scheme
    pre_scheme = ConditionBlock(neg_input,
        Statement(
            ClearException(),
            Raise(ML_FPE_Invalid),
            Return(FP_QNaN(self.precision))
        ),
        ConditionBlock(vx_nan_or_inf,
            ConditionBlock(vx_inf,
                Statement(
                    ClearException(),
                    Return(FP_PlusInfty(self.precision)),
                ),
                Statement(
                    ClearException(),
                    ConditionBlock(vx_snan,
                        Raise(ML_FPE_Invalid)
                    ),
                    Return(FP_QNaN(self.precision))
                )
            ),
            ConditionBlock(vx_subnormal,
                ConditionBlock(vx_zero, 
                    Statement(
                        ClearException(),
                        Raise(ML_FPE_DivideByZero),
                        Return(FP_MinusInfty(self.precision)),
                    ),
                    Return(result_subnormal)
                ),
                ConditionBlock(vx_one,
                    Statement(
                        ClearException(),
                        Return(FP_PlusZero(self.precision)),
                    ),
                    ConditionBlock(exp_mone,
                        Return(result2),
                        Return(result)
                    )
                    #ConditionBlock(cond_one,
                        #Return(new_result_one),
                        #ConditionBlock(exp_mone,
                            #Return(result2),
                            #Return(result)
                        #)
                    #)
                )
            )
        )
    )
    scheme = pre_scheme

    return scheme

  standard_test_cases = [(sollya.parse("0x1.fe9a5p-1"),), (sollya.parse("0x1.fe9a5p-1"),)]

  def numeric_emulate(self, input_value):
    return log(input_value)

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_Log.get_default_args())
  args = arg_template.arg_extraction()

  ml_log          = ML_Log(args)

  ml_log.gen_implementation()
