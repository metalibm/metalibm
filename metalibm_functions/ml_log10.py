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
        S2, Interval, ceil, floor, round, inf, sup, log, exp, log10,
        guessdegree, RN, x
)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract, is_gappa_installed
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  
from metalibm_core.utility.debug_utils import *

class ML_Log10(ML_Function("log10")):
  def __init__(self, args):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log10,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log10 = {
        "output_file": "my_log10f.c",
        "function_name": "my_log10f",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log10.update(kw)
    return DefaultArgTemplate(**default_args_log10)

  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    #mpfr_x = emulate_implementation.add_input_variable("x", ML_Mpfr_t)
    #mpfr_rnd = emulate_implementation.add_input_variable("rnd", ML_Int32)
    emulate_func_name = "mpfr_log10"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    #emulate_func_op.declare_prototype = emulate_func
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self):
    #func_implementation = CodeFunction(self.function_name, output_format = self.precision)
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    sollya_precision = self.get_input_precision().get_sollya_object()

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

    log2_hi_value = round(log10(2), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), RN)
    log2_lo_value = round(log10(2) - log2_hi_value, self.precision.sollya_object, RN)

    log2_hi = Constant(log2_hi_value, precision = self.precision)
    log2_lo = Constant(log2_lo_value, precision = self.precision)

    vx_exp  = ExponentExtraction(vx, tag = "vx_exp", debug = debugd)

    int_precision = self.precision.get_integer_format()

    # retrieving processor inverse approximation table
    dummy_var = Variable("dummy", precision = self.precision)
    dummy_div_seed = DivisionSeed(dummy_var, precision = self.precision)
    inv_approx_table = self.processor.get_recursive_implementation(dummy_div_seed, language = None, table_getter = lambda self: self.approx_table_map)

    # table creation
    table_index_size = 7
    table_index_range = range(1, 2**table_index_size)
    log_table = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision)
    log_table[0][0] = 0.0
    log_table[0][1] = 0.0
    for i in table_index_range:
        #inv_value = (1.0 + (self.processor.inv_approx_table[i] / S2**9) + S2**-52) * S2**-1
        #inv_value = (1.0 + (inv_approx_table[i][0] / S2**9) ) * S2**-1
        inv_value = inv_approx_table[i][0]
        value_high = round(log10(inv_value), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
        value_low = round(log10(inv_value) - value_high, sollya_precision, sollya.RN)
        log_table[i][0] = value_high
        log_table[i][1] = value_low

    # determining log_table range
    high_index_function = lambda table, i: table[i][0]
    low_index_function  = lambda table, i: table[i][1]
    table_high_interval = log_table.get_subset_interval(high_index_function, table_index_range)
    table_low_interval  = log_table.get_subset_interval(low_index_function,  table_index_range)

    def compute_log(_vx, exp_corr_factor = None):
        _vx_mant = MantissaExtraction(_vx, tag = "_vx_mant", debug = debug_lftolx)
        _vx_exp  = ExponentExtraction(_vx, tag = "_vx_exp", debug = debugd)

        table_index = BitLogicAnd(BitLogicRightShift(TypeCast(_vx_mant, precision = int_precision, debug = debuglx), self.precision.get_field_size() - 7, debug = debuglx), 0x7f, tag = "table_index", debug = debuglld) 

        # argument reduction
        # TODO: detect if single operand inverse seed is supported by the targeted architecture
        pre_arg_red_index = TypeCast(BitLogicAnd(TypeCast(DivisionSeed(_vx_mant, precision = self.precision, tag = "seed", debug = debug_lftolx, silent = True), precision = ML_UInt64), Constant(-2, precision = ML_UInt64), precision = ML_UInt64), precision = self.precision, tag = "pre_arg_red_index", debug = debug_lftolx)
        arg_red_index = Select(Equal(table_index, 0), 1.0, pre_arg_red_index, tag = "arg_red_index", debug = debug_lftolx)
        #if not processor.is_supported_operation(arg_red_index):
        #    if self.precision != ML_Binary32:
        #        arg_red_index = DivisionSeed(Conversion(_vx_mant, precision = ML_Binary32), precision = ML_Binary32,  
        _red_vx        = arg_red_index * _vx_mant - 1.0
        inv_err = S2**-7
        red_interval = Interval(1 - inv_err, 1 + inv_err)
        _red_vx.set_attributes(tag = "_red_vx", debug = debug_lftolx, interval = red_interval)

        # return in case of standard (non-special) input
        _log_inv_lo = TableLoad(log_table, table_index, 1, tag = "log_inv_lo", debug = debug_lftolx) 
        _log_inv_hi = TableLoad(log_table, table_index, 0, tag = "log_inv_hi", debug = debug_lftolx)

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(-inv_err, inv_err)
        poly_degree = sup(guessdegree(log10(1+sollya.x)/sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        global_poly_object = Polynomial.build_from_approximation(log10(1+x)/x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, sollya.absolute)
        poly_object = global_poly_object#.sub_poly(start_index = 1)

        Log.report(Log.Info, "generating polynomial evaluation scheme")
        _poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)
        _poly.set_attributes(tag = "poly", debug = debug_lftolx)
        Log.report(Log.Info, global_poly_object.get_sollya_object())

        corr_exp = Conversion(_vx_exp if exp_corr_factor == None else _vx_exp + exp_corr_factor, precision = self.precision)
        split_red_vx = Split(_red_vx, precision = ML_DoubleDouble, tag = "split_red_vx", debug = debug_ddtolx) 
        red_vx_hi = split_red_vx.hi
        red_vx_lo = split_red_vx.lo

        # result = _red_vx * poly - log_inv_hi - log_inv_lo + _vx_exp * log2_hi + _vx_exp * log2_lo
        pre_result = -_log_inv_hi + ((_red_vx * _poly + (corr_exp * log2_lo - _log_inv_lo)))
        pre_result.set_attributes(tag = "pre_result", debug = debug_lftolx)
        exact_log2_hi_exp = corr_exp * log2_hi
        exact_log2_hi_exp.set_attributes(tag = "exact_log2_hi_hex", debug = debug_lftolx)
        cancel_part = (corr_exp * log2_hi - _log_inv_hi)
        cancel_part.set_attributes(tag = "cancel_part", debug = debug_lftolx)
        sub_part = red_vx_hi + cancel_part
        sub_part.set_attributes(tag = "sub_part", debug = debug_lftolx)
        #result_one_low_part = (red_vx_hi * _poly + (red_vx_lo + (red_vx_lo * _poly + (corr_exp * log2_lo - _log_inv_lo))))
        result_one_low_part = ((red_vx_lo + (red_vx_lo * _poly + (corr_exp * log2_lo - _log_inv_lo))))
        result_one_low_part.set_attributes(tag = "result_one_low_part", debug = debug_lftolx)
        _result_one = ((sub_part) + red_vx_hi * _poly) + result_one_low_part 
        _result = exact_log2_hi_exp + pre_result
        return _result, _poly, _log_inv_lo, _log_inv_hi, _red_vx, _result_one, corr_exp 

    result, poly, log_inv_lo, log_inv_hi, red_vx, new_result_one, corr_exp = compute_log(vx)
    result.set_attributes(tag = "result", debug = debug_lftolx)
    new_result_one.set_attributes(tag = "new_result_one", debug = debug_lftolx)

    # building eval error map
    eval_error_map = {
      red_vx: Variable("red_vx", precision = self.precision, interval = red_vx.get_interval()),
      log_inv_hi: Variable("log_inv_hi", precision = self.precision, interval = table_high_interval),
      log_inv_lo: Variable("log_inv_lo", precision = self.precision, interval = table_low_interval),
      corr_exp: Variable("corr_exp_g", precision = self.precision, interval = self.precision.get_exponent_interval()), 
    }
    # computing gappa error
    if is_gappa_installed():
      poly_eval_error = self.get_eval_error(result, eval_error_map)
      Log.report(Log.Info, "poly_eval_error: ", poly_eval_error)


    neg_input = Comparison(vx, 0, likely = False, specifier = Comparison.Less, debug = debugd, tag = "neg_input")
    vx_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debugd, tag = "nan_or_inf")
    vx_snan = Test(vx, specifier = Test.IsSignalingNaN, likely = False, debug = debugd, tag = "snan")
    vx_inf  = Test(vx, specifier = Test.IsInfty, likely = False, debug = debugd, tag = "inf")
    vx_subnormal = Test(vx, specifier = Test.IsSubnormal, likely = False, debug = debugd, tag = "vx_subnormal")
    vx_zero = Test(vx, specifier = Test.IsZero, likely = False, debug = debugd, tag = "vx_zero")

    exp_mone = Equal(vx_exp, -1, tag = "exp_minus_one", debug = debugd, likely = False)
    vx_one = Equal(vx, 1.0, tag = "vx_one", likely = False, debug = debugd)

    # exp=-1 case
    Log.report(Log.Info, "managing exp=-1 case")
    #red_vx_2 = arg_red_index * vx_mant * 0.5
    #approx_interval2 = Interval(0.5 - inv_err, 0.5 + inv_err)
    #poly_degree2 = sup(guessdegree(log(x), approx_interval2, S2**-(self.precision.get_field_size()+1))) + 1
    #poly_object2 = Polynomial.build_from_approximation(log(sollya.x), poly_degree, [self.precision]*(poly_degree+1), approx_interval2, sollya.absolute)
    #print "poly_object2: ", poly_object2.get_sollya_object()
    #poly2 = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object2, red_vx_2, unified_precision = self.precision)
    #poly2.set_attributes(tag = "poly2", debug = debug_lftolx)
    #result2 = (poly2 - log_inv_hi - log_inv_lo)

    log_subtract = -log_inv_hi - log2_hi
    log_subtract.set_attributes(tag = "log_subtract", debug = debug_lftolx)
    result2 = (log_subtract) + ((poly * red_vx) - (log_inv_lo + log2_lo))
    result2.set_attributes(tag = "result2", debug = debug_lftolx)

    m100 = -100
    S2100 = Constant(S2**100, precision = self.precision)
    result_subnormal, _, _, _, _, _, _ = compute_log(vx * S2100, exp_corr_factor = m100)

    Log.report(Log.Info, "managing close to 1.0 cases")
    one_err = S2**-7
    approx_interval_one = Interval(-one_err, one_err)
    red_vx_one = vx - 1.0
    poly_degree_one = sup(guessdegree(log10(1+sollya.x)/sollya.x, approx_interval_one, S2**-(self.precision.get_field_size()+1))) + 1
    poly_object_one = Polynomial.build_from_approximation(log10(1+sollya.x)/sollya.x, poly_degree_one, [self.precision]*(poly_degree_one+1), approx_interval_one, sollya.absolute).sub_poly(start_index = 1)
    poly_one = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object_one, red_vx_one, unified_precision = self.precision)
    poly_one.set_attributes(tag = "poly_one", debug = debug_lftolx)
    result_one = red_vx_one + red_vx_one * poly_one
    cond_one = (vx < (1+one_err)) & (vx > (1 - one_err))
    cond_one.set_attributes(tag = "cond_one", debug = debugd, likely = False)


    # main scheme
    Log.report(Log.Info, "MDL scheme")
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

  def numeric_emulate(self, input_value):
    return log10(input_value)


if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_Log10.get_default_args())
  args = arg_template.arg_extraction()


  ml_log10 = ML_Log10(args)
  ml_log10.gen_implementation()
