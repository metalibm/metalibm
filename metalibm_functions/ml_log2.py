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
    Interval, ceil, floor, round, inf, sup, log, exp, log2,
    guessdegree, x, RN, absolute
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import * 

class ML_Log2(ML_Function("ml_log2")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log2,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log2 = {
        "output_file": "my_log2f.c",
        "function_name": "my_log2f",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log2.update(kw)
    return DefaultArgTemplate(**default_args_log2)


  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode

        Deprecated: the new test bench uses numeric_emulate method
    """
    emulate_func_name = "mpfr_log2"
    emulate_func_op = FunctionOperator(
        emulate_func_name, arg_map = {
            0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)
        }, require_header = ["mpfr.h"]
    )
    emulate_func   = FunctionObject(
        emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op
    )
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))
    return mpfr_call


  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.get_input_precision()) 

    sollya_precision = self.get_input_precision().get_sollya_object()

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    # testing special value inputs
    test_nan_or_inf = Test(
        vx, specifier=Test.IsInfOrNaN, likely=False,
        debug=True, tag="nan_or_inf"
    )
    test_nan = Test(vx, specifier=Test.IsNaN, debug=True, tag="is_nan_test")
    test_positive = Comparison(
        vx, 0,
        specifier=Comparison.GreaterOrEqual,
        debug=True, tag="inf_sign"
    )
    test_signaling_nan = Test(
        vx, specifier=Test.IsSignalingNaN,
        debug=True, tag="is_signaling_nan")
    # if input is a signaling NaN, raise an invalid exception and returns
    # a quiet NaN
    return_snan = Statement(
        ExpRaiseReturn(ML_FPE_Invalid, return_value=FP_QNaN(self.precision)))

    vx_exp  = ExponentExtraction(vx, tag = "vx_exp", debug = debugd)

    int_precision = self.precision.get_integer_format()

    # log2(vx)
    # r = vx_mant
    # e = vx_exp
    # vx reduced to r in [1, 2[
    # log2(vx) = log2(r * 2^e)
    #          = log2(r) + e
    #
    ## log2(r) is approximated by
    #  log2(r) = log2(inv_seed(r) * r / inv_seed(r)
    #          = log2(inv_seed(r) * r) - log2(inv_seed(r))
    # inv_seed(r) in ]1/2, 1] => log2(inv_seed(r)) in ]-1, 0]
    #
    # inv_seed(r) * r ~ 1
    # we can easily tabulate -log2(inv_seed(r))
    #

    # retrieving processor inverse approximation table
    dummy_var = Variable("dummy", precision=self.precision)
    dummy_div_seed = ReciprocalSeed(dummy_var, precision=self.precision)
    inv_approx_table = self.processor.get_recursive_implementation(
        dummy_div_seed, language=None,
        table_getter=lambda self: self.approx_table_map)
    # table creation
    table_index_size = 7
    log_table = ML_NewTable(
        dimensions=[2**table_index_size, 2],
        storage_precision=self.precision, tag=self.uniquify_name("inv_table"))
    # value for index 0 is set to 0.0
    log_table[0][0] = 0.0
    log_table[0][1] = 0.0
    for i in range(1, 2**table_index_size):
        #inv_value = (1.0 + (self.processor.inv_approx_table[i] / S2**9) + S2**-52) * S2**-1
        #inv_value = (1.0 + (inv_approx_table[i][0] / S2**9) ) * S2**-1
        #print inv_approx_table[i][0], inv_value
        inv_value = inv_approx_table[i]
        value_high_bitsize = self.precision.get_field_size() - (self.precision.get_exponent_size() + 1)
        value_high = round(log2(inv_value), value_high_bitsize, sollya.RN)
        value_low = round(log2(inv_value) - value_high, sollya_precision, sollya.RN)
        log_table[i][0] = value_high
        log_table[i][1] = value_low

    def compute_log(_vx, exp_corr_factor = None):
        _vx_mant = MantissaExtraction(
            _vx, tag="_vx_mant", precision=self.precision, debug=debug_lftolx)
        _vx_exp  = ExponentExtraction(_vx, tag="_vx_exp", debug=debugd)

        # The main table is indexed by the 7 most significant bits
        # of the mantissa
        table_index = inv_approx_table.index_function(_vx_mant)
        table_index.set_attributes(tag="table_index", debug=debuglld)

        # argument reduction
        # Using AND -2 to exclude LSB set to 1 for Newton-Raphson convergence
        # TODO: detect if single operand inverse seed is supported by the targeted architecture
        pre_arg_red_index = TypeCast(
            BitLogicAnd(
                TypeCast(
                    ReciprocalSeed(
                        _vx_mant, precision=self.precision, tag="seed",
                        debug=debug_lftolx, silent=True
                    ), precision=ML_UInt64
                ),
                Constant(-2, precision=ML_UInt64), precision=ML_UInt64
            ),
            precision=self.precision, tag="pre_arg_red_index",
            debug=debug_lftolx
        )
        arg_red_index = Select(
            Equal(table_index, 0), 1.0, pre_arg_red_index,
            tag="arg_red_index", debug=debug_lftolx
        )
        _red_vx        = FMA(arg_red_index, _vx_mant, -1.0)
        _red_vx.set_attributes(tag="_red_vx", debug=debug_lftolx)
        inv_err = S2**-inv_approx_table.index_size
        red_interval = Interval(1 - inv_err, 1 + inv_err)

        # return in case of standard (non-special) input
        _log_inv_lo = TableLoad(log_table, table_index, 1, tag="log_inv_lo", debug=debug_lftolx) 
        _log_inv_hi = TableLoad(log_table, table_index, 0, tag="log_inv_hi", debug=debug_lftolx)

        Log.report(Log.Verbose, "building mathematical polynomial")
        approx_interval = Interval(-inv_err, inv_err)
        poly_degree = sup(guessdegree(log2(1+sollya.x)/sollya.x, approx_interval, S2**-(self.precision.get_field_size() * 1.1))) + 1
        sollya.settings.display = sollya.hexadecimal
        global_poly_object, approx_error = Polynomial.build_from_approximation_with_error(
            log2(1+sollya.x)/sollya.x, poly_degree,
            [self.precision]*(poly_degree+1),
            approx_interval,
            sollya.absolute,
            error_function=lambda p, f, ai, mod, t: sollya.dirtyinfnorm(p - f, ai)
        )
        Log.report(Log.Info, "poly_degree={}, approx_error={}".format(poly_degree, approx_error))
        poly_object = global_poly_object.sub_poly(start_index=1,offset=1)
        #poly_object = global_poly_object.sub_poly(start_index=0,offset=0)

        Attributes.set_default_silent(True)
        Attributes.set_default_rounding_mode(ML_RoundToNearest)

        Log.report(Log.Verbose, "generating polynomial evaluation scheme")
        pre_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, _red_vx, unified_precision = self.precision)
        _poly = FMA(pre_poly, _red_vx, global_poly_object.get_cst_coeff(0, self.precision))
        _poly.set_attributes(tag = "poly", debug = debug_lftolx)
        Log.report(Log.Verbose, "sollya global_poly_object: {}".format(
            global_poly_object.get_sollya_object()
        ))
        Log.report(Log.Verbose, "sollya poly_object: {}".format(
             poly_object.get_sollya_object()
        ))

        corr_exp = _vx_exp if exp_corr_factor == None else _vx_exp + exp_corr_factor

        Attributes.unset_default_rounding_mode()
        Attributes.unset_default_silent()

        pre_result = -_log_inv_hi + (_red_vx * _poly + (- _log_inv_lo))
        pre_result.set_attributes(tag = "pre_result", debug = debug_lftolx)
        exact_log2_hi_exp = Conversion(corr_exp, precision = self.precision)
        exact_log2_hi_exp.set_attributes(tag = "exact_log2_hi_hex", debug = debug_lftolx)
        _result = exact_log2_hi_exp + pre_result
        return _result, _poly, _log_inv_lo, _log_inv_hi, _red_vx

    result, poly, log_inv_lo, log_inv_hi, red_vx = compute_log(vx)
    result.set_attributes(tag = "result", debug=debug_lftolx)

    # specific input value predicate
    neg_input = Comparison(vx, 0, likely=False, specifier=Comparison.Less,
                debug=debugd, tag="neg_input")
    vx_nan_or_inf=Test(vx, specifier=Test.IsInfOrNaN, likely=False,
                       debug=debugd, tag="nan_or_inf")
    vx_snan = Test(vx, specifier=Test.IsSignalingNaN, likely=False,
                   debug=debugd, tag="vx_snan")
    vx_inf  = Test(vx, specifier=Test.IsInfty, likely=False,
                   debug=debugd, tag="vx_inf")
    vx_subnormal = Test(vx, specifier=Test.IsSubnormal, likely=False,
                        debug=debugd, tag="vx_subnormal")
    vx_zero = Test(vx, specifier=Test.IsZero, likely=False,
                   debug=debugd, tag="vx_zero")

    exp_mone = Equal(vx_exp, -1, tag = "exp_minus_one", debug=debugd, likely=False)
    vx_one = Equal(vx, 1.0, tag = "vx_one", likely = False, debug = debugd)

    # Specific specific for the case exp == -1
    # log2(x) = log2(m) - 1
    #
    # as m in [1, 2[, log2(m) in [0, 1[
    # if r is close to 2, a catastrophic cancellation can occur
    #
    # r = seed(m)
    # log2(x) = log2(seed(m) * m / seed(m)) - 1
    #         = log2(seed(m) * m) - log2(seed(m)) - 1
    #
    # for m really close to 2 => seed(m) = 0.5
    #     => log2(x) = log2(0.5 * m)
    #                = 
    result_exp_m1 = (-log_inv_hi - 1.0) + FMA(poly, red_vx, -log_inv_lo)
    result_exp_m1.set_attributes(tag = "result_exp_m1", debug=debug_lftolx)

    m100 = -100
    S2100 = Constant(S2**100, precision = self.precision)
    result_subnormal, _, _, _, _ = compute_log(vx * S2100, exp_corr_factor = m100)
    result_subnormal.set_attributes(tag = "result_subnormal", debug = debug_lftolx)

    one_err = S2**-7
    approx_interval_one = Interval(-one_err, one_err)
    red_vx_one = vx - 1.0
    poly_degree_one = sup(guessdegree(log(1+x)/x, approx_interval_one, S2**-(self.precision.get_field_size()+1))) + 1
    poly_object_one = Polynomial.build_from_approximation(log(1+sollya.x)/sollya.x, poly_degree_one, [self.precision]*(poly_degree_one+1), approx_interval_one, absolute).sub_poly(start_index = 1)
    poly_one = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object_one, red_vx_one, unified_precision = self.precision)
    poly_one.set_attributes(tag = "poly_one", debug = debug_lftolx)
    result_one = red_vx_one + red_vx_one * poly_one
    cond_one = (vx < (1+one_err)) & (vx > (1 - one_err))
    cond_one.set_attributes(tag = "cond_one", debug = debugd, likely = False)


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
                    Statement(
                        ClearException(),
                        result_subnormal,
                        Return(result_subnormal)
                    )
                ),
                ConditionBlock(vx_one,
                    Statement(
                        ClearException(),
                        Return(FP_PlusZero(self.precision)),
                    ),
                    ConditionBlock(exp_mone,
                        Return(result_exp_m1),
                        Return(result)
                    )
                )
            )
        )
    )
    scheme = Statement(result, pre_scheme)
    return scheme


  standard_test_cases = [(sollya.parse("0x1.ffd6906acffc7p-1"),)]

  def numeric_emulate(self, input_value):
    """ Numeric emulation to generate expected value
        corresponding to input_value input """
    return log2(input_value)



if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_Log2.get_default_args())
  args = arg_template.arg_extraction()

  ml_log2          = ML_Log2(args)
  ml_log2.gen_implementation()

