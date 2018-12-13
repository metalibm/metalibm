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
    Interval, ceil, floor, round, inf, sup, log, exp, log10,
    guessdegree, RN, x
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.opt.ml_blocks import Mul211, Add222, Add212, Mul222

from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract, is_gappa_installed
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.debug_utils import *



class ML_Log10(ML_Function("log10")):
  def __init__(self, args):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)
    self.basis = args.basis


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log10,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log10 = {
        "output_file": "ml_log10f.c",
        "function_name": "ml_log10f",
        "basis": 10,
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log10.update(kw)
    return DefaultArgTemplate(**default_args_log10)


  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.get_input_precision())

    sollya_precision = self.get_input_precision().get_sollya_object()

    log_f = sollya.log(sollya.x) # /sollya.log(self.basis)

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

    log2_hi_value = round(log_f(2), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), RN)
    log2_lo_value = round(log_f(2) - log2_hi_value, self.precision.sollya_object, RN)

    log2_hi = Constant(log2_hi_value, precision = self.precision)
    log2_lo = Constant(log2_lo_value, precision = self.precision)

    vx_exp  = ExponentExtraction(vx, tag = "vx_exp", debug = debugd)

    int_precision = self.precision.get_integer_format()


    #---------------------
    # Approximation scheme
    #---------------------
    # log10(x) = log10(m.2^e) = log10(m.2^(e-t+t))
    #           = log10(m.2^-t) + (e+t) log10(2)
    #  t = (m > sqrt(2)) ? 1 : 0  is used to avoid catastrophic cancellation
    #  when e = -1 and m ~ 2
    #
    #
    # log10(m.2^-t) = log10(m.r/r.2^-t) = log10(m.r) + log10(2^-t/r)
    #               = log10(m.r) - log10(r.2^t)
    #     where r = rcp(m) an approximation of 1/m such that r.m ~ 1

    # retrieving processor inverse approximation table
    dummy_var = Variable("dummy", precision = self.precision)
    dummy_div_seed = ReciprocalSeed(dummy_var, precision = self.precision)
    inv_approx_table = self.processor.get_recursive_implementation(
        dummy_div_seed, language=None,
        table_getter= lambda self: self.approx_table_map)

    # table creation
    table_index_size = inv_approx_table.index_size
    table_index_range = range(1, 2**table_index_size)
    log_table = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision)
    log_table_tho = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision)
    log_table[0][0] = 0.0
    log_table[0][1] = 0.0
    log_table_tho[0][0] = 0.0
    log_table_tho[0][1] = 0.0
    hi_size = self.precision.get_field_size() - (self.precision.get_exponent_size() + 1)
    for i in table_index_range:
        #inv_value = (1.0 + (self.processor.inv_approx_table[i] / S2**9) + S2**-52) * S2**-1
        #inv_value = (1.0 + (inv_approx_table[i][0] / S2**9) ) * S2**-1
        inv_value = inv_approx_table[i]
        value_high = round(log_f(inv_value), hi_size, sollya.RN)
        value_low = round(log_f(inv_value) - value_high, sollya_precision, sollya.RN)
        log_table[i][0] = value_high
        log_table[i][1] = value_low

        inv_value_tho = S2 * inv_approx_table[i]
        value_high_tho = round(log_f(inv_value_tho), hi_size, sollya.RN)
        value_low_tho = round(log_f(inv_value_tho) - value_high_tho, sollya_precision, sollya.RN)
        log_table_tho[i][0] = value_high_tho
        log_table_tho[i][1] = value_low_tho

    # determining log_table range
    high_index_function = lambda table, i: table[i][0]
    low_index_function  = lambda table, i: table[i][1]
    table_high_interval = log_table.get_subset_interval(high_index_function, table_index_range)
    table_low_interval  = log_table.get_subset_interval(low_index_function,  table_index_range)

    def compute_log(_vx, exp_corr_factor = None):
        _vx_mant = MantissaExtraction(_vx, tag="_vx_mant", precision=self.precision, debug = debug_multi)
        _vx_exp  = ExponentExtraction(_vx, tag="_vx_exp", debug = debugd)

        table_index = inv_approx_table.index_function(_vx_mant)

        table_index.set_attributes(tag="table_index", debug=debug_multi)

        tho_cond = _vx_mant > Constant(sollya.sqrt(2), precision=self.precision)
        tho = Select(
            tho_cond,
            Constant(1.0, precision=self.precision),
            Constant(0.0, precision=self.precision),
            precision=self.precision,
            tag="tho",
            debug=debug_multi
        )

        rcp = ReciprocalSeed(_vx_mant, precision=self.precision, tag="rcp")
        r = Multiplication(
            rcp,
            _vx_mant,
            precision=self.precision,
            tag="r"
        )

        # argument reduction
        # TODO: detect if single operand inverse seed is supported by the targeted architecture
        pre_arg_red_index = TypeCast(
            BitLogicAnd(
                TypeCast(
                    ReciprocalSeed(
                        _vx_mant, precision = self.precision,
                        tag = "seed", debug = debug_multi, silent = True
                    ), precision = ML_UInt64
                ),
                Constant(-2, precision = ML_UInt64),
                precision = ML_UInt64
            ),
            precision=self.precision,
            tag="pre_arg_red_index", debug = debug_multi)

        arg_red_index = Select(Equal(table_index, 0), 1.0, pre_arg_red_index, tag = "arg_red_index", debug = debug_multi)
        _red_vx        = arg_red_index * _vx_mant - 1.0
        inv_err = S2**-6
        red_interval = Interval(1 - inv_err, 1 + inv_err)
        _red_vx.set_attributes(tag = "_red_vx", debug = debug_multi, interval = red_interval)

        # return in case of standard (non-special) input
        _log_inv_lo = Select(
            tho_cond,
            TableLoad(log_table_tho, table_index, 1),
            TableLoad(log_table, table_index, 1),
            tag = "log_inv_lo",
            debug=debug_multi
        )

        _log_inv_hi = Select(
            tho_cond,
            TableLoad(log_table_tho, table_index, 0),
            TableLoad(log_table, table_index, 0),
            tag="log_inv_hi",
            debug=debug_multi
        )

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(-inv_err, inv_err)
        poly_degree = sup(guessdegree(log(1+sollya.x)/sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        global_poly_object = Polynomial.build_from_approximation(log(1+x)/x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, sollya.absolute)
        poly_object = global_poly_object.sub_poly(start_index=1)

        Log.report(Log.Info, "generating polynomial evaluation scheme")
        _poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision=self.precision)
        _poly.set_attributes(tag = "poly", debug = debug_multi)
        Log.report(Log.Info, poly_object.get_sollya_object())

        corr_exp = Conversion(_vx_exp if exp_corr_factor == None else _vx_exp + exp_corr_factor, precision = self.precision) + tho
        corr_exp.set_attributes(tag="corr_exp", debug=debug_multi)

        # _poly approximates log10(1+r)/r
        # _poly * red_vx approximates log10(x)

        m0h, m0l = Mul211(
            _red_vx,
            _poly
        )
        m0h, m0l = Add212(
            _red_vx,
            m0h,
            m0l
        )
        m0h.set_attributes(tag="m0h", debug=debug_multi)
        m0l.set_attributes(tag="m0l")
        l0_h = corr_exp * log2_hi
        l0_l = corr_exp * log2_lo
        l0_h.set_attributes(tag="l0_h")
        l0_l.set_attributes(tag="l0_l")
        rh, rl = Add222(l0_h,l0_l, m0h, m0l)
        rh.set_attributes(tag="rh0", debug=debug_multi)
        rl.set_attributes(tag="rl0", debug=debug_multi)
        rh, rl = Add222(-_log_inv_hi, -_log_inv_lo, rh, rl)
        rh.set_attributes(tag="rh", debug=debug_multi)
        rl.set_attributes(tag="rl", debug=debug_multi)

        if sollya.log(self.basis) != 1.0:
            lbh = self.precision.round_sollya_object(1/sollya.log(self.basis))
            lbl = self.precision.round_sollya_object(1/sollya.log(self.basis) - lbh)
            rh, rl = Mul222(rh, rl, lbh, lbl)
            return rh
        else:
            return rh

    result = compute_log(vx)
    result.set_attributes(tag = "result", debug=debug_multi)

    if False:
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

    # exp=-1 case
    Log.report(Log.Info, "managing exp=-1 case")
    #red_vx_2 = arg_red_index * vx_mant * 0.5
    #approx_interval2 = Interval(0.5 - inv_err, 0.5 + inv_err)
    #poly_degree2 = sup(guessdegree(log(x), approx_interval2, S2**-(self.precision.get_field_size()+1))) + 1
    #poly_object2 = Polynomial.build_from_approximation(log(sollya.x), poly_degree, [self.precision]*(poly_degree+1), approx_interval2, sollya.absolute)
    #print "poly_object2: ", poly_object2.get_sollya_object()
    #poly2 = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object2, red_vx_2, unified_precision = self.precision)
    #poly2.set_attributes(tag = "poly2", debug = debug_multi)
    #result2 = (poly2 - log_inv_hi - log_inv_lo)

    m100 = -100
    S2100 = Constant(S2**100, precision = self.precision)
    result_subnormal = compute_log(vx * S2100, exp_corr_factor=m100)


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
                Return(result)
            )
        )
    )
    scheme = pre_scheme
    return scheme

  def numeric_emulate(self, input_value):
    return sollya.log(input_value)/sollya.log(self.basis)

  standard_test_cases = [(sollya.parse("0x1.42af3ap-1"), None)]


if __name__ == "__main__":
    # auto-test
    ARG_TEMPLATE = ML_NewArgTemplate(default_arg=ML_Log10.get_default_args())

    ARG_TEMPLATE.parser.add_argument(
      "--basis", dest="basis", action="store", default=10,
      type=sollya.parse,
      help="logarithm basis")

    args = ARG_TEMPLATE.arg_extraction()

    ml_log10 = ML_Log10(args)
    ml_log10.gen_implementation()
