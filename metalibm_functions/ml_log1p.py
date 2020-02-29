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
#
# created:            Mar   11th, 2016
# last-modified:      Mar    3rd, 2019
#
# description: meta-implementation of log(1+x)
#
###############################################################################

import sollya

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, log1p,
    guessdegree
)

from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.ml_table import ML_NewTable, generic_mantissa_msb_index_fct
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.special_values import (
        FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.debug_utils import debug_multi


# static constant for numerical value 2
S2 = sollya.SollyaObject(2)

class ML_Log1p(ML_FunctionBasis):
    function_name = "ml_log1p"
    def __init__(self, args):
        ML_FunctionBasis.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Log1p,
                builtin from a default argument mapping overloaded with @p kw """
        default_args_log1p = {
                "output_file": "my_log1p.c",
                "function_name": "my_log1pf",
                "precision": ML_Binary32,
                "accuracy": ML_Faithful,
                "target": GenericProcessor.get_target_instance(),
                "passes": [("start:instantiate_abstract_prec"), ("start:instantiate_prec"), ("start:basic_legalization"), ("start:expand_multi_precision")],
        }
        default_args_log1p.update(kw)
        return DefaultArgTemplate(**default_args_log1p)

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("x", self.precision)
        sollya_precision = self.get_input_precision().sollya_object

        # local overloading of RaiseReturn operation
        def ExpRaiseReturn(*args, **kwords):
                kwords["arg_value"] = vx
                kwords["function_name"] = self.function_name
                return RaiseReturn(*args, **kwords)

        # 2-limb approximation of log(2)
        # hi part precision is reduced to provide exact operation
        # when multiplied by an exponent value
        log2_hi_value = round(log(2), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
        log2_lo_value = round(log(2) - log2_hi_value, self.precision.sollya_object, sollya.RN)

        log2_hi = Constant(log2_hi_value, precision=self.precision)
        log2_lo = Constant(log2_lo_value, precision=self.precision)


        int_precision = self.precision.get_integer_format()

        # retrieving processor inverse approximation table
        dummy_var = Variable("dummy", precision = self.precision)
        dummy_rcp_seed = ReciprocalSeed(dummy_var, precision = self.precision)
        inv_approx_table = self.processor.get_recursive_implementation(dummy_rcp_seed, language = None, table_getter = lambda self: self.approx_table_map)

        # table creation
        table_index_size = inv_approx_table.index_size
        log_table = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision)
        # storing accurate logarithm approximation of value returned
        # by the fast reciprocal operation
        for i in range(0, 2**table_index_size):
            inv_value = inv_approx_table[i]
            value_high = round(log(inv_value), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
            value_low = round(log(inv_value) - value_high, sollya_precision, sollya.RN)
            log_table[i][0] = value_high
            log_table[i][1] = value_low


        neg_input = Comparison(vx, -1, likely=False, precision=ML_Bool, specifier=Comparison.Less, debug=debug_multi, tag="neg_input")
        vx_nan_or_inf = Test(vx, specifier=Test.IsInfOrNaN, likely=False, precision=ML_Bool, debug=debug_multi, tag="nan_or_inf")
        vx_snan = Test(vx, specifier=Test.IsSignalingNaN, likely=False, debug=debug_multi, tag="snan")
        vx_inf    = Test(vx, specifier=Test.IsInfty, likely=False, debug=debug_multi, tag="inf")
        vx_subnormal = Test(vx, specifier=Test.IsSubnormal, likely=False, debug=debug_multi, tag="vx_subnormal")

        # for x = m.2^e, such that e >= 0
        #
        # log(1+x) = log(1 + m.2^e)
        #          = log(2^e . 2^-e + m.2^e)
        #          = log(2^e . (2^-e + m))
        #          = log(2^e) + log(2^-e + m)
        #          = e . log(2) + log (2^-e + m)
        #
        # t = (2^-e + m)
        # t = m_t . 2^e_t
        # r ~ 1 / m_t   => r.m_t ~ 1 ~ 0
        #
        # t' = t . 2^-e_t
        #    = 2^-e-e_t + m . 2^-e_t
        #
        # if e >= 0, then 2^-e <= 1, then 1 <= m + 2^-e <= 3
        # r = m_r . 2^e_r
        #
        # log(1+x) = e.log(2) + log(r . 2^e_t . 2^-e_t . (2^-e + m) / r)
        #          = e.log(2) + log(r . 2^(-e-e_t) + r.m.2^-e_t) + e_t . log(2)- log(r)
        #          = (e+e_t).log(2) + log(r . t') - log(r)
        #          = (e+e_t).log(2) + log(r . t') - log(r)
        #          = (e+e_t).log(2) + P_log1p(r . t' - 1) - log(r)
        #
        #

        # argument reduction
        m = MantissaExtraction(vx, tag="vx", precision=self.precision, debug=debug_multi)
        e = ExponentExtraction(vx, tag="e", precision=int_precision, debug=debug_multi)

        # 2^-e
        TwoMinusE = ExponentInsertion(-e, tag="Two_minus_e", precision=self.precision, debug=debug_multi)
        t = Addition(TwoMinusE, m, precision=self.precision, tag="t", debug=debug_multi)

        m_t = MantissaExtraction(t, tag="m_t", precision=self.precision, debug=debug_multi)
        e_t = ExponentExtraction(t, tag="e_t", precision=int_precision, debug=debug_multi)

        # 2^(-e-e_t)
        TwoMinusEEt = ExponentInsertion(-e-e_t, tag="Two_minus_e_et", precision=self.precision)
        TwoMinusEt = ExponentInsertion(-e_t, tag="Two_minus_et", precision=self.precision, debug=debug_multi)

        rcp_mt = ReciprocalSeed(m_t, tag="rcp_mt", precision=self.precision, debug=debug_multi)

        INDEX_SIZE = table_index_size
        table_index = generic_mantissa_msb_index_fct(INDEX_SIZE, m_t)
        table_index.set_attributes(tag="table_index", debug=debug_multi)

        log_inv_lo = TableLoad(log_table, table_index, 1, tag="log_inv_lo", debug=debug_multi) 
        log_inv_hi = TableLoad(log_table, table_index, 0, tag="log_inv_hi", debug=debug_multi)

        inv_err = S2**-6 # TODO: link to target DivisionSeed precision

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(-inv_err, inv_err)
        approx_fct = sollya.log1p(sollya.x) / (sollya.x)
        poly_degree = sup(guessdegree(approx_fct, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
        Log.report(Log.Debug, "poly_degree is {}", poly_degree)
        global_poly_object = Polynomial.build_from_approximation(approx_fct, poly_degree, [self.precision]*(poly_degree+1), approx_interval, sollya.absolute)
        poly_object = global_poly_object # .sub_poly(start_index=1)

        EXT_PRECISION_MAP = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble,
            ML_SingleSingle: ML_TripleSingle,
            ML_DoubleDouble: ML_TripleDouble
        }
        if not self.precision in EXT_PRECISION_MAP:
            Log.report(Log.Error, "no extended precision available for {}", self.precision)

        ext_precision = EXT_PRECISION_MAP[self.precision]

        # pre_rtp = r . 2^(-e-e_t) + m .2^-e_t
        pre_rtp = Addition(
            rcp_mt * TwoMinusEEt,
            Multiplication(
                rcp_mt,
                Multiplication(
                    m,
                    TwoMinusEt,
                    precision=self.precision,
                    tag="pre_mult",
                    debug=debug_multi,
                ),
                precision=ext_precision,
                tag="pre_mult2",
                debug=debug_multi,
            ),
            precision=ext_precision,
            tag="pre_rtp",
            debug=debug_multi
        )
        pre_red_vx = Addition(
            pre_rtp,
            -1,
            precision=ext_precision,
        )

        red_vx = Conversion(pre_red_vx, precision=self.precision, tag="red_vx", debug=debug_multi)

        Log.report(Log.Info, "generating polynomial evaluation scheme")
        poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, red_vx, unified_precision=self.precision)

        poly.set_attributes(tag="poly", debug=debug_multi)
        Log.report(Log.Debug, "{}", global_poly_object.get_sollya_object())

        fp_e = Conversion(e + e_t, precision=self.precision, tag="fp_e", debug=debug_multi)


        ext_poly = Multiplication(red_vx, poly, precision=ext_precision)

        pre_result = Addition(
            Addition(
                fp_e * log2_hi,
                fp_e * log2_lo,
                precision=ext_precision
            ),
            Addition(
                Addition(
                    -log_inv_hi,
                    -log_inv_lo,
                    precision=ext_precision
                ),
                ext_poly,
                precision=ext_precision
            ),
            precision=ext_precision
        )

        result = Conversion(pre_result, precision=self.precision, tag="result", debug=debug_multi)


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
                Return(result)
            )
        )
        scheme = pre_scheme
        return scheme

    def numeric_emulate(self, input_value):
        return log1p(input_value)

    standard_test_cases = [
        (1.0, None),
        (1.0, None),
        (1.0, None),
        (1.0, None),
    ]
    _ = [
        (4.0, None),
        (1.0, None),
        (0.5, None),
        (1.5, None),
        (1024.0, None),
        (sollya.parse("0x1.13b2c6p-2"), None),
        (sollya.parse("0x1.2cb10ap-5"), None),
        (0.0, None),
        (sollya.parse("0x1.ce4492p-21"), None),
    ]



if __name__ == "__main__":
        # auto-test
        arg_template = ML_NewArgTemplate(default_arg=ML_Log1p.get_default_args())
        args = arg_template.arg_extraction()

        ml_log1p = ML_Log1p(args)
        ml_log1p.gen_implementation()
