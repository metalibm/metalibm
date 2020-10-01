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
# last-modified:    Oct 5th, 2018
#
# Description:      Meta-implementation of floating-point division
###############################################################################
import sollya

from sollya import Interval, sup

from metalibm_core.core.ml_operations import (
    Variable,
    Abs, Equal,
    Min, Max, Comparison,
    ReciprocalSeed, Constant,
    SpecificOperation, Test,
    ConditionBlock, Statement, Return,
    ExponentInsertion, ExponentExtraction,
    EmptyOperand, Raise,
    LogicalOr, Select,
    MantissaExtraction, ExponentExtraction,
    Loop,
    ReferenceAssign, Dereference,
)
from metalibm_core.core.attributes import Attributes
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_SingleSingle, ML_DoubleDouble,
    ML_Int32,
    ML_RoundToNearest, ML_GlobalRoundMode,
    ML_FPE_Invalid, ML_FPE_DivideByZero, ML_FPE_Inexact, ML_FPE_Underflow,
    ML_Bool, ML_Exact,
)
from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty,
    FP_MinusZero, FP_PlusZero,
    is_nan,
)
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.core.meta_interval import MetaInterval, MetaIntervalList

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.code_object import  GappaCodeObject
from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_NewArgTemplate, DefaultArgTemplate
from metalibm_core.utility.debug_utils import debug_multi
from metalibm_core.utility.log_report import Log


S2 = sollya.SollyaObject(2)



class ML_RemQuo(ML_FunctionBasis):
    function_name = "ml_remquo"
    arity = 2

    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args=args)

    @staticmethod
    def get_default_args(**args):
        """ Generate a default argument structure set specifically for
            the Hyperbolic Cosine """
        default_div_args = {
            "precision": ML_Binary64,
            "accuracy": ML_CorrectlyRounded,
            "target": GenericProcessor.get_target_instance(),
            "output_file": "my_remquo.c",
            "function_name": "my_remquo",
            "input_intervals": [None, None],
            "auto_test_range": DefaultArgTemplate.auto_test_range * 2,
            "bench_test_range": DefaultArgTemplate.bench_test_range * 2,
            "language": C_Code,
            "passes": ["typing:basic_legalization", "beforecodegen:expand_multi_precision"],
        }
        default_div_args.update(args)
        return DefaultArgTemplate(**default_div_args)

    def generate_scheme(self):
        int_precision = self.precision.get_integer_format()
        # We wish to compute vx / vy
        vx = self.implementation.add_input_variable("x", self.precision, interval=self.input_intervals[0])
        vy = self.implementation.add_input_variable("y", self.precision, interval=self.input_intervals[1])
        #quo = self.implementation.add_input_variable("quo", ML_Pointer_Format(int_precision))




        i = Variable("i", precision=int_precision, var_type=Variable.Local)
        q = Variable("q", precision=int_precision, var_type=Variable.Local)

        CI = lambda v: Constant(v, precision=int_precision)
        CF = lambda v: Constant(v, precision=self.precision)

        vx_subnormal = Test(vx, specifier=Test.IsSubnormal, tag="vx_subnormal")
        vy_subnormal = Test(vy, specifier=Test.IsSubnormal, tag="vy_subnormal")

        DELTA_EXP = self.precision.get_mantissa_size()
        scale_factor = Constant(2.0**DELTA_EXP, precision=self.precision)
        inv_scale_factor = Constant(2.0**-DELTA_EXP, precision=self.precision)

        scaled_vx = Select(vx_subnormal, vx * scale_factor, vx, tag="scaled_vx")
        scaled_vy = Select(vy_subnormal, vy * scale_factor, vy, tag="scaled_vy")

        real_ex = ExponentExtraction(vx, tag="real_ex", precision=int_precision)
        real_ey = ExponentExtraction(vy, tag="real_ey", precision=int_precision)
        scaled_ex = ExponentExtraction(scaled_vx, tag="scaled_ex", precision=int_precision)
        scaled_ey = ExponentExtraction(scaled_vy, tag="scaled_ey", precision=int_precision)

        ex = scaled_ex - Select(vx_subnormal, Constant(DELTA_EXP, precision=int_precision), 0)
        ex.set_tag("ex")
        ey = scaled_ey - Select(vy_subnormal, Constant(DELTA_EXP, precision=int_precision), 0)
        ey.set_tag("ey")


        ey_half0 = (real_ey) / 2
        ey_half1 = (real_ey) - ey_half0

        scale_ey_half0 = ExponentInsertion(ey_half0, precision=self.precision, tag="scale_ey_half0")
        scale_ey_half1 = ExponentInsertion(ey_half1, precision=self.precision, tag="scale_ey_half1")

        mx = MantissaExtraction(Abs(scaled_vx), tag="mx")
        my = MantissaExtraction(Abs(scaled_vy), tag="my")

        # vx / vy = vx * 2^-ex * 2^(ex-ey) / (vy * 2^-ey)
        # vx % vy

        post_mx = Variable("post_mx", precision=self.precision, var_type=Variable.Local)


        loop = Statement(
            ex, ey, mx, my,
            ReferenceAssign(q, CI(0)),
            Loop(
                ReferenceAssign(i, CI(0)), i < (real_ex - real_ey),
                Statement(
                    ReferenceAssign(i, i+CI(1)),
                    ReferenceAssign(q, q << 1 + Select(mx >= my, CI(1), CI(0))),
                    ReferenceAssign(mx, CF(2) * (mx - Select(mx >= my, my, CF(0))))
                )
            ),
            # unscaling remainder
            ReferenceAssign(mx, ((mx * scale_ey_half0) * scale_ey_half1).modify_attributes(tag="scaled_rem")),
            ReferenceAssign(my, ((my * scale_ey_half0) * scale_ey_half1).modify_attributes(tag="scaled_rem_my")),
            Loop(
                Statement(), my > Abs(vy),
                Statement(
                    ReferenceAssign(q, q << 1 + Select(mx >= Abs(my), CI(1), CI(0))),
                    ReferenceAssign(mx, (mx - Select(mx >= Abs(my), Abs(my), CF(0)))),
                    ReferenceAssign(my, my * 0.5),
                )
            ),
            Loop(
                ReferenceAssign(i, CI(0)), mx >= Abs(vy),
                Statement(
                    ReferenceAssign(q, q + Select(mx >= Abs(vy), CI(1), CI(0))),
                    ReferenceAssign(mx, (mx - Select(mx >= Abs(vy), Abs(vy), CF(0))))
                ),
            )
        )
        scheme = Statement(
            # x or y is NaN, a NaN is returned
            ConditionBlock(
                LogicalOr(Test(vx, specifier=Test.IsNaN), Test(vy, specifier=Test.IsNaN)),
                Return(FP_QNaN(self.precision))
            ),
            #
            ConditionBlock(
                Test(vx, specifier=Test.IsInfty),
                Return(FP_QNaN(self.precision))
            ),
            ConditionBlock(
                Test(vy, specifier=Test.IsZero),
                Return(FP_QNaN(self.precision))
            ),
            ConditionBlock(
                Test(vx, specifier=Test.IsZero),
                Return(FP_PlusZero(self.precision))
            ),
            ConditionBlock(
                Abs(vx) < Abs(vy),
                Return(vx),
            ),
            ConditionBlock(
                Equal(vx, vy),
                Return(FP_PlusZero(self.precision)),
            ),
            ConditionBlock(
                Equal(vx, -vy),
                Return(FP_MinusZero(self.precision)),
            ),
            loop,
            # ReferenceAssign(Dereference(quo, precision=int_precision), Constant(0, precision=int_precision)),
            Return(mx),
        )

        return scheme

    def numeric_emulate(self, vx, vy):
        """ Numeric emulation of exponential """
        if is_nan(vx) or is_nan(vy) or vy == 0:
            return FP_QNaN(self.precision)
        return sollya.euclidian_mod(vx, vy)


    standard_test_cases = [
        #result is 0x0.0000000000505p-1022 vs expected0x0.0000000000a3cp-1022
        # (sollya.parse("0x1.9f9f4e9a29fcfp-421"), sollya.parse("0x0.0000000001b59p-1022"), sollya.parse("0x0.0000000000a3cp-1022")),
        (sollya.parse("0x1.9906165fb3e61p+62"), sollya.parse("0x0.0000000005e7dp-1022")),
        (sollya.parse("0x1.77f00143ba3f4p+943"), sollya.parse("0x0.000000000001p-1022")),
        (sollya.parse("0x0.000000000001p-1022"), sollya.parse("0x0.000000000001p-1022")),
        (sollya.parse("0x0.000000000348bp-1022"), sollya.parse("0x0.000000000001p-1022")),
        (sollya.parse("0x1.bcf3955c3b244p-130"), sollya.parse("0x1.77aef33890951p-1003")),
        (sollya.parse("0x1.8de59bd84c51ep-866"), sollya.parse("0x1.045aa9bf14fb1p-774")),
        (sollya.parse("0x1.9f9f4e9a29fcfp-421"), sollya.parse("0x0.0000000001b59p-1022")),
        (sollya.parse("0x1.2e1c59b43a459p+953"), sollya.parse("0x0.0000001cf5319p-1022")),
    ]



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_arg=ML_RemQuo.get_default_args()
    )

    ARGS = arg_template.arg_extraction()

    ml_remquo = ML_RemQuo(ARGS)
    ml_remquo.gen_implementation()
