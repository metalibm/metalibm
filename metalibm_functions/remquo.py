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
    Constant,
    Test,
    NotEqual,
    ConditionBlock, Statement, Return,
    ExponentInsertion, ExponentExtraction,
    LogicalOr, Select,
    LogicalAnd, LogicalNot,
    MantissaExtraction, ExponentExtraction,
    Loop,
    Modulo, TypeCast,
    ReferenceAssign, Dereference,
)
from metalibm_core.core.attributes import Attributes
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64, ML_SingleSingle, ML_DoubleDouble,
    ML_Int32,
    ML_UInt64, ML_Int64,
    ML_Bool, ML_Exact,
)
from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty,
    FP_MinusZero, FP_PlusZero,
    is_nan, is_zero, is_infty,
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
        # initializing class specific arguments (required by ML_FunctionBasis init)
        self.quotient_mode = args.quotient_mode
        self.quotient_size = args.quotient_size
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

    def get_output_precision(self):
        if self.quotient_mode:
            return self.precision.get_integer_format()
        else:
            return self.precision

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

        normalized_vx = Select(vx_subnormal, vx * scale_factor, vx, tag="scaled_vx")
        normalized_vy = Select(vy_subnormal, vy * scale_factor, vy, tag="scaled_vy")

        real_ex = ExponentExtraction(vx, tag="real_ex", precision=int_precision)
        real_ey = ExponentExtraction(vy, tag="real_ey", precision=int_precision)

        # if real_e<x/y> is +1023 then it may Overflow in -real_ex for ExponentInsertion
        # which only supports downto -1022 before falling into subnormal numbers (which are
        # not supported by ExponentInsertion)
        real_ex_h0 = real_ex / 2
        real_ex_h1 = real_ex - real_ex_h0

        real_ey_h0 = real_ey / 2
        real_ey_h1 = real_ey - real_ey_h0

        EI = lambda v: ExponentInsertion(v, precision=self.precision)

        mx = Abs((vx * EI(-real_ex_h0)) * EI(-real_ex_h1), tag="mx")
        my = Abs((vy * EI(-real_ey_h0)) * EI(-real_ey_h1), tag="pre_my")

        ey_half0 = (real_ey) / 2
        ey_half1 = (real_ey) - ey_half0

        scale_ey_half0 = ExponentInsertion(ey_half0, precision=self.precision, tag="scale_ey_half0")
        scale_ey_half1 = ExponentInsertion(ey_half1, precision=self.precision, tag="scale_ey_half1")

        # if only vy is subnormal we want to normalize it
        normal_cond = LogicalAnd(vy_subnormal, LogicalNot(vx_subnormal))
        my = Select(normal_cond, Abs(MantissaExtraction(vy * scale_factor)), my, tag="my")


        # vx / vy = vx * 2^-ex * 2^(ex-ey) / (vy * 2^-ey)
        # vx % vy

        post_mx = Variable("post_mx", precision=self.precision, var_type=Variable.Local)

        def LogicalXor(a, b):
            return LogicalOr(LogicalAnd(a, LogicalNot(b)), LogicalAnd(LogicalNot(a), b))

        rem_sign = Select(vx < 0, CF(-1), CF(1), precision=self.precision, tag="rem_sign")
        quo_sign = Select(LogicalXor(vx <0, vy < 0), CI(-1), CI(1), precision=int_precision, tag="quo_sign")


        loop = Statement(
            real_ex, real_ey, mx, my,
            ReferenceAssign(q, CI(0)),
            Loop(
                ReferenceAssign(i, CI(0)), i < (real_ex - real_ey),
                Statement(
                    ReferenceAssign(i, i+CI(1)),
                    ReferenceAssign(q, ((q << 1) + Select(mx >= my, CI(1), CI(0))).modify_attributes(tag="step1_q")),
                    ReferenceAssign(mx, (CF(2) * (mx - Select(mx >= my, my, CF(0)))).modify_attributes(tag="step1_mx"))
                )
            ),
            # unscaling remainder
            ReferenceAssign(mx, ((mx * scale_ey_half0) * scale_ey_half1).modify_attributes(tag="scaled_rem")),
            ReferenceAssign(my, ((my * scale_ey_half0) * scale_ey_half1).modify_attributes(tag="scaled_rem_my")),
            Loop(
                Statement(), LogicalAnd(my > Abs(vy), NotEqual(mx, 0)),
                Statement(
                    ReferenceAssign(q, ((q << 1) + Select(mx >= Abs(my), CI(1), CI(0))).modify_attributes(tag="step2_q")),
                    ReferenceAssign(mx, (mx - Select(mx >= Abs(my), Abs(my), CF(0))).modify_attributes(tag="step2_mx")),
                    ReferenceAssign(my, my * 0.5),
                )
            ),
            ReferenceAssign(q, q << 1),
            Loop(
                ReferenceAssign(i, CI(0)), mx > Abs(vy),
                Statement(
                    ReferenceAssign(q, (q + Select(mx > Abs(vy), CI(1), CI(0))).modify_attributes(tag="step3_q")),
                    ReferenceAssign(mx, (mx - Select(mx > Abs(vy), Abs(vy), CF(0))).modify_attributes(tag="step3_mx"))
                ),
            ),
            ReferenceAssign(q, q + Select(mx >= Abs(vy), CI(1), CI(0))),
            ReferenceAssign(mx, (mx - Select(mx >= Abs(vy), Abs(vy), CF(0)))),
            ConditionBlock(
                mx > Abs(vy * 0.5),
                Statement(
                    ReferenceAssign(q, q + CI(1)),
                    ReferenceAssign(mx, (mx - Abs(vy)))
                )
            ),
            ConditionBlock(
                # if the remainder is exactly half the dividend
                # we need to make sure the quotient is even
                LogicalAnd(
                    Equal(mx, Abs(vy * 0.5)),
                    Equal(Modulo(q, CI(2)), CI(1)),
                ),
                Statement(
                    ReferenceAssign(q, q + CI(1)),
                    ReferenceAssign(mx, (mx - Abs(vy)))
                )
            ),
            ReferenceAssign(mx, rem_sign * mx),
            ReferenceAssign(q, quo_sign * q),
            ReferenceAssign(q,
                Modulo(TypeCast(q, precision=ML_UInt64), Constant(2**self.quotient_size, precision=ML_UInt64), tag="mod_q")
            ),
        )
        mod_scheme = Statement(
            # x or y is NaN, a NaN is returned
            ConditionBlock(
                LogicalOr(Test(vx, specifier=Test.IsNaN), Test(vy, specifier=Test.IsNaN)),
                Return(FP_QNaN(self.precision))
            ),
            #
            ConditionBlock(
                Test(vy, specifier=Test.IsZero),
                Return(FP_QNaN(self.precision))
            ),
            ConditionBlock(
                Test(vx, specifier=Test.IsZero),
                Return(vx)
            ),
            ConditionBlock(
                Test(vx, specifier=Test.IsInfty),
                Return(FP_QNaN(self.precision))
            ),
            ConditionBlock(
                Test(vy, specifier=Test.IsInfty),
                Return(FP_QNaN(self.precision))
            ),
            ConditionBlock(
                Abs(vx) < Abs(vy * 0.5),
                Return(vx),
            ),
            ConditionBlock(
                Equal(vx, vy),
                # 0 with the same sign as x
                Return(vx - vx),
            ),
            ConditionBlock(
                Equal(vx, -vy),
                # 0 with the same sign as x
                Return(vx - vx),
            ),
            loop,
            # ReferenceAssign(Dereference(quo, precision=int_precision), Constant(0, precision=int_precision)),
            Return(mx),
        )

        # quotient invalid value
        QUO_INVALID_VALUE = 0

        quo_scheme = Statement(
            # x or y is NaN, a NaN is returned
            ConditionBlock(
                LogicalOr(Test(vx, specifier=Test.IsNaN), Test(vy, specifier=Test.IsNaN)),
                Return(QUO_INVALID_VALUE),
            ),
            #
            ConditionBlock(
                Test(vy, specifier=Test.IsZero),
                Return(QUO_INVALID_VALUE),
            ),
            ConditionBlock(
                Test(vx, specifier=Test.IsZero),
                Return(0),
            ),
            ConditionBlock(
                Test(vx, specifier=Test.IsInfty),
                Return(QUO_INVALID_VALUE),
            ),
            ConditionBlock(
                Test(vy, specifier=Test.IsInfty),
                Return(QUO_INVALID_VALUE),
            ),
            ConditionBlock(
                Abs(vx) < Abs(vy * 0.5),
                Return(0),
            ),
            ConditionBlock(
                Equal(vx, vy),
                Return(1),
            ),
            ConditionBlock(
                Equal(vx, -vy),
                Return(-1),
            ),
            loop,
            # ReferenceAssign(Dereference(quo, precision=int_precision), Constant(0, precision=int_precision)),
            Return(q),

        )

        if self.quotient_mode:
            return quo_scheme
        else:
            return mod_scheme

    def numeric_emulate(self, vx, vy):
        """ Numeric emulation of exponential """
        if self.quotient_mode:
            if is_nan(vx) or is_nan(vy) or is_zero(vy) or is_infty(vx) or is_infty(vy):
                # invalid value specified by OpenCL-C
                return 0
            if is_zero(vx):
                # valid value
                return 0
        else:
            if is_nan(vx) or is_nan(vy) or is_zero(vy):
                return FP_QNaN(self.precision)
            elif is_zero(vx):
                return vx
            elif is_infty(vx):
                return FP_QNaN(self.precision)
            elif is_infty(vy):
                return FP_QNaN(self.precision)
        # factorizing canonical cases (including correctionà
        # between qutoient_mode and remainder mode
        pre_mod = sollya.euclidian_mod(vx, vy)
        pre_quo = sollya.euclidian_div(vx, vy)
        if abs(pre_mod) > abs(vy / 2):
            pre_mod -= vy
            pre_quo += 1
        if self.quotient_mode:
            return sollya.euclidian_mod(pre_quo, 2**self.quotient_size)
        else:
            return pre_mod


    @property
    def standard_test_cases(self):
        return [
            (sollya.parse("0x1.fffffffffffffp+1023"), sollya.parse("-0x1.fffffffffffffp+1023")),
            (sollya.parse("0x0.a9f466178b1fcp-1022"), sollya.parse("0x0.b22f552dc829ap-1022")),
            (sollya.parse("0x1.4af8b07942537p-430"), sollya.parse("-0x0.f72be041645b7p-1022")),
            #result is 0x0.0000000000505p-1022 vs expected0x0.0000000000a3cp-1022
            #(sollya.parse("0x1.9f9f4e9a29fcfp-421"), sollya.parse("0x0.0000000001b59p-1022"), sollya.parse("0x0.0000000000a3cp-1022")),
            (sollya.parse("0x1.9906165fb3e61p+62"), sollya.parse("0x1.9906165fb3e61p+60")),
            (sollya.parse("0x1.9906165fb3e61p+62"), sollya.parse("0x0.0000000005e7dp-1022")),
            (sollya.parse("0x1.77f00143ba3f4p+943"), sollya.parse("0x0.000000000001p-1022")),
            (sollya.parse("0x0.000000000001p-1022"), sollya.parse("0x0.000000000001p-1022")),
            (sollya.parse("0x0.000000000348bp-1022"), sollya.parse("0x0.000000000001p-1022")),
            (sollya.parse("0x1.bcf3955c3b244p-130"), sollya.parse("0x1.77aef33890951p-1003")),
            (sollya.parse("0x1.8de59bd84c51ep-866"), sollya.parse("0x1.045aa9bf14fb1p-774")),
            (sollya.parse("0x1.9f9f4e9a29fcfp-421"), sollya.parse("0x0.0000000001b59p-1022")),
            (sollya.parse("0x1.2e1c59b43a459p+953"), sollya.parse("0x0.0000001cf5319p-1022")),
            (sollya.parse("-0x1.86c83abe0854ep+268"), FP_MinusInfty(self.precision)),
            # bad sign of remainder
            (sollya.parse("0x1.d3fb9968850a5p-960"), sollya.parse("-0x0.23c1ed19c45fp-1022")),
            # bad sign of zero
            (FP_MinusZero(self.precision), sollya.parse("0x1.85200a9235193p-450")),
            # bad remainder
            (sollya.parse("0x1.fffffffffffffp+1023"), sollya.parse("0x1.1f31bcd002a7ap-803")),
            # bad sign
            (sollya.parse("-0x1.4607d0c9fc1a7p-878"), sollya.parse("-0x1.9b666b840b1bp-1023")),
        ]



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_arg=ML_RemQuo.get_default_args()
    )
    arg_template.get_parser().add_argument(
         "--quotient-size", dest="quotient_size", default=3, type=int,
        action="store", help="number of bit to return for the quotient")
    arg_template.get_parser().add_argument(
         "--quotient-mode", dest="quotient_mode", const=True, default=False,
        action="store_const", help="number of bit to return for the quotient")

    ARGS = arg_template.arg_extraction()

    ml_remquo = ML_RemQuo(ARGS)
    ml_remquo.gen_implementation()
