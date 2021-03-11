import random

from metalibm_core.core.simple_scalar_function import ScalarBinaryFunction
from metalibm_core.core.ml_formats import ML_Binary32, ML_Int32
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_operations import (
    Statement, Return,
    ExponentExtraction, ExponentInsertion, MantissaExtraction,
    Conversion, Division, Multiplication,
    Variable, ReciprocalSeed, NearestInteger,
    Select, NotEqual, Equal, Modulo,
    LogicalNot, LogicalAnd, LogicalOr,
    ConditionBlock, Abs, CopySign, Constant,
    Test,
    FMA)

from metalibm_core.core.special_values import (
    FP_PlusInfty, FP_PlusZero, FP_MinusZero,
    FP_MinusInfty, FP_QNaN, FP_SpecialValue,
    is_nan, is_plus_infty, is_minus_infty, is_zero,
    is_plus_zero, is_minus_zero,
    SOLLYA_NAN, SOLLYA_INFTY)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate, DefaultArgTemplate
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_functions.ml_exp2 import ML_Exp2
from metalibm_functions.generic_log import ML_GenericLog
from metalibm_functions.ml_div import ML_Division
from metalibm_core.code_generation.code_function import FunctionGroup

import sollya
import bigfloat

class MetaRootN(ScalarBinaryFunction):
    function_name = "ml_rootn"
    arity = 2

    def __init__(self, args):
        ScalarBinaryFunction.__init__(self, args)
        # expand floating-point division
        self.division_implementation = None
        self.expand_div = args.expand_div

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetaAtan,
                builtin from a default argument mapping overloaded with @p kw
        """
        default_args_rootn = {
            "output_file": "rootn.c",
            "function_name": "rootn",
            "input_precisions": [ML_Binary32, ML_Int32],
            "accuracy": ML_Faithful,
            "input_intervals": [sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(-2**24, 2**24)],
            "auto_test_range": [sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(-2**24, 2**24)],
            "target": GenericProcessor.get_target_instance(),
            "expand_div": False,
        }
        default_args_rootn.update(kw)
        return DefaultArgTemplate(**default_args_rootn)

    def generate_function_list(self):
        self.implementation.set_scheme(self.generate_scheme())
        if self.division_implementation is None:
            return FunctionGroup([self.implementation])
        else:
            return FunctionGroup([self.implementation, self.division_implementation])

    def generate_scalar_scheme(self, vx, n):
        # fixing inputs' node tag
        vx.set_attributes(tag="x")
        n.set_attributes(tag="n")

        int_precision = self.precision.get_integer_format()

        # assuming x = m.2^e (m in [1, 2[)
        #          n, positive or null integers
        #
        # rootn(x, n) = x^(1/n)
        #             = exp(1/n * log(x))
        #             = 2^(1/n * log2(x))
        #             = 2^(1/n * (log2(m) + e))
        #

        # approximation log2(m)

        # retrieving processor inverse approximation table
        dummy_var = Variable("dummy", precision = self.precision)
        dummy_div_seed = ReciprocalSeed(dummy_var, precision = self.precision)
        inv_approx_table = self.processor.get_recursive_implementation(
            dummy_div_seed, language=None,
            table_getter= lambda self: self.approx_table_map)

        log_f = sollya.log(sollya.x) # /sollya.log(self.basis)

        use_reciprocal = False

        # non-scaled vx used to compute vx^1
        unmodified_vx = vx

        is_subnormal = Test(vx, specifier=Test.IsSubnormal, tag="is_subnormal")
        exp_correction_factor = self.precision.get_mantissa_size()
        mantissa_factor = Constant(2**exp_correction_factor, tag="mantissa_factor")
        vx = Select(is_subnormal, vx * mantissa_factor, vx, tag="corrected_vx")

        m = MantissaExtraction(vx, tag="m", precision=self.precision)
        e = ExponentExtraction(vx, tag="e", precision=int_precision)
        e = Select(is_subnormal, e -exp_correction_factor, e, tag="corrected_e")

        ml_log_args = ML_GenericLog.get_default_args(precision=self.precision, basis=2)
        ml_log = ML_GenericLog(ml_log_args)
        log_table, log_table_tho, table_index_range = ml_log.generate_log_table(log_f, inv_approx_table)
        log_approx = ml_log.generate_reduced_log_split(Abs(m, precision=self.precision), log_f, inv_approx_table, log_table)
        # floating-point version of n
        n_f = Conversion(n, precision=self.precision, tag="n_f")
        inv_n = Division(Constant(1, precision=self.precision), n_f)

        log_approx = Select(Equal(vx, 0), FP_MinusInfty(self.precision), log_approx)
        log_approx.set_attributes(tag="log_approx", debug=debug_multi)
        if use_reciprocal:
            r = Multiplication(log_approx, inv_n, tag="r", debug=debug_multi)
        else:
            r = Division(log_approx, n_f, tag="r", debug=debug_multi)

        # e_n ~ e / n
        e_f = Conversion(e, precision=self.precision, tag="e_f")
        if use_reciprocal:
            e_n = Multiplication(e_f, inv_n, tag="e_n")
        else:
            e_n = Division(e_f, n_f, tag="e_n")
        error_e_n = FMA(e_n, -n_f, e_f, tag="error_e_n")
        e_n_int = NearestInteger(e_n, precision=self.precision, tag="e_n_int")
        pre_e_n_frac = e_n - e_n_int
        pre_e_n_frac.set_attributes(tag="pre_e_n_frac")
        e_n_frac = pre_e_n_frac + error_e_n * inv_n
        e_n_frac.set_attributes(tag="e_n_frac")

        ml_exp2_args = ML_Exp2.get_default_args(precision=self.precision)
        ml_exp2 = ML_Exp2(ml_exp2_args)
        exp2_r = ml_exp2.generate_scalar_scheme(r, inline_select=True)
        exp2_r.set_attributes(tag="exp2_r", debug=debug_multi)

        exp2_e_n_frac = ml_exp2.generate_scalar_scheme(e_n_frac, inline_select=True)
        exp2_e_n_frac.set_attributes(tag="exp2_e_n_frac", debug=debug_multi)

        exp2_e_n_int = ExponentInsertion(Conversion(e_n_int, precision=int_precision), precision=self.precision, tag="exp2_e_n_int")

        n_is_even = Equal(Modulo(n, 2), 0, tag="n_is_even", debug=debug_multi)
        n_is_odd = LogicalNot(n_is_even, tag="n_is_odd")
        result_sign = Select(n_is_odd, CopySign(Constant(1.0, precision=self.precision), vx), 1)


        # managing n == -1
        if self.expand_div:
            ml_division_args = ML_Division.get_default_args(precision=self.precision, input_formats=[self.precision]*2)
            ml_division = ML_Division(ml_division_args)
            self.division_implementation = ml_division.implementation
            self.division_implementation.set_scheme(ml_division.generate_scheme())
            ml_division_fct = self.division_implementation.get_function_object()
        else:
            ml_division_fct = Division

        # manage n=1 separately to avoid catastrophic propagation of errors
        # between log2 and exp2 to eventually compute the identity function
        # test-case #3
        result = ConditionBlock(
            LogicalOr(
                LogicalOr(
                    Test(vx, specifier=Test.IsNaN),
                    Equal(n, 0)
                ),
                LogicalAnd(n_is_even, vx < 0)
            ),
            Return(FP_QNaN(self.precision)),
            Statement(
                ConditionBlock(
                    Equal(n,-1, tag="n_is_mone"),
                    #Return(Division(Constant(1, precision=self.precision), unmodified_vx, tag="div_res", precision=self.precision)),
                    Return(ml_division_fct(Constant(1, precision=self.precision), unmodified_vx, tag="div_res", precision=self.precision)),
                ),
                ConditionBlock(
                    # rootn( ±inf, n) is +∞ for even n< 0.
                    Test(vx, specifier=Test.IsInfty),
                    Statement(
                        ConditionBlock(
                            n < 0,
                            #LogicalAnd(n_is_odd, n < 0),
                            Return(Select(Test(vx, specifier=Test.IsPositiveInfty),
                                          Constant(FP_PlusZero(self.precision), precision=self.precision),
                                          Constant(FP_MinusZero(self.precision), precision=self.precision),
                                          precision=self.precision)),
                            Return(vx),
                        ),
                    ),
                ),
                ConditionBlock(
                    # rootn(±0, n) is ±∞ for odd n < 0.
                    LogicalAnd(LogicalAnd(n_is_odd, n < 0), Equal(vx, 0), tag="n_is_odd_and_neg"),
                    Return(Select(Test(vx, specifier=Test.IsPositiveZero),
                                  Constant(FP_PlusInfty(self.precision), precision=self.precision),
                                  Constant(FP_MinusInfty(self.precision), precision=self.precision),
                                  precision=self.precision)),
                ),
                ConditionBlock(
                    # rootn( ±0, n) is +∞ for even n< 0.
                    LogicalAnd(LogicalAnd(n_is_even, n < 0), Equal(vx, 0)),
                    Return(FP_PlusInfty(self.precision))
                ),
                ConditionBlock(
                    # rootn(±0, n) is +0 for even n > 0.
                    LogicalAnd(n_is_even, Equal(vx, 0)),
                    Return(vx)
                ),
                ConditionBlock(
                    Equal(n, 1),
                    Return(unmodified_vx),
                    Return(result_sign * exp2_r * exp2_e_n_int * exp2_e_n_frac)))
            )
        return result


    def numeric_emulate(self, vx, n):
        """ Numeric emulation of n-th root """
        if FP_SpecialValue.is_special_value(vx):
            if is_nan(vx):
                return FP_QNaN(self.precision)
            elif is_plus_infty(vx):
                return SOLLYA_INFTY
            elif is_minus_infty(vx):
                if int(n) % 2 == 1:
                    return vx
                else:
                    return FP_QNaN(self.precision)
            elif is_zero(vx):
                if int(n) % 2 != 0 and n < 0:
                    if is_plus_zero(vx):
                        return FP_PlusInfty(self.precision)
                    else:
                        return FP_MinusInfty(self.precision)
                elif int(n) % 2 == 0:
                    if n < 0:
                        return FP_PlusInfty(self.precision)
                    elif n > 0:
                        return FP_PlusZero(self.precision)
                return FP_QNaN(self.precision)
            else:
                raise NotImplementedError
        # OpenCL-C rootn, x < 0 and y odd: -exp2(log2(-x) / y)
        S2 = sollya.SollyaObject(2)
        if vx < 0:
            if int(n) % 2 != 0:
                if n > 0:
                    v = -bigfloat.root(sollya.SollyaObject(-vx).bigfloat(), int(n))
                else:
                    v = -S2**(sollya.log2(-vx) / n)
            else:
                return FP_QNaN(self.precision)
        elif n < 0:
            # OpenCL-C definition
            v = S2**(sollya.log2(vx) / n)
        else:
            v = bigfloat.root(sollya.SollyaObject(vx).bigfloat(), int(n))
        return sollya.SollyaObject(v)

    @property
    def standard_test_cases(self):
        general_list= [
            # ERROR: rootn: inf ulp error at {inf, -2}: *0x0p+0 vs. inf (0x7f800000) at index: 1226
            (FP_PlusInfty(self.precision), -2, FP_PlusZero(self.precision)),
            # ERROR: rootn: inf ulp error at {inf, -2147483648}: *0x0.0000000000000p+0 vs. inf
            (FP_PlusInfty(self.precision), -2147483648, FP_PlusZero(self.precision)),
            #
            (FP_PlusZero(self.precision), -1, FP_PlusInfty(self.precision)),
            (FP_MinusInfty(self.precision), 1, FP_MinusInfty(self.precision)),
            (FP_MinusInfty(self.precision), -1, FP_MinusZero(self.precision)),
            # ERROR coucou7: rootn: -inf ulp error at {inf 7f800000, 479638026}: *inf vs. 0x1.000018p+0 (0x3f80000c) at index: 2367
            (FP_PlusInfty(self.precision), 479638026, FP_PlusInfty(self.precision)),
            (FP_MinusInfty(self.precision), 479638026),
            #(FP_MinusInfty(self.precision), -479638026),
            #(FP_PlusInfty(self.precision), -479638026),
            # rootn( ±0, n) is ±∞ for odd n< 0.
            (FP_PlusZero(self.precision), -1337, FP_PlusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1337, FP_MinusInfty(self.precision)),
            # rootn( ±0, n) is +∞ for even n< 0.
            (FP_PlusZero(self.precision), -1330, FP_PlusInfty(self.precision)),
            # rootn( ±0, n) is +0 for even n> 0.
            (FP_PlusZero(self.precision), random.randrange(0, 2**31, 2), FP_PlusZero(self.precision)),
            (FP_MinusZero(self.precision), random.randrange(0, 2**31, 2), FP_PlusZero(self.precision)),
            # rootn( ±0, n) is ±0 for odd n> 0.
            (FP_PlusZero(self.precision), random.randrange(1, 2**31, 2), FP_PlusZero(self.precision)),
            (FP_MinusZero(self.precision), random.randrange(1, 2**31, 2), FP_MinusZero(self.precision)),
            # rootn( x, n) returns a NaN for x< 0 and n is even.
            (-random.random(), 2 * random.randrange(1, 2**30), FP_QNaN(self.precision)),
            # rootn( x, 0 ) returns a NaN
            (random.random(), 0, FP_QNaN(self.precision)),
            # vx=nan
            (sollya.parse("-nan"), -1811577079, sollya.parse("nan")),
            (sollya.parse("-nan"), 832501219, sollya.parse("nan")),
            (sollya.parse("-nan"), -857435762, sollya.parse("nan")),
            (sollya.parse("-nan"), -1503049611, sollya.parse("nan")),
            (sollya.parse("-nan"), 2105620996, sollya.parse("nan")),
            #ERROR: rootn: inf ulp error at {-nan, 832501219}: *-nan vs. -0x1.00000df2bed98p+1
            #ERROR: rootn: inf ulp error at {-nan, -857435762}: *-nan vs. 0x1.0000000000000p+1
            #ERROR: rootn: inf ulp error at {-nan, -1503049611}: *-nan vs. -0x1.0000000000000p+1
            #ERROR: rootn: inf ulp error at {-nan, 2105620996}: *-nan vs. 0x1.00000583c4b7ap+1
            (sollya.parse("-0x1.cd150ap-105"), 105297051),
            (sollya.parse("0x1.ec3bf8p+71"), -1650769017),
            # test-case #12
            (0.1, 17),
            # test-case #11, fails in OpenCL CTS
            (sollya.parse("0x0.000000001d600p-1022"), 14),
            # test-case #10, fails test with dar(2**-23)
            (sollya.parse("-0x1.20aadp-114"), 17),
            # test-case #9
            (sollya.parse("0x1.a44d8ep+121"), 7),
            # test-case #8
            (sollya.parse("-0x1.3ef124p+103"), 3),
            # test-case #7
            (sollya.parse("-0x1.01047ep-2"), 39),
            # test-case #6
            (sollya.parse("-0x1.0105bp+67"), 23),
            # test-case #5
            (sollya.parse("0x1.c1f72p+51"), 6),
            # special cases
            (sollya.parse("0x0p+0"), 1),
            (sollya.parse("0x0p+0"), 0),
            # test-case #3, catastrophic error for n=1
            (sollya.parse("0x1.fc61a2p-121"), 1.0),
            # test-case #4 , k=14 < 0 not supported by bigfloat
            # (sollya.parse("0x1.ad067ap-66"), -14),
        ]
        # NOTE: expected value assumed 32-bit precision output
        fp_32_only = [
            #
            (sollya.parse("0x1.80bb0ep+70"), 377778829, sollya.parse("0x1.000002p+0")),
        ]
        # NOTE: the following test-case are only valid if meta-function supports 64-bit integer
        #       2nd_input
        fp_64_only = [
            (sollya.parse("0x1.fffffffffffffp+1023"), -1, sollya.parse("0x0.4000000000000p-1022")),
            (sollya.parse("-0x1.fffffffffffffp1023"), -1, sollya.parse("-0x0.4000000000000p-1022")),
            #(sollya.parse("-0x1.fffffffffffffp+1023"), 1),
            #(sollya.parse("0x1.fffffffffffffp+1023"), -1),
            # ERROR coucou8: rootn: inf ulp error at {-inf, 1854324695}: *-inf vs. -0x1.0000066bfdd60p+0
            (FP_MinusInfty(self.precision), 1854324695, FP_MinusInfty(self.precision)),
            # ERROR: rootn: -60.962402 ulp error at {0x0.000000001d600p-1022, 14}: *0x1.67d4ff97d1fd9p-76 vs. 0x1.67d4ff97d1f9cp-76
            (sollya.parse("0x0.000000001d600p-1022"), 14, sollya.parse("0x1.67d4ff97d1fd9p-76")),
            # ERROR: rootn: -430452000.000000 ulp error at {0x1.ffffffff38c00p-306, 384017876}: *0x1.ffffed870ff01p-1 vs. 0x1.ffffebec8d1d2p-1
            (sollya.parse("0x1.ffffffff38c00p-306"), 384017876, sollya.parse("0x1.ffffed870ff01p-1")), # vs. 0x1.ffffebec8d1d2p-1
            # ERROR: rootn: 92996584.000000 ulp error at {0x1.ffffffffdae80p-858, -888750231}: *0x1.00000b36b1173p+0 vs. 0x1.00000b8f6155ep+0
            (sollya.parse("0x1.ffffffffdae80p-858"), -888750231, sollya.parse("0x1.00000b36b1173p+0")),
            # ERROR: rootn: 379474.906250 ulp error at {0x0.0000000000022p-1022, -1538297900}: *0x1.00000814a68ffp+0 vs. 0x1.0000081503352p+0
            (sollya.parse("0x0.00000006abfffp-1022"), -1221802473, sollya.parse("0x1.00000a01818a4p+0")),
            (sollya.parse("0x1.ffffffffd0a00p-260"), 1108043946, sollya.parse("0x1.fffffa9042997p-1")),
            (sollya.parse("0x1.3fffffffff1c0p-927"), -1997086266, sollya.parse("0x1.0000056564c5ep+0")),
            (sollya.parse("0x1.ffffffff38c00p-306"), 384017876, sollya.parse("0x1.ffffed870ff01p-1")),
            (sollya.parse("0x0.15c000000002ap-1022"), 740015941, sollya.parse("0x1.ffffdfc47b57ep-1")),
            (sollya.parse("0x0.00000000227ffp-1022"), -1859058847, sollya.parse("0x1.0000069c7a01bp+0")),
            (sollya.parse("0x0.0568000000012p-1022"), -447352599, sollya.parse("0x1.00001ab640c38p+0")),
            (sollya.parse("0x0.000000000000dp-1022"), 132283432, sollya.parse("0x1.ffff43d1db82ap-1")),
            (sollya.parse("-0x1.c80000000026ap+1023"), 275148531, sollya.parse("-0x1.00002b45a7314p+0")),
            (sollya.parse("0x0.022200000000ep-1022"), -1969769414, sollya.parse("0x1.000006130e858p+0")),
            (sollya.parse("0x0.0000000000011p-1022"), 851990770, sollya.parse("0x1.ffffe2cafaff6p-1")),
            (sollya.parse("0x1.8fffffffff348p-1010"), 526938360, sollya.parse("0x1.ffffd372e2b81p-1")),
            (sollya.parse("0x0.0000000000317p-1022"), -1315106194, sollya.parse("0x1.0000096973ac9p+0")),
            (sollya.parse("0x1.1ffffffff2d20p-971"), 378658008, sollya.parse("0x1.ffffc45e803b2p-1")),
            #
            (sollya.parse("0x0.0568000000012p-1022"), -447352599, sollya.parse("0x1.00001ab640c38p+0")),
            #
            (sollya.parse("0x1.ffffffffd0a00p-260"), 1108043946, sollya.parse("0x1.fffffa9042997p-1")),
            (FP_MinusZero(self.precision), -21015979, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -85403731, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -180488973, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1365227287, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1802885579, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1681209663, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1152797721, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -1614890585, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -812655517, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -628647891, FP_MinusInfty(self.precision)),
            (sollya.parse("0x1.ffffffffdae80p-858"), -888750231, sollya.parse("0x1.00000b36b1173p+0")),
            (sollya.parse("0x0.0568000000012p-1022"), -447352599, sollya.parse("0x1.00001ab640c38p+0")),
            (sollya.parse("0x0.00000006abfffp-1022"), -1221802473, sollya.parse("0x1.00000a01818a4p+0")),
            (sollya.parse("0x0.0000000000022p-1022"), -1538297900, sollya.parse("0x1.00000814a68ffp+0")),
            #ERROR: rootn: inf ulp error at {-0x0.0000000000000p+0, -1889147085}: *-inf vs. inf
            #ERROR: rootn: inf ulp error at {-0x0.0000000000000p+0, -373548013}: *-inf vs. inf
            (FP_MinusZero(self.precision), -1889147085, FP_MinusInfty(self.precision)),
            (FP_MinusZero(self.precision), -373548013, FP_MinusInfty(self.precision)),
            #ERROR: rootn: inf ulp error at {-0x0.0000000000000p+0, -1889147085}: *-inf vs. inf
            #ERROR: rootn: inf ulp error at {-0x0.0000000000000p+0, -373548013}: *-inf vs. inf
            # Cluster0@0.0: PE 0: error[84]: ml_rootn(-0x1.b1a6765727e72p-902, -7.734955e+08/-773495525), result is -0x1.00000d8cb5b3cp+0 vs expected [nan;nan]
            (sollya.parse("-0x1.b1a6765727e72p-902"), -773495525),
            # ERROR: rootn: -40564819207303340847894502572032.000000 ulp error at {-0x0.fffffffffffffp-1022, 1}: *-0x0.fffffffffffffp-1022 vs. -0x1.ffffffffffffep-970
            (sollya.parse("-0x0.fffffffffffffp-1022 "), 1, sollya.parse("-0x0.fffffffffffffp-1022 ")),
            # ERROR: rootn: 1125899906842624.000000 ulp error at {-0x1.fffffffffffffp+1023, -1}: *-0x0.4000000000000p-1022 vs. -0x0.0000000000000p+0
            (sollya.parse("-0x1.fffffffffffffp+1023"), -1, sollya.parse("-0x0.4000000000000p-1022")),
            (sollya.parse("0x1.fffffffffffffp+1023"), -1, sollya.parse("0x0.4000000000000p-1022")),
            (sollya.parse("0x1.8d8f7b2e21fdbp+70"), 5.227689e+06, None),
        ]

        return (fp_64_only if self.precision.get_bit_size() >= 64 else []) \
               + (fp_32_only if self.precision.get_bit_size() == 32 else []) \
               + general_list


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate(default_arg=MetaRootN.get_default_args())

    arg_template.get_parser().add_argument(
         "--expand-div", dest="expand_div", default=False, const=int,
        action="store_const", help="expand division to meta-division")

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

    # declaring meta-function instance
    meta_function = MetaRootN(args)

    # generating meta_function
    meta_function.gen_implementation()
