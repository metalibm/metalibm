from metalibm_core.core.simple_scalar_function import ScalarBinaryFunction
from metalibm_core.core.ml_formats import ML_Binary32, ML_Int32
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_operations import (
    Statement, Return,
    ExponentExtraction, ExponentInsertion, MantissaExtraction,
    Conversion, Division, Multiplication,
    Variable, ReciprocalSeed, NearestInteger,
    Select, Equal, Modulo,
    LogicalNot, LogicalAnd, LogicalOr,
    ConditionBlock, Abs, CopySign, Constant,
    Test,
    FMA)

from metalibm_core.core.special_values import (
    FP_PlusInfty, FP_MinusInfty,
    FP_QNaN, FP_SpecialValue,
    FP_PlusZero, FP_MinusZero,
    is_nan, is_plus_infty, is_minus_infty, is_zero,
    is_plus_zero, is_minus_zero, is_infty,
    is_snan,
    SOLLYA_NAN, SOLLYA_INFTY)
# alias
is_special_value = FP_SpecialValue.is_special_value

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate, DefaultArgTemplate
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_functions.ml_exp2 import ML_Exp2
from metalibm_functions.generic_log import ML_GenericLog

import sollya
from sollya import Interval
import bigfloat

class MetaPow(ScalarBinaryFunction):
    function_name = "ml_pow"
    arity = 2

    def __init__(self, args):
        ScalarBinaryFunction.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetaAtan,
                builtin from a default argument mapping overloaded with @p kw
        """
        default_args_pow = {
            "output_file": "ml_pow.c",
            "function_name": "ml_pow",
            "input_precisions": [ML_Binary32, ML_Binary32],
            "accuracy": ML_Faithful,
            "input_intervals": [None, Interval(-2**24, 2**24)], # sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 2**31-1)],
            "auto_test_range": [None, Interval(-2**24, 2**24)], # sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 47)],
            "target": GenericProcessor.get_target_instance()
        }
        default_args_pow.update(kw)
        return DefaultArgTemplate(**default_args_pow)

    def generate_scalar_scheme(self, vx, vy):
        # fixing inputs' node tag
        vx.set_attributes(tag="x")
        vy.set_attributes(tag="y")

        int_precision = self.precision.get_integer_format()

        # assuming x = m.2^e (m in [1, 2[)
        #          n, positive or null integers
        #
        # pow(x, n) = x^(y)
        #             = exp(y * log(x))
        #             = 2^(y * log2(x))
        #             = 2^(y * (log2(m) + e))
        #
        e = ExponentExtraction(vx, tag="e", precision=int_precision)
        m = MantissaExtraction(vx, tag="m", precision=self.precision)

        # approximation log2(m)

        # retrieving processor inverse approximation table
        dummy_var = Variable("dummy", precision = self.precision)
        dummy_div_seed = ReciprocalSeed(dummy_var, precision = self.precision)
        inv_approx_table = self.processor.get_recursive_implementation(
            dummy_div_seed, language=None,
            table_getter= lambda self: self.approx_table_map)

        log_f = sollya.log(sollya.x) # /sollya.log(self.basis)



        ml_log_args = ML_GenericLog.get_default_args(precision=self.precision, basis=2)
        ml_log = ML_GenericLog(ml_log_args)
        log_table, log_table_tho, table_index_range = ml_log.generate_log_table(log_f, inv_approx_table)
        log_approx = ml_log.generate_reduced_log_split(Abs(m, precision=self.precision), log_f, inv_approx_table, log_table)

        log_approx = Select(Equal(vx, 0), FP_MinusInfty(self.precision), log_approx)
        log_approx.set_attributes(tag="log_approx", debug=debug_multi)
        r = Multiplication(log_approx, vy, tag="r", debug=debug_multi)


        # 2^(y * (log2(m) + e)) = 2^(y * log2(m)) * 2^(y * e)
        #
        # log_approx = log2(Abs(m))
        # r = y * log_approx ~ y * log2(m)
        #
        # NOTES: manage cases where e is negative and
        # (y * log2(m)) AND (y * e) could cancel out
        # if e positive, whichever the sign of y (y * log2(m)) and (y * e) CANNOT
        # be of opposite signs

        # log2(m) in [0, 1[ so cancellation can occur only if e == -1
        # we split 2^x in 2^x = 2^t0 * 2^t1
        # if e < 0: t0 = y * (log2(m) + e), t1=0
        # else:     t0 = y * log2(m), t1 = y * e

        t_cond = e < 0

        # e_y ~ e * y
        e_f = Conversion(e, precision=self.precision)
        #t0 = Select(t_cond, (e_f + log_approx) * vy, Multiplication(e_f, vy), tag="t0")
        #NearestInteger(t0, precision=self.precision, tag="t0_int")

        EY = NearestInteger(e_f * vy, tag="EY", precision=self.precision)
        LY = NearestInteger(log_approx * vy, tag="LY", precision=self.precision)
        t0_int = Select(t_cond, EY + LY, EY, tag="t0_int")
        t0_frac = Select(t_cond, FMA(e_f, vy, -EY) + FMA(log_approx, vy, -LY) ,EY - t0_int, tag="t0_frac")
        #t0_frac.set_attributes(tag="t0_frac")

        ml_exp2_args = ML_Exp2.get_default_args(precision=self.precision)
        ml_exp2 = ML_Exp2(ml_exp2_args)

        exp2_t0_frac = ml_exp2.generate_scalar_scheme(t0_frac, inline_select=True)
        exp2_t0_frac.set_attributes(tag="exp2_t0_frac", debug=debug_multi)

        exp2_t0_int = ExponentInsertion(Conversion(t0_int, precision=int_precision), precision=self.precision, tag="exp2_t0_int")

        t1 = Select(t_cond, Constant(0, precision=self.precision), r)
        exp2_t1 = ml_exp2.generate_scalar_scheme(t1, inline_select=True)
        exp2_t1.set_attributes(tag="exp2_t1", debug=debug_multi)

        result_sign = Constant(1.0, precision=self.precision) # Select(n_is_odd, CopySign(vx, Constant(1.0, precision=self.precision)), 1)

        y_int = NearestInteger(vy, precision=self.precision)
        y_is_integer = Equal(y_int, vy)
        y_is_even = LogicalOr(
            # if y is a number (exc. inf) greater than 2**mantissa_size * 2,
            # then it is an integer multiple of 2 => even
            Abs(vy) >= 2**(self.precision.get_mantissa_size()+1),
            LogicalAnd(
                y_is_integer and Abs(vy) < 2**(self.precision.get_mantissa_size()+1),
                # we want to limit the modulo computation to an integer input
                Equal(Modulo(Conversion(y_int, precision=int_precision), 2), 0)
            )
        )
        y_is_odd = LogicalAnd(
            LogicalAnd(
                Abs(vy) < 2**(self.precision.get_mantissa_size()+1),
                y_is_integer
            ),
            Equal(Modulo(Conversion(y_int, precision=int_precision), 2), 1)
        )


        # special cases management
        special_case_results = Statement(
            # x is sNaN OR y is sNaN
            ConditionBlock(
                LogicalOr(Test(vx, specifier=Test.IsSignalingNaN), Test(vy, specifier=Test.IsSignalingNaN)),
                Return(FP_QNaN(self.precision))
            ),
            # pow(x, ±0) is 1 if x is not a signaling NaN
            ConditionBlock(
                Test(vy, specifier=Test.IsZero),
                Return(Constant(1.0, precision=self.precision))
            ),
            # pow(±0, y) is ±∞ and signals the divideByZero exception for y an odd integer <0
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), LogicalAnd(y_is_odd, vy < 0)),
                Return(Select(Test(vx, specifier=Test.IsPositiveZero), FP_PlusInfty(self.precision), FP_MinusInfty(self.precision))),
            ),
            # pow(±0, −∞) is +∞ with no exception
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), Test(vy, specifier=Test.IsNegativeInfty)),
                Return(FP_MinusInfty(self.precision)),
            ),
            # pow(±0, +∞) is +0 with no exception
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), Test(vy, specifier=Test.IsPositiveInfty)),
                Return(FP_PlusInfty(self.precision)),
            ),
            # pow(±0, y) is ±0 for finite y>0 an odd integer
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), LogicalAnd(y_is_odd, vy > 0)),
                Return(vx),
            ),
            # pow(−1, ±∞) is 1 with no exception
            ConditionBlock(
                LogicalAnd(Equal(vx, -1), Test(vy, specifier=Test.IsInfty)),
                Return(Constant(1.0, precision=self.precision)),
            ),
            # pow(+1, y) is 1 for any y (even a quiet NaN)
            ConditionBlock(
                vx == 1,
                Return(Constant(1.0, precision=self.precision)),
            ),
            # pow(x, +∞) is +0 for −1<x<1
            ConditionBlock(
                LogicalAnd(Abs(vx) < 1, Test(vy, specifier=Test.IsPositiveInfty)),
                Return(FP_PlusZero(self.precision))
            ),
            # pow(x, +∞) is +∞ for x<−1 or for 1<x (including ±∞)
            ConditionBlock(
                LogicalAnd(Abs(vx) > 1, Test(vy, specifier=Test.IsPositiveInfty)),
                Return(FP_PlusInfty(self.precision))
            ),
            # pow(x, −∞) is +∞ for −1<x<1
            ConditionBlock(
                LogicalAnd(Abs(vx) < 1, Test(vy, specifier=Test.IsNegativeInfty)),
                Return(FP_PlusInfty(self.precision))
            ),
            # pow(x, −∞) is +0 for x<−1 or for 1<x (including ±∞)
            ConditionBlock(
                LogicalAnd(Abs(vx) > 1, Test(vy, specifier=Test.IsNegativeInfty)),
                Return(FP_PlusZero(self.precision))
            ),
            # pow(+∞, y) is +0 for a number y < 0
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsPositiveInfty), vy < 0),
                Return(FP_PlusZero(self.precision))
            ),
            # pow(+∞, y) is +∞ for a number y > 0
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsPositiveInfty), vy > 0),
                Return(FP_PlusInfty(self.precision))
            ),
            # pow(−∞, y) is −0 for finite y < 0 an odd integer
            # TODO: check y is finite
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsNegativeInfty), LogicalAnd(y_is_odd, vy < 0)),
                Return(FP_MinusZero(self.precision)),
            ),
            # pow(−∞, y) is −∞ for finite y > 0 an odd integer
            # TODO: check y is finite
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsNegativeInfty), LogicalAnd(y_is_odd, vy > 0)),
                Return(FP_MinusInfty(self.precision)),
            ),
            # pow(−∞, y) is +0 for finite y < 0 and not an odd integer
            # TODO: check y is finite
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsNegativeInfty), LogicalAnd(LogicalNot(y_is_odd), vy < 0)),
                Return(FP_PlusZero(self.precision)),
            ),
            # pow(−∞, y) is +∞ for finite y > 0 and not an odd integer
            # TODO: check y is finite
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsNegativeInfty), LogicalAnd(LogicalNot(y_is_odd), vy > 0)),
                Return(FP_PlusInfty(self.precision)),
            ),
            # pow(±0, y) is +∞ and signals the divideByZero exception for finite y<0 and not an odd integer
            # TODO: signal divideByZero exception
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), LogicalAnd(LogicalNot(y_is_odd), vy < 0)),
                Return(FP_PlusInfty(self.precision)),
            ),
            # pow(±0, y) is +0 for finite y>0 and not an odd integer
            ConditionBlock(
                LogicalAnd(Test(vx, specifier=Test.IsZero), LogicalAnd(LogicalNot(y_is_odd), vy > 0)),
                Return(FP_PlusZero(self.precision)),
            ),
        )

        # manage n=1 separately to avoid catastrophic propagation of errors
        # between log2 and exp2 to eventually compute the identity function
        # test-case #3
        result = Statement(
            special_case_results,
            # fallback default cases
            Return(result_sign * exp2_t1 * exp2_t0_int * exp2_t0_frac))
        return result


    def numeric_emulate(self, vx, vy):
        """ Numeric emulation of pow """

        if is_snan(vx) or is_snan(vy):
            return FP_QNaN(self.precision)
        # pow(x, ±0) is 1 if x is not a signaling NaN
        if is_zero(vy):
            return 1.0
        # pow(±0, y) is ±∞ and signals the divideByZero exception for y an odd integer <0
        if is_plus_zero(vx) and not is_special_value(vy) and (int(vy) == vy) and (int(vy) % 2 == 1) and vy < 0:
            return FP_PlusInfty(self.precision)
        if is_minus_zero(vx) and not is_special_value(vy) and (int(vy) == vy) and (int(vy) % 2 == 1) and vy < 0:
            return FP_MinusInfty(self.precision)
        # pow(±0, −∞) is +∞ with no exception
        if is_zero(vx) and is_minus_zero(vy):
            return FP_MinusInfty(self.precision)
        # pow(±0, +∞) is +0 with no exception
        if is_zero(vx) and is_plus_zero(vy):
            return FP_PlusZero(self.precision)
        # pow(±0, y) is ±0 for finite y>0 an odd integer
        if is_zero(vx) and not is_special_value(vy) and (int(vy) == vy) and (int(vy) % 2 == 1) and vy > 0:
            return vx
        # pow(−1, ±∞) is 1 with no exception
        if vx == -1.0 and is_infty(vy):
            return 1
        # pow(+1, y) is 1 for any y (even a quiet NaN)
        if vx == 1.0:
            return 1.0
        # pow(x, +∞) is +0 for −1<x<1
        if -1 < vx < 1 and is_plus_infty(vy):
            return FP_PlusZero(self.precision)
        # pow(x, +∞) is +∞ for x<−1 or for 1<x (including ±∞)
        if (vx < -1 or vx > 1) and is_plus_infty(vy):
            return FP_PlusInfty(self.precision)
        # pow(x, −∞) is +∞ for −1<x<1
        if -1 < vx < 1 and is_minus_infty(vy):
            return FP_PlusInfty(self.precision)
        # pow(x, −∞) is +0 for x<−1 or for 1<x (including ±∞)
        if is_minus_infty(vy) and (vx < -1 or vx > 1):
            return FP_PlusZero(self.precision)
        # pow(+∞, y) is +0 for a number y < 0
        if is_plus_infty(vx) and vy < 0:
            return FP_PlusZero(self.precision)
        # pow(+∞, y) is +∞ for a number y > 0
        if is_plus_infty(vx) and vy > 0:
            return FP_PlusInfty(self.precision)
        # pow(−∞, y) is −0 for finite y < 0 an odd integer
        if is_minus_infty(vx) and vy < 0 and not is_special_value(vy) and int(vy) == vy and int(vy) % 2 == 1:
            return FP_MinusZero(self.precision)
        # pow(−∞, y) is −∞ for finite y > 0 an odd integer
        if is_minus_infty(vx) and vy > 0 and not is_special_value(vy) and  int(vy) == vy and int(vy) % 2 == 1:
            return FP_MinusInfty(self.precision)
        # pow(−∞, y) is +0 for finite y < 0 and not an odd integer
        if is_minus_infty(vx) and vy < 0 and not is_special_value(vy) and not(int(vy) == vy and int(vy) % 2 == 1):
            return FP_PlusZero(self.precision)
        # pow(−∞, y) is +∞ for finite y > 0 and not an odd integer
        if is_minus_infty(vx) and vy > 0 and not is_special_value(vy) and not(int(vy) == vy and int(vy) % 2 == 1):
            return FP_PlusInfty(self.precision)
        # pow(±0, y) is +∞ and signals the divideByZero exception for finite y<0 and not an odd integer
        if is_zero(vx) and vy < 0 and not is_special_value(vy) and not(int(vy) == vy and int(vy) % 2 == 1):
            return FP_PlusInfty(self.precision)
        # pow(±0, y) is +0 for finite y>0 and not an odd integer
        if is_zero(vx) and vy > 0 and not is_special_value(vy) and not(int(vy) == vy and int(vy) % 2 == 1):
            return FP_PlusZero(self.precision)
        # TODO
        # pow(x, y) signals the invalid operation exception for finite x<0 and finite non-integer y.
        return sollya.SollyaObject(vx)**sollya.SollyaObject(vy)

    @property
    def standard_test_cases(self):
        generic_list = [
            # test-case #1
            (sollya.parse("0x1.bbe2f2p-1"), sollya.parse("0x1.2d34ep+9")),
            # test-case #2
            (sollya.parse("0x0p+0"), sollya.parse("0x1.a45a2ep-56"), FP_PlusZero(self.precision)),
            # test-case #0
            (sollya.parse("0x1.5d20b8p-115"), sollya.parse("0x1.c20048p+0")),
            # special cases
            (sollya.parse("0x0p+0"), 1),
            (sollya.parse("0x0p+0"), 0),
        ]
        fp64_list = [
            # subnormal output
            (sollya.parse("0x1.21998d0c5039bp-976"), sollya.parse("0x1.bc68e3d0ffd24p+3")),
        ]
        return generic_list + (fp64_list if self.precision.get_bit_size() >= 64 else [])


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate(default_arg=MetaPow.get_default_args())

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

    # declaring meta-function instance
    meta_function = MetaPow(args)

    # generating meta_function
    meta_function.gen_implementation()
