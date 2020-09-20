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
    FMA)

from metalibm_core.core.special_values import (
    FP_MinusInfty, FP_QNaN, FP_SpecialValue,
    is_nan, is_plus_infty, is_minus_infty, is_zero,
    SOLLYA_NAN, SOLLYA_INFTY)

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate, DefaultArgTemplate
from metalibm_core.utility.debug_utils import debug_multi

from metalibm_functions.ml_exp2 import ML_Exp2
from metalibm_functions.generic_log import ML_GenericLog

import sollya
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
            "input_intervals": [None, None], # sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 2**31-1)],
            "auto_test_range": [None, None], # sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 47)],
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

        # e_y ~ e * y
        e_f = Conversion(e, precision=self.precision)
        e_y = Multiplication(e_f, vy, tag="e_y")
        e_y_int = NearestInteger(e_y, precision=self.precision, tag="e_y_int")
        e_y_frac = e_y - e_y_int
        e_y_frac.set_attributes(tag="pre_e_y_frac")

        ml_exp2_args = ML_Exp2.get_default_args(precision=self.precision)
        ml_exp2 = ML_Exp2(ml_exp2_args)
        exp2_r = ml_exp2.generate_scalar_scheme(r, inline_select=True)
        exp2_r.set_attributes(tag="exp2_r", debug=debug_multi)

        exp2_e_y_frac = ml_exp2.generate_scalar_scheme(e_y_frac, inline_select=True)
        exp2_e_y_frac.set_attributes(tag="exp2_e_y_frac", debug=debug_multi)

        exp2_e_y_int = ExponentInsertion(Conversion(e_y_int, precision=int_precision), precision=self.precision, tag="exp2_e_y_int")

        result_sign = Constant(1.0, precision=self.precision) # Select(n_is_odd, CopySign(vx, Constant(1.0, precision=self.precision)), 1)

        # manage n=1 separately to avoid catastrophic propagation of errors
        # between log2 and exp2 to eventually compute the identity function
        # test-case #3
        result = Statement(
            ConditionBlock(
                LogicalAnd(Equal(vx, 0), Equal(vy, 0)),
                Return(Constant(1.0, precision=self.precision)),
                Return(result_sign * exp2_r * exp2_e_y_int * exp2_e_y_frac)))
        return result


    def numeric_emulate(self, vx, vy):
        """ Numeric emulation of pow """
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
                return FP_QNaN(self.precision)
            else:
                raise NotImplementedError
        return sollya.SollyaObject(vx)**sollya.SollyaObject(vy)

    standard_test_cases = [
        # test-case #0
        (sollya.parse("0x1.5d20b8p-115"), sollya.parse("0x1.c20048p+0")),
        # special cases
        (sollya.parse("0x0p+0"), 1),
        (sollya.parse("0x0p+0"), 0),
    ]


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate(default_arg=MetaPow.get_default_args())

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

    # declaring meta-function instance
    meta_function = MetaPow(args)

    # generating meta_function
    meta_function.gen_implementation()
