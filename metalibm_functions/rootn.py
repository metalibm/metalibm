from metalibm_core.core.simple_scalar_function import ScalarBinaryFunction
from metalibm_core.core.ml_formats import ML_Binary32, ML_Int32
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_operations import (
    Statement, Return,
    ExponentExtraction, ExponentInsertion, MantissaExtraction,
    Conversion, Division,
    Variable, ReciprocalSeed, NearestInteger,
    Select, Equal, Modulo,
    LogicalNot, LogicalAnd, LogicalOr,
    ConditionBlock, Abs, CopySign, Constant)

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

class MetaRootN(ScalarBinaryFunction):
    function_name = "ml_rootn"
    arity = 2

    def __init__(self, args):
        ScalarBinaryFunction.__init__(self, args)

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
            "input_intervals": [sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 2**31-1)],
            "auto_test_range": [sollya.Interval(-2.0**126, 2.0**126), sollya.Interval(0, 2**31-1)],
            "target": GenericProcessor.get_target_instance()
        }
        default_args_rootn.update(kw)
        return DefaultArgTemplate(**default_args_rootn)

    def generate_scalar_scheme(self, vx, n):
        # fixing inputs' node tag
        vx.set_attributes(tag="x")
        n.set_attributes(tag="n")

        # assuming x = m.2^e (m in [1, 2[)
        #          n, positive or null integers
        #
        # rootn(x, n) = x^(1/n)
        #             = exp(1/n * log(x))
        #             = 2^(1/n * log2(x))
        #             = 2^(1/n * (log2(m) + e))
        #
        e = ExponentExtraction(vx, tag="e")
        m = MantissaExtraction(vx, tag="m")

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
        log_approx = ml_log.generate_reduced_log(Abs(vx), log_f, inv_approx_table,
                                                 log_table, log_table_tho)

        log_approx = Select(Equal(vx, 0), FP_MinusInfty(self.precision), log_approx)
        log_approx.set_attributes(tag="log_approx", debug=debug_multi)
        r = Division(log_approx, Conversion(n, precision=self.precision))
        r.set_attributes(tag="r", debug=debug_multi)

        ml_exp2_args = ML_Exp2.get_default_args(precision=self.precision)
        ml_exp2 = ML_Exp2(ml_exp2_args)
        ml_exp2_scheme = ml_exp2.generate_scalar_scheme(r, inline_select=True)
        ml_exp2_scheme.set_attributes(tag="exp2_r", debug=debug_multi)

        n_is_odd = Equal(Modulo(n, 2), 1, tag="n_is_odd", debug=debug_multi)
        result_sign = Select(n_is_odd, CopySign(vx, Constant(1.0, precision=self.precision)), 1)

        # manage n=1 separately to avoid catastrophic propagation of errors
        # between log2 and exp2 to eventually compute the identity function
        # test-case #3
        result = ConditionBlock(
            LogicalOr(Equal(n, 0), LogicalAnd(LogicalNot(n_is_odd), vx < 0)),
            Return(FP_QNaN(self.precision)),
            ConditionBlock(
                Equal(n, 1),
                Return(vx),
                Return(result_sign * ml_exp2_scheme)))
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
                return FP_QNaN(self.precision)
            else:
                raise NotImplementedError
        v = bigfloat.root(sollya.SollyaObject(vx).bigfloat(), int(n))
        return sollya.SollyaObject(v)

    standard_test_cases = [
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


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate(default_arg=MetaRootN.get_default_args())

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

    # declaring meta-function instance
    meta_function = MetaRootN(args)

    # generating meta_function
    meta_function.gen_implementation()
