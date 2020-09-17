from metalibm_core.core.simple_scalar_function import ScalarBinaryFunction
from metalibm_core.core.ml_formats import ML_Binary32, ML_Int32
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_operations import (
    Statement, Return,
    ExponentExtraction, ExponentInsertion, MantissaExtraction,
    Conversion, Division,
    Variable, ReciprocalSeed, NearestInteger,
    Select, Equal)

from metalibm_core.core.special_values import FP_MinusInfty

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
            "input_intervals": [None, None],
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
        log_approx = ml_log.generate_reduced_log(vx, log_f, inv_approx_table,
                                                 log_table, log_table_tho)

        log_approx = Select(Equal(vx, 0), FP_MinusInfty(self.precision), log_approx)
        log_approx.set_attributes(tag="log_approx")
        r = Division(log_approx, Conversion(n, precision=self.precision), tag="r", debug=debug_multi)

        ml_exp2_args = ML_Exp2.get_default_args(precision=self.precision)
        ml_exp2 = ML_Exp2(ml_exp2_args)
        ml_exp2_scheme = ml_exp2.generate_scalar_scheme(r)

        return ml_exp2_scheme


    def numeric_emulate(self, vx, n):
        """ Numeric emulation of n-th root """
        v = bigfloat.root(sollya.SollyaObject(vx).bigfloat(), int(n))
        print(v)
        return sollya.SollyaObject(v)

    standard_test_cases = [
        (sollya.parse("0x0p+0"), 1),
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
