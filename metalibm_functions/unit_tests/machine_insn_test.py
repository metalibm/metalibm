from metalibm_core.core.ml_operations import (
    Variable, Statement, Multiplication, Addition,
    Return,
)
from metalibm_core.core.ml_formats import ML_Binary32

from metalibm_core.code_generation.machine_program_linearizer import MachineInsnGenerator

from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)


from metalibm_core.opt.p_gen_bb import (
    Pass_GenerateBasicBlock, Pass_BBSimplification,
    Pass_SSATranslate
)

from metalibm_core.code_generation.asmde_translator import AssemblySynthesizer


class ML_UT_MachineInsnGeneration(ML_FunctionBasis):
    function_basis = "mt_ut_machine_insn_generation"

    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        ML_FunctionBasis.__init__(self, args)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for current class,
            builtin from a default argument mapping overloaded with @p kw """
        default_args = {
            "output_file": "ut_machine_insn_generation.S",
            "function_name": "ut_machine_insn_generation",
            "precision": ML_Binary32,
            "passes": [
                "start:gen_basic_block",
                "start:basic_block_simplification",
                "start:dump",
                "start:linearize_op_graph",
                "start:dump",
                "start:register_allocation",
                "start:dump",
            ],
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        var = self.implementation.add_input_variable("x", self.precision)
        var_y = self.implementation.add_input_variable("y", self.precision)
        var_z = self.implementation.add_input_variable("z", self.precision)
        mult = Multiplication(var, var_z, precision=self.precision)
        add = Addition(var_y, mult, precision=self.precision)

        test_program = Statement(
            add,
            Return(add)
        )
        return test_program

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MachineInsnGeneration.get_default_args())
    args = arg_template.arg_extraction()

    ut_machine_insn_generation = ML_UT_MachineInsnGeneration(args)
    #ut_machine_insn_generation.generate_scheme()
    ut_machine_insn_generation.gen_implementation()
