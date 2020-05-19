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



from metalibm_functions.unit_tests.utils import TestRunner


class UT_MachineInsnGeneration(ML_FunctionBasis, TestRunner):
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
                # default pass for dummy asm target are enough
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

    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_machine_insn_gen = UT_MachineInsnGeneration(args)
        ut_machine_insn_gen.gen_implementation()
        return True

# main runner for unit tests (called by valid.soft_unit_test)
run_test = UT_MachineInsnGeneration.__call__

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=UT_MachineInsnGeneration.get_default_args())
    args = arg_template.arg_extraction()

    ut_machine_insn_generation = UT_MachineInsnGeneration(args)
    if run_test(args):
        exit(0)
    else:
        exit(1)
