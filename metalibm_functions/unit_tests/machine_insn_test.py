from metalibm_core.core.ml_operations import (
    Variable, Statement,
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
            "output_file": "ut_machine_insn_generation.c",
            "function_name": "ut_machine_insn_generation",
        }
        default_args.update(kw)
        return DefaultArgTemplate(**default_args)

    def generate_scheme(self):
        var = Variable("x", precision=ML_Binary32)

        test_program = Statement(
            var + var * var
        )

        TARGET = None

        # generate basic-block
        pass_bb_gen = Pass_GenerateBasicBlock(TARGET)

        # execute_on_graph must be called to get a result BasicBlockList
        bb_list = pass_bb_gen.execute_on_graph(test_program)
        print("pre-simplify bb_list:")

        print(bb_list.get_str(depth=None, display_precision=True))

        # pass_bb_simplify = Pass_BBSimplification(TARGET)
        # bb_list =  pass_bb_simplify.execute_on_optree(bb_list)
        # print("post-simplify bb_list:")
        # print(bb_list.get_str(depth=None, display_precision=True))

        # linearizer
        machine_insn_linearizer = MachineInsnGenerator(TARGET)

        linearized_program = machine_insn_linearizer.linearize_graph(bb_list)
        print("linearized_program:")
        print(linearized_program.get_str(depth=None, display_precision=True))

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_UT_MachineInsnGeneration.get_default_args())
    args = arg_template.arg_extraction()

    ut_machine_insn_generation = ML_UT_MachineInsnGeneration(args)
    ut_machine_insn_generation.generate_scheme()
