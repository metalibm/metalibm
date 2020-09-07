# importing parent class for all meta-functions
from metalibm_core.core.ml_function import ML_FunctionBasis
# importing single-precision format
from metalibm_core.core.ml_formats import ML_Binary32
# importing basic MDL node classes
from metalibm_core.core.ml_operations import Statement, Return
# importing main command argument class for metalibm
from metalibm_core.utility.ml_template import ML_NewArgTemplate

class NewMetaFunction(ML_FunctionBasis):
    function_name = "new_meta_function"

    def generate_scheme(self):
        # declare a new input parameters vx whose tag is "x" and
        # whose format is single precision
        vx = self.implementation.add_input_variable("x", self.get_input_precision(0))

        # declare a new input parameters vy whose tag is "y" and
        # whose format is single precision
        vy = self.implementation.add_input_variable("x", self.get_input_precision(0))

        # declare main operation graph for the meta-function:
        # a single Statement containing a single return statement which
        # the addition of the two inputs variable in single-precision
        main_scheme = Statement(
            Return(vx + vy, precision=ML_Binary32)
        )
        return main_scheme


if __name__ == "__main__":
    # declaring standard argument structure
    arg_template = ML_NewArgTemplate()

    # filling arg_template structure with command line options
    args = arg_template.arg_extraction()

    # declaring meta-function instance
    meta_function = NewMetaFunction(args)

    # generating meta_function
    meta_function.gen_implementation()
