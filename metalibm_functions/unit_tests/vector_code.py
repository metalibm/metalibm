# -*- coding: utf-8 -*-

import sys

from sollya import S2, Interval

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import * 

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  


class ML_UT_VectorCode(ML_Function("ml_ut_vector_code")):
  def __init__(self, args=DefaultArgTemplate): 
    # initializing base class
    ML_FunctionBasis.__init__(self, args) 


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_vector_code.c",
        "function_name": "ut_vector_code",
        "precision": ML_Binary32,
        "target": FixedPointBackend(),
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)

  def generate_function_list(self):
    # declaring function input variable
    vx = self.implementation.add_input_variable("x", self.precision)
    vy = self.implementation.add_input_variable("y", self.precision)
    # declaring specific interval for input variable <x>
    vx.set_interval(Interval(-1, 1))


    vec = Variable("vec", precision = v2float32, var_type = Variable.Local)

    vec2 = Multiplication(vec, vec, precision = v2float32)
    vec3 = Addition(vec, vec2, precision = v2float32)

    result = Addition(vec3[0], vec3[1], precision = ML_Binary32)

    scheme = Statement(
      ReferenceAssign(vec[0], vx),
      ReferenceAssign(vec[1], vy),
      Return(result)
    )

    # dummy scheme to make functionnal code generation
    self.implementation.set_scheme(scheme)

    return [self.implementation]


def run_test(args):
  ml_ut_vector_code = ML_UT_VectorCode(args)
  ml_ut_vector_code.gen_implementation()
  return True

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_args=ML_UT_VectorCode.get_default_args())
  args = arg_template.arg_extraction()


  if run_test(args):
    exit(0)
  else:
    exit(1)


