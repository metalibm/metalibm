from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import Variable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.code_generation.code_object import NestedCode
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import FunctionOperator

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.common import ML_NotImplemented


def libc_naming(base_name, io_precisions, in_arity = 1, out_arity = 1):
  precision_map = {
    ML_Binary32: "sf",
    ML_Binary64: "df",
    ML_Int32: "si",
    ML_Int64: "sd",
    ML_UInt32: "ui",
    ML_UInt64: "ud"
  }
  format_list = [io_precisions[0]] * (in_arity + out_arity) if len(io_precisions) == 1 else io_precisions
  format_suffix = ""
  previous_format = None
  counter = 0
  for precision in format_list:
    if precision != previous_format:
      if counter != 0:
        format_suffix += "%s%d" % (precision_map[previous_format], counter)
      counter = 0
      previous_format = precision
    counter += 1
  return base_name + format_suffix
  


class ML_FunctionBasis(object):
  """A class from which all metafunction inherit"""
  name = "function_basis"

  def __init__(self,
             # Naming
             base_name = "unknown_function",
             function_name=None,
             output_file = None,
             # Specification
             io_precisions = [ML_Binary32], 
             abs_accuracy = None,
             libm_compliant = True,
             # Optimization parameters
             processor = GenericProcessor(),
             fuse_fma = True, 
             fast_path_extract = True,
             # Debug verbosity
             debug_flag = False
         ):
    # io_precisions must be a list
    #     -> with a single element
    # XOR -> with as many elements as function arity (input + output arities)
    self.io_precisions = io_precisions

    # Naming logic, using provided information if available, otherwise deriving from base_name
    # base_name is e.g. exp
    # function_name is e.g. expf or expd or whatever 
    self.function_name = function_name if function_name else libc_naming(base_name, self.io_precisions)

    self.output_file = output_file if output_file else self.function_name + ".c"

    self.debug_flag=debug_flag

    # TODO: FIX which i/o precision to select
    self.sollya_precision = self.get_output_precision().sollya_object

    self.abs_accuracy = abs_accuracy if abs_accuracy else S2**(-self.get_output_precision().get_precision())

    self.libm_compliant = libm_compliant
    
    self.processor = processor

    self.fuse_fma = fuse_fma
    self.fast_path_extract = fast_path_extract

    self.implementation = CodeFunction(self.function_name, output_format = self.get_output_precision())
    self.opt_engine = OptimizationEngine(self.processor)
  
  def generate_emulate(self):
    raise ML_NotImplemented()


  def generate_emulate_wrapper(self, test_input   = Variable("vx", precision = ML_Mpfr_t), mpfr_rnd = Variable("rnd", precision = ML_Int32), test_output = Variable("result", precision = ML_Mpfr_t)):
    scheme = self.generate_emulate(test_output, test_input, mpfr_rnd)

    wrapper_processor = MPFRProcessor()

    code_generator = CCodeGenerator(wrapper_processor, declare_cst = False, disable_debug = True, libm_compliant = self.libm_compliant)
    code_object = NestedCode(code_generator, static_cst = True)
    code_generator.generate_expr(code_object, scheme, folded = False, initial = False)
    return code_object, code_generator


  def get_output_precision(self):
    return self.io_precisions[0]

  def get_input_precision(self):
    return self.io_precisions[-1]


  def get_sollya_precision(self):
    """ return the main precision use for sollya calls """
    return self.sollya_precision


  def generate_scheme(self):
    """ generate MDL scheme for function implementation """
    Log.report(Log.Error, "generate_scheme must be overloaded by ML_FunctionBasis child")

  def optimise_scheme(self, scheme):
    """ default scheme optimization """
    # fusing FMA
    print "MDL fusing FMA"
    scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

    print "MDL abstract scheme"
    self.opt_engine.instantiate_abstract_precision(scheme, None)

    #print scheme.get_str(depth = None, display_precision = True)

    print "MDL instantiated scheme"
    self.opt_engine.instantiate_precision(scheme, default_precision = ML_Binary32)


    print "subexpression sharing"
    self.opt_engine.subexpression_sharing(scheme)

    print "silencing operation"
    self.opt_engine.silence_fp_operations(scheme)

    print "checking processor support"
    self.opt_engine.check_processor_support(scheme)

    return scheme

  def generate_C(self, scheme):
    """ Final C generation, once the evaluation scheme has been optimized"""
    # registering scheme as function implementation
    self.implementation.set_scheme(scheme)
    self.C_code_generator = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not self.debug_flag, libm_compliant = self.libm_compliant)
    self.result = self.implementation.get_definition(self.C_code_generator, C_Code, static_cst = True)
    self.result.add_header("support_lib/ml_special_values.h")
    self.result.add_header("math.h")
    self.result.add_header("stdio.h")
    self.result.add_header("inttypes.h")
    #print self.result.get(self.C_code_generator)

    Log.report(Log.Info, "Generating C code in " + self.implementation.get_name() + ".c")
    output_stream = open("%s.c" % self.implementation.get_name(), "w")
    output_stream.write(self.result.get(self.C_code_generator))
    output_stream.close()

  def gen_implementation(self):
    # generate scheme
    scheme = self.generate_scheme()

    # optimize scheme
    opt_scheme = self.optimise_scheme(scheme)

    # generate C code to implement scheme
    self.generate_C(scheme)


  # Currently mostly empty, to be populated someday
  def gen_emulation_code(self, precode, code, postcode):
    """generate C code that emulates the function, typically using MPFR.
    precode is declaration code (before the test loop)
    postcode is clean-up code (after the test loop)
    Takes the input and output names from input_list and output_list.
    Must postfix output names with "ref_", "ref_ru_", "ref_rd_"

    This class method performs commonly used initializations. 
    It initializes the MPFR versions of the inputs and outputs, 
    with the same names prefixed with "mp" and possibly postfixed with "rd" and "ru".

    It should be overloaded by actual metafunctions, and called by the overloading function. 
    """

  @staticmethod
  def get_name():
    return ML_FunctionBasis.function_name


        
def ML_Function(name):
  new_class = type(name, (ML_FunctionBasis,), {"function_name": name})
  new_class.get_name = staticmethod(lambda: name) 
  return new_class
