from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.ml_operations import Variable, ReferenceAssign, Statement, Return, ML_ArithmeticOperation, ConditionBlock, LogicalAnd 
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.code_generation.code_object import NestedCode
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import FunctionOperator

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

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

    self.debug_flag = debug_flag

    # TODO: FIX which i/o precision to select
    # TODO: incompatible with fixed-point formats
    # self.sollya_precision = self.get_output_precision().get_sollya_object()

    self.abs_accuracy = abs_accuracy if abs_accuracy else S2**(-self.get_output_precision().get_precision())

    self.libm_compliant = libm_compliant
    
    self.processor = processor

    self.fuse_fma = fuse_fma
    self.fast_path_extract = fast_path_extract

    self.implementation = CodeFunction(self.function_name, output_format = self.get_output_precision())
    self.opt_engine = OptimizationEngine(self.processor)
    self.gappa_engine = GappaCodeGenerator(self.processor, declare_cst = True, disable_debug = True)

    self.C_code_generator = CCodeGenerator(self.processor, declare_cst = False, disable_debug = not self.debug_flag, libm_compliant = self.libm_compliant)
    self.main_code_object = NestedCode(self.C_code_generator, static_cst = True)

  def get_eval_error(self, optree, variable_copy_map = {}, goal_precision = ML_Exact, gappa_filename = "gappa_eval_error.g", relative_error = False):
    """ wrapper for GappaCodeGenerator get_eval_error_v2 function """
    copy_map = {}
    for leaf in variable_copy_map: 
      copy_map[leaf] = variable_copy_map[leaf]
    opt_optree = self.optimise_scheme(optree, copy = copy_map, verbose = False)
    new_variable_copy_map = {}
    for leaf in variable_copy_map:
      new_variable_copy_map[leaf.get_handle().get_node()] = variable_copy_map[leaf]
    return self.gappa_engine.get_eval_error_v2(self.opt_engine, opt_optree, new_variable_copy_map if variable_copy_map != None else {}, goal_precision, gappa_filename, relative_error = relative_error)

  def uniquify_name(self, base_name):
    """ return a unique identifier, combining base_name + function_name """
    return "%s_%s" % (self.function_name, base_name)

  
  def generate_emulate(self):
    raise ML_NotImplemented()


  def generate_emulate_wrapper(self, test_input   = Variable("vx", precision = ML_Mpfr_t), mpfr_rnd = Variable("rnd", precision = ML_Int32), test_output = Variable("result", precision = ML_Mpfr_t, var_type = Variable.Local), test_ternary = Variable("ternary", precision = ML_Int32, var_type = Variable.Local)):
    scheme = self.generate_emulate(test_ternary, test_output, test_input, mpfr_rnd)

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

  ## 
  # @return main_scheme, [list of sub-CodeFunction object]
  def generate_function_list(self):
    self.implementation.set_scheme(self.generate_scheme())
    return [self.implementation]

  def optimise_scheme(self, pre_scheme, copy = None, enable_subexpr_sharing = True, verbose = True):
    """ default scheme optimization """
    # copying when required
    scheme = pre_scheme if copy is None else pre_scheme.copy(copy)
    # fusing FMA
    if self.fuse_fma:
      if verbose: print "MDL fusing FMA"
      scheme = self.opt_engine.fuse_multiply_add(scheme, silence = True)

    if verbose: print "MDL abstract scheme"
    self.opt_engine.instantiate_abstract_precision(scheme, None)

    if verbose: print "MDL instantiated scheme"
    self.opt_engine.instantiate_precision(scheme, default_precision = None)

    if enable_subexpr_sharing:
      if verbose: print "subexpression sharing"
      self.opt_engine.subexpression_sharing(scheme)

    if verbose: print "silencing operation"
    self.opt_engine.silence_fp_operations(scheme)

    if verbose: print "checking processor support"
    self.opt_engine.check_processor_support(scheme)

    return scheme

  def get_main_code_object(self):
    return self.main_code_object

  def generate_C(self, code_function_list):
    """ Final C generation, once the evaluation scheme has been optimized"""
    # registering scheme as function implementation
    #self.implementation.set_scheme(scheme)
    # main code object
    code_object = self.get_main_code_object()
    self.result = code_object
    for code_function in code_function_list:
      self.result = code_function.add_definition(self.C_code_generator, C_Code, code_object, static_cst = True)

    # adding headers
    self.result.add_header("support_lib/ml_special_values.h")
    self.result.add_header("math.h")
    self.result.add_header("stdio.h")
    self.result.add_header("inttypes.h")

    Log.report(Log.Info, "Generating C code in " + self.implementation.get_name() + ".c")
    output_stream = open("%s.c" % self.implementation.get_name(), "w")
    output_stream.write(self.result.get(self.C_code_generator))
    output_stream.close()

  def gen_implementation(self, display_after_gen = False, display_after_opt = False, enable_subexpr_sharing = True):
    # generate scheme
    code_function_list = self.generate_function_list()

    for code_function in code_function_list:
      scheme = code_function.get_scheme()
      if display_after_gen:
        print "function %s, after gen " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

      # optimize scheme
      opt_scheme = self.optimise_scheme(scheme, enable_subexpr_sharing = enable_subexpr_sharing)

      if display_after_opt:
        print "function %s, after opt " % code_function.get_name()
        print scheme.get_str(depth = None, display_precision = True, memoization_map = {})

    # generate C code to implement scheme
    self.generate_C(code_function_list)

  ## 
  # @param optree ML_Operation object to be externalized
  # @param arg_list list of ML_Operation objects to be used as arguments
  # @return pair ML_Operation, ML_Funct
  def externalize_call(self, optree, arg_list, tag = "foo", result_format = None):
    # determining return format
    return_format = optree.get_precision() if result_format is None else result_format
    assert(not result_format is None and "external call result format must be defined")
    function_name = self.main_code_object.declare_free_function_name(tag)

    ext_function = CodeFunction(function_name, output_format = return_format)

    # creating argument copy
    arg_map = {}
    arg_index = 0
    for arg in arg_list:
      arg_tag = arg.get_tag(default = "arg_%d" % arg_index)
      arg_index += 1
      arg_map[arg] = ext_function.add_input_variable(arg_tag, arg.get_precision())

    # copying optree while swapping argument for variables
    optree_copy = optree.copy(copy_map = arg_map)
    # instanciating external function scheme
    if isinstance(optree, ML_ArithmeticOperation):
      function_optree = Statement(Return(optree_copy))
    else:
      function_optree = Statement(optree_copy)
    ext_function.set_scheme(function_optree)
    ext_function_object = ext_function.get_function_object()
    self.main_code_object.declare_function(function_name, ext_function_object)
    return ext_function_object(*arg_list), ext_function

  def vectorize_scheme(self, optree, arg_list):
    def fallback_policy(cond, cond_block, if_branch, else_branch):
      return if_branch, [cond]
    def and_merge_conditions(condition_list):
      assert(len(condition_list) >= 1)
      if len(condition_list) == 1:
        return condition_list[0]
      else:
        half_size = len(condition_list) / 2
        first_half  = and_merge_conditions(condition_list[:half_size])
        second_half = and_merge_conditions(condition_list[half_size:])
        return LogicalAnd(first_half, second_half)

    linearized_most_likely_path, validity_list = self.opt_engine.extract_vectorizable_path(optree, fallback_policy)

    assert(isinstance(linearized_most_likely_path, ML_ArithmeticOperation))
    likely_result = linearized_most_likely_path

    callback, callback_function = self.externalize_call(optree, arg_list, tag = "scalar_callback", result_format = self.get_output_precision())

    vectorized_scheme = Statement(
      likely_result,
      ConditionBlock(and_merge_conditions(validity_list),
        Return(likely_result),
        Return(callback)
      )
    )
    return vectorized_scheme, callback_function


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
