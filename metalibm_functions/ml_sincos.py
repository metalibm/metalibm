# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import (
        S2, Interval, ceil, floor, round, inf, sup, pi, log, exp, cos, sin,
        guessdegree, dirtyinfnorm
)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import ML_NewArgTemplate, ArgDefault
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

# disabling sollya's rounding warning
sollya.roundingwarnings = sollya.off 
sollya.verbosity = 0
sollya.showmessagenumbers = sollya.on

## Implementation of sine or cosine sharing a common
#  approximation scheme
class ML_SinCos(ML_Function("ml_cos")):
  """ Implementation of cosinus function """
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
               precision = ML_Binary32, 
               accuracy  = ML_Faithful,
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               fast_path_extract = True,
               target = GenericProcessor(), 
               output_file = "ml_cos.c", 
               function_name = "ml_cos", 
               sin_output = True):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "cos",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      arg_template = arg_template
    )
    self.precision = precision
    self.sin_output = sin_output



  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_SinCos functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cos" if self.sin_output else "mpfr_sin"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self): 
    # declaring CodeFunction and retrieving input variable
    vx = self.implementation.add_input_variable("x", self.precision)

    Log.report(Log.Info, "generating implementation scheme")
    if self.debug_flag: 
        Log.report(Log.Info, "debug has been enabled")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    debug_precision = {ML_Binary32: debug_ftox, ML_Binary64: debug_lftolx}[self.precision]


    test_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
    test_nan        = Test(vx, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
    test_positive   = Comparison(vx, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")

    test_signaling_nan = Test(vx, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
    return_snan        = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    # return in case of infinity input
    infty_return = Statement(ConditionBlock(test_positive, Return(FP_PlusInfty(self.precision)), Return(FP_PlusZero(self.precision))))
    # return in case of specific value input (NaN or inf)
    specific_return = ConditionBlock(test_nan, ConditionBlock(test_signaling_nan, return_snan, Return(FP_QNaN(self.precision))), infty_return)
    # return in case of standard (non-special) input

    # bound to determine extended precision and standard precision
    # computation
    red_bound     = S2**10

    sollya_precision = self.precision.get_sollya_object()
    hi_precision = self.precision.get_field_size() - 13

    # extended precision to be used when more accuracy
    # than the working precision is required
    ext_precision = {
      ML_Binary32: ML_Binary64,
      ML_Binary64: ML_Binary64
    }[self.precision]
    promote = {
      ML_Binary32: lambda x: Conversion(x, precision = ext_precision, tag = "promote"),
      ML_Binary64: lambda x: x,
    }[self.precision]

    # argument reduction
    # m 
    frac_pi_index = {ML_Binary32: 4, ML_Binary64: 6}[self.precision]

    # 2^m / pi 
    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, sollya.RN)
    # pi / 2^m, high part
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, sollya.RN)

    inv_frac_pi_ext = round(pi / S2**frac_pi_index, ext_precision.get_sollya_object(), sollya.RN)
    # pi / 2^m, low part
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, sollya.RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(
      vx, 
      Constant(frac_pi, precision = self.precision), 
      precision = self.precision,
      tag = "vx_pi"
    )
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = debugd)
    # k in floating-point precision
    fk = Conversion(k, precision = self.precision, tag = "fk")

    inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
    inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

    # red_vx_hi = (vx - inv_frac_pi_cst * fk)
    red_vx_hi = FusedMultiplyAdd(inv_frac_pi_cst, fk, vx, specifier = FusedMultiplyAdd.SubtractNegate, precision = self.precision)
    red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
    # red_vx_lo_sub = inv_frac_pi_lo_cst * fk
    # red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
    vx_d = Conversion(vx, precision = self.precision, tag = "vx_d")
    #pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk
    red_vx_std = FusedMultiplyAdd(inv_frac_pi_lo_cst, fk, red_vx_hi, specifier = FusedMultiplyAdd.SubtractNegate, precision = self.precision, tag = "red_vx_std")
    # pre_red_vx_d_hi = (vx_d - inv_frac_pi_cst * fk)
    # pre_red_vx_d_hi.set_attributes(tag = "pre_red_vx_d_hi", precision = self.precision, debug = debug_lftolx)
    # pre_red_vx_d = pre_red_vx_d_hi - inv_frac_pi_lo_cst * fk
    # pre_red_vx_d.set_attributes(tag = "pre_red_vx_d", debug = debug_lftolx, precision = self.precision)

    # to compute sine we offset x by 3pi/2 
    # which means add 3  * S2^(frac_pi_index-1) to k
    if self.sin_output:
      offset_k = Addition(
        k,
        Constant(3 * S2**(frac_pi_index - 1), precision = ML_Int32),
        precision = ML_Int32,
        tag = "offset_k"
      )
    else:
      offset_k = k

    # multi-path shared variable
    modk       = Variable("modk", precision = ML_Int32, var_type = Variable.Local)
    red_vx     = Variable("red_vx", precision = self.precision, var_type = Variable.Local)
    red_vx_ext = Variable("red_vx_ext", precision = ext_precision, var_type = Variable.Local)

    
    #modk = Modulo(offset_k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "modk", debug = True)
    # faster modulo using bitwise logic
    modk_std = BitLogicAnd(offset_k, 2**(frac_pi_index+1)-1, precision = ML_Int32, tag = "modk", debug = True)


    approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

    red_vx.set_interval(approx_interval)

    Log.report(Log.Info, "approx interval: %s\n" % approx_interval)

    error_goal_approx = S2**-self.precision.get_precision()


    Log.report(Log.Info, "building tabulated approximation for sin and cos")
    poly_degree_vector = [None] * 2**(frac_pi_index+1)


    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

    table_index_size = frac_pi_index+1
    cos_table_hi = ML_NewTable(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_hi"))
    cos_table_lo = ML_NewTable(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_lo"))
    sin_table = ML_NewTable(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("sin_table"))
    if not isinstance(ext_precision, ML_Compound_Format):
      sin_table_ext = ML_NewTable(dimensions = [2**table_index_size], storage_precision = ext_precision, tag = self.uniquify_name("sin_table_ext"))
      cos_table_ext = ML_NewTable(dimensions = [2**table_index_size], storage_precision = ext_precision, tag = self.uniquify_name("cos_table_ext"))

    #cos_hi_prec = self.precision.get_sollya_object() # int(self.precision.get_field_size() * 0.7)
    cos_hi_prec =  int(self.precision.get_field_size() - 2)

    for i in xrange(2**(frac_pi_index+1)):
      local_x = i*pi/S2**frac_pi_index
      cos_local_hi = round(cos(local_x), cos_hi_prec, sollya.RN)
      cos_table_hi[i][0] = cos_local_hi 
      cos_table_lo[i][0] = round(cos(local_x) - cos_local_hi, self.precision.get_sollya_object(), sollya.RN)

      sin_table[i][0] = round(sin(local_x), self.precision.get_sollya_object(), sollya.RN)

      # extended table
      if not isinstance(ext_precision, ML_Compound_Format):
        cos_table_ext[i] = round(cos(local_x), ext_precision.get_sollya_object(), sollya.RN)
        sin_table_ext[i] = round(sin(local_x), ext_precision.get_sollya_object(), sollya.RN)


    tabulated_cos_hi = TableLoad(cos_table_hi, modk, 0, tag = "tab_cos_hi", debug = debug_precision, precision = self.precision) 
    tabulated_cos_lo = TableLoad(cos_table_lo, modk, 0, tag = "tab_cos_lo", debug = debug_precision, precision = self.precision) 
    tabulated_sin  = TableLoad(sin_table, modk, 0, tag = "tab_sin", debug = debug_precision, precision = self.precision) 


    Log.report(Log.Info, "tabulated cos hi interval: {}".format(cos_table_hi.get_interval()))
    Log.report(Log.Info, "tabulated cos lo interval: {}".format(cos_table_lo.get_interval()))
    Log.report(Log.Info, "tabulated sin    interval: {}".format(sin_table.get_interval()))

    Log.report(Log.Info, "building mathematical polynomials for sin and cos")
    poly_degree_cos   = sup(guessdegree(cos(sollya.x), approx_interval, S2**-(self.precision.get_field_size()+1)) ) + 2
    poly_degree_sin   = sup(guessdegree(sin(sollya.x) / sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 2

    poly_degree_cos_list = range(0, poly_degree_cos + 1, 2)
    poly_degree_sin_list = range(0, poly_degree_sin + 1, 2)

    if self.precision is ML_Binary32:
      poly_degree_cos_list = [0, 2, 4, 6]
      poly_degree_sin_list = [0, 2, 4, 6]

    else:
      poly_degree_cos_list = [0, 2, 4, 5, 6, 7, 8]
      poly_degree_sin_list = [0, 2, 4, 5, 6, 7, 8]

    # cosine polynomial: limiting first and second coefficient precision to 1-bit
    poly_cos_prec_list = [1, 1] + [self.precision] * (len(poly_degree_cos_list) - 2)
    # sine polynomial: limiting first coefficient precision to 1-bit
    poly_sin_prec_list = [1] + [self.precision] * (len(poly_degree_sin_list) - 1)

    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    # Polynomial approximations
    poly_object_cos, poly_error_cos = Polynomial.build_from_approximation_with_error(cos(sollya.x), poly_degree_cos_list, poly_cos_prec_list, approx_interval, sollya.absolute, error_function = error_function)
    poly_object_sin, poly_error_sin = Polynomial.build_from_approximation_with_error(sin(sollya.x)/sollya.x, poly_degree_sin_list, poly_sin_prec_list, approx_interval, sollya.absolute, error_function = error_function)

    Log.report(Log.Info, "poly error cos: {} / {:d}".format(poly_error_cos, int(sollya.log2(poly_error_cos))))
    Log.report(Log.Info, "poly error sin: {0} / {1:d}".format(poly_error_sin, int(sollya.log2(poly_error_sin))))
  
    # Polynomial evaluation scheme
    poly_cos = polynomial_scheme_builder(poly_object_cos.sub_poly(start_index = 4, offset = 1), red_vx, unified_precision = self.precision)
    poly_sin = polynomial_scheme_builder(poly_object_sin.sub_poly(start_index = 2), red_vx, unified_precision = self.precision)
    poly_cos.set_attributes(tag = "poly_cos", debug = debug_precision)
    poly_sin.set_attributes(tag = "poly_sin", debug = debug_precision)


    # evaluation scheme
    # cos_eval_d = (tabulated_cos_hi - red_vx * (tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))) + tabulated_cos_lo
    # cos_eval_d.set_precision(self.precision)
    if not isinstance(ext_precision, ML_Compound_Format):
      tabulated_cos_ext = TableLoad(cos_table_ext, modk, tag = "tab_cos_ext", debug = debug_precision, precision = ext_precision) 
      tabulated_sin_ext  = TableLoad(sin_table_ext, modk, tag = "tab_sin_ext", debug = debug_precision, precision = ext_precision) 
      red_vx_ext_std = FusedMultiplyAdd(
        Constant(inv_frac_pi_ext , precision = ext_precision),
        promote(fk), 
        promote(vx), 
        specifier = FusedMultiplyAdd.Subtract, 
        precision = ext_precision,
        tag = "red_vx_ext",
        debug = debug_multi
      ) 
      
      #Addition(
      #  promote(- red_vx_hi), 
      #  promote( red_vx_lo_sub),
      #  precision = ext_precision,
      #  tag = "red_vx_ext",
      #  debug = debug_multi
      #)
    else:
      tabulated_cos_ext = Addition(
        promote(tabulated_cos_hi),
        promote(tabulated_cos_lo),
        precision = ext_precision
      )
      tabulated_sin_ext = promote(tabulated_sin)
      red_vx_ext_std = promote(
        Negation(
          red_vx,
          precision = self.precision
        )
      )

    cos_eval_d_ext = Subtraction(
      Addition(
        tabulated_cos_ext,
        Multiplication(
          red_vx_ext,
          tabulated_sin_ext,
          precision = ext_precision
        ),
        precision = ext_precision
      ),
      Multiplication(
        promote(red_vx),
        promote((((tabulated_cos_hi * red_vx * 0.5) + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))),
        precision = ext_precision
      ),
      precision = ext_precision
    )

    cos_eval_d = Conversion(cos_eval_d_ext, precision = self.precision, tag = "cos_eval_d")


    cos_eval_d.set_attributes(tag = "cos_eval_d", debug = debug_precision, precision = self.precision)





    # computing evaluation error for cos_eval_d
    cos_eval_d_eerror = []
    for i in xrange(0, 2**(frac_pi_index-1)-3):
      copy_map = {
        tabulated_cos_hi : Variable("tabulated_cos_hi", interval = Interval(cos_table_hi[i][0]), precision = self.precision), 
        tabulated_cos_lo : Variable("tabulated_cos_lo", interval = Interval(cos_table_lo[i][0]), precision = self.precision), 
        tabulated_sin    : Variable("tabulated_sin", interval = Interval(sin_table[i][0]), precision = self.precision),
        red_vx           : Variable("red_vx", precision = self.precision, interval = approx_interval),
      }

      # disabled because Abs is not supported in Gappa backend
      #cos_eval_d_eerror_local = sup(abs(self.get_eval_error(cos_eval_d, variable_copy_map = copy_map, relative_error = True)))
      #cos_eval_d_eerror.append(cos_eval_d_eerror_local)
      #if cos_eval_d_eerror_local > S2**-52:
      #  print "cos_eval_d_eerror_local: ", i, cos_eval_d_eerror_local

    # Log.report(Log.Info, "max cos_eval_d error: {}".format(max(cos_eval_d_eerror)))


    result = Statement(
       Return(cos_eval_d)
    )

    #######################################################################
    #                    LARGE ARGUMENT MANAGEMENT                        #
    #                 (lar: Large Argument Reduction)                     #
    #######################################################################
    # payne and hanek argument reduction for large arguments
    ph_k = 4
    ph_frac_pi     = S2**ph_k / pi
    ph_inv_frac_pi = pi / S2**ph_k 
    # ph_chunk_num = 
    ph_statement, ph_acc, ph_acc_int = generate_payne_hanek(vx, ph_frac_pi, self.precision, n = 100, k = ph_k)

    # assigning Large Argument Reduction reduced variable
    lar_vx = Variable("lar_vx", precision = self.precision, var_type = Variable.Local)
    lar_tab_index = Variable("lar_tab_index", precision = ML_Int32, var_type = Variable.Local, debug = debug_multi)

    lar_red_vx = Multiplication(
      lar_vx,
      Constant(ph_inv_frac_pi, precision = self.precision),
      precision = self.precision,
      tag = "lar_red_vx",
      debug = debug_multi
    )


    if not isinstance(ext_precision, ML_Compound_Format):
      lar_red_vx_ext = Multiplication(
        promote(lar_vx),
        Constant(ph_inv_frac_pi, precision = ext_precision),
        precision = ext_precision,
        tag = "lar_red_vx_ext",
        debug = debug_multi
      )
    else:
      lar_red_vx_ext = Multiplication(
        promote(lar_vx),
        Constant(ph_inv_frac_pi, precision = self.precision),
        precision = ext_precision,
        tag = "lar_red_vx_ext",
        debug = debug_multi
      )

    lar_cond = vx >= red_bound
    lar_cond.set_attributes(tag = "lar_cond", likely = False, debug = debug_multi)

    
    

    int_precision = {
      ML_Binary64: ML_Int64,
      ML_Binary32: ML_Int32
    }[self.precision]
    C32 = Constant(2**(ph_k+1), precision = int_precision, tag = "C32")
    ph_acc_int_red = Conversion(
      Select(ph_acc_int < Constant(0, precision = int_precision), ph_acc_int + C32, ph_acc_int, precision = int_precision, tag = "ph_acc_int_red"),
      precision = ML_Int32,
      tag = "ph_acc_int_red",
      debug = debug_multi
    )

    if self.sin_output:
      lar_offset_k = Addition(
        ph_acc_int_red,
        Constant(3 * S2**(frac_pi_index - 1), precision = ML_Int32),
        precision = ML_Int32,
        tag = "lar_offset_k"
      )
    else:
      lar_offset_k = ph_acc_int_red


    lar_statement = Statement(
        ph_statement,
        ReferenceAssign(lar_vx, ph_acc, debug = debug_precision),
        ReferenceAssign(red_vx, lar_red_vx, debug = debug_precision),
        ReferenceAssign(red_vx_ext, -lar_red_vx_ext, debug = debug_precision),
        ReferenceAssign(
          modk, 
          BitLogicAnd(
            lar_offset_k, 
            2**(frac_pi_index+1) - 1,
            precision = ML_Int32
          ),
          debug = debug_multi),
        prevent_optimization = True
      )

    scheme = Statement(
      modk,
      red_vx,
      red_vx_ext,
      ConditionBlock(
        lar_cond,
        lar_statement,
        Statement(
          ReferenceAssign(modk, modk_std),
          ReferenceAssign(red_vx, red_vx_std),
          ReferenceAssign(red_vx_ext, red_vx_ext_std),
        )
      ),
      result
    )


    return scheme

  def numeric_emulate(self, input_value):
    if self.sin_output:
      return sin(input_value)
    else:
      return cos(input_value)

  standard_test_cases =[[sollya.parse(x)] for x in  [ "0x1.e57612p+19", "0x1.0ef65ap+9", "0x1.c20874p+9", "-0x1.419768p+18", "-0x1.fd0846p+2", "0x1.d5c0bcp-4", "-0x1.3e25bp+2"]]



if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_function_name = "new_sincos", default_output_file = "new_sincos.c" )
  # argument extraction 
  arg_template.get_parser().add_argument("--sin", dest = "sin_output", default = False, const = True, action = "store_const", help = "select sine output (default is cosine)")

  args = arg_template.arg_extraction()

  ml_sincos = ML_SinCos(args, sin_output = args.sin_output)
  ml_sincos.gen_implementation()
