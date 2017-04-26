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
               output_file = "cosf.c", 
               function_name = "cosf", 
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
    vx = Abs(self.implementation.add_input_variable("x", self.precision), tag = "vx", precision = self.precision) 

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

    sollya_precision = self.precision.get_sollya_object()
    hi_precision = self.precision.get_field_size() - 3


    # argument reduction
    # m 
    frac_pi_index = {ML_Binary32: 4, ML_Binary64: 6}[self.precision]

    # 2^m / pi 
    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, sollya.RN)
    # pi / 2^m, high part
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, sollya.RN)
    # pi / 2^m, low part
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, sollya.RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = debugd)
    # k in floating-point precision
    fk = Conversion(k, precision = self.precision, tag = "fk")

    inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
    inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

    red_vx_hi = (vx - inv_frac_pi_cst * fk)
    red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
    red_vx_lo_sub = inv_frac_pi_lo_cst * fk
    red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
    vx_d = Conversion(vx, precision = self.precision, tag = "vx_d")
    pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d_hi = (vx_d - inv_frac_pi_cst * fk)
    pre_red_vx_d_hi.set_attributes(tag = "pre_red_vx_d_hi", precision = self.precision, debug = debug_lftolx)
    pre_red_vx_d = pre_red_vx_d_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d.set_attributes(tag = "pre_red_vx_d", debug = debug_lftolx, precision = self.precision)


    modk = Modulo(k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "modk", debug = True)

    sel_c = Equal(BitLogicAnd(modk, 2**(frac_pi_index-1)), 2**(frac_pi_index-1))
    sel_c.set_attributes(tag = "sel_c", debug = debugd)
    red_vx = pre_red_vx # Select(sel_c, -pre_red_vx, pre_red_vx)
    red_vx.set_attributes(tag = "red_vx", debug = debug_precision, precision = self.precision)

    red_vx_d = Select(sel_c, -pre_red_vx_d, pre_red_vx_d)
    red_vx_d.set_attributes(tag = "red_vx_d", debug = debug_lftolx, precision = self.precision)

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

    #cos_hi_prec = self.precision.get_sollya_object() # int(self.precision.get_field_size() * 0.7)
    cos_hi_prec =  int(self.precision.get_field_size() - 2)

    for i in xrange(2**(frac_pi_index+1)):
      local_x = i*pi/S2**frac_pi_index
      cos_local_hi = round(cos(local_x), cos_hi_prec, sollya.RN)
      cos_table_hi[i][0] = cos_local_hi 
      cos_table_lo[i][0] = round(cos(local_x) - cos_local_hi, self.precision.get_sollya_object(), sollya.RN)

      sin_table[i][0] = round(sin(local_x), self.precision.get_sollya_object(), sollya.RN)


    tabulated_cos_hi = TableLoad(cos_table_hi, modk, 0, tag = "tab_cos_hi", debug = debug_precision, precision = self.precision) 
    tabulated_cos_lo = TableLoad(cos_table_lo, modk, 0, tag = "tab_cos_lo", debug = debug_precision, precision = self.precision) 
    tabulated_sin  = TableLoad(sin_table, modk, 0, tag = "tab_sin", debug = debug_precision, precision = self.precision) 

    Log.report(Log.Info, "tabulated cos hi interval: {}".format(cos_table_hi.get_interval()))
    Log.report(Log.Info, "tabulated cos lo interval: {}".format(cos_table_lo.get_interval()))
    Log.report(Log.Info, "tabulated sin    interval: {}".format(sin_table.get_interval()))

    Log.report(Log.Info, "building mathematical polynomials for sin and cos")
    poly_degree_cos   = sup(guessdegree(cos(sollya.x), approx_interval, S2**-(self.precision.get_field_size()+1)) ) + 2
    poly_degree_sin   = sup(guessdegree(sin(sollya.x) / sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 2

    print poly_degree_cos, poly_degree_sin

    poly_degree_cos_list = range(0, poly_degree_cos + 1, 2)
    poly_degree_sin_list = range(0, poly_degree_sin + 1, 2)

    if self.precision is ML_Binary32:
      poly_degree_cos_list = [0, 2, 4, 6]
      poly_degree_sin_list = [0, 2, 4, 6]

    else:
      poly_degree_cos_list = [0, 2, 4, 6]
      poly_degree_sin_list = [0, 2, 4, 6]

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

    # extended precision to be used when more accuracy
    # than the working precision is required
    ext_precision = {
      ML_Binary32: ML_Binary64,
      ML_Binary64: ML_DoubleDouble
    }[self.precision]

    # evaluation scheme
    cos_eval_d = (tabulated_cos_hi - red_vx * (tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))) + tabulated_cos_lo
    cos_eval_d.set_precision(self.precision)

    cos_eval_d_ext = Subtraction(
        Addition(
          Conversion(tabulated_cos_hi, precision = ext_precision),
          Conversion(tabulated_cos_lo, precision = ext_precision)
        ),
        Multiplication(
          Conversion(red_vx, precision = ext_precision),
          Conversion((tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos)))), precision = ext_precision),
          precision = ext_precision
        ),
        precision = ext_precision
      )
    cos_eval_d = Conversion(cos_eval_d_ext, precision = self.precision, tag = "cos_eval_d")


    # cos_eval_d = tabulated_cos_hi + FusedMultiplyAdd(- red_vx, (tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos)))), tabulated_cos_lo)

    cos_eval_d.set_attributes(tag = "cos_eval_d", debug = debug_precision, precision = self.precision)


    promote = {
      ML_Binary32: lambda x: Conversion(x, precision = ext_precision, tag = "promote"),
      ML_Binary64: lambda x: x,
    }[self.precision]


    tab_cos_dd = (promote(tabulated_cos_hi) + promote(tabulated_cos_lo)).modify_attributes(precision = ext_precision, tag = "tab_cos_dd")

    #cos_eval_lo_dd_op0 = tab_cos_dd
    cos_eval_lo_dd_op1 = (promote(-red_vx) * promote(tabulated_sin)).modify_attributes(precision = ext_precision)
    cos_eval_lo_dd_op2 = ((-tabulated_cos_hi) * ((red_vx * red_vx) * 0.5)).modify_attributes(precision = self.precision)
    cos_eval_lo_dd_op3 = ((-tabulated_sin) * (poly_sin * red_vx)).modify_attributes(precision = self.precision)
    cos_eval_lo_dd_op4 = ((tabulated_cos_hi) * (red_vx * poly_cos)).modify_attributes(precision = self.precision)

    # Extended precision evaluation
    cos_eval_d2_add = AdditionN(
      promote(tabulated_cos_hi),
      AdditionN(
        cos_eval_lo_dd_op1, 
        promote(
          AdditionN(
            cos_eval_lo_dd_op2,
            cos_eval_lo_dd_op3,
            cos_eval_lo_dd_op4,
            tabulated_cos_lo,
            unbreakable = True,
            precision = self.precision
          )
        ),
        precision = ext_precision,
        unbreakable = True
      ),
      precision = ext_precision,
      unbreakable = True
    )


    # cos_eval_d2 is a more precise evaluation than cos_eval_d
    # using extended precision for some of the last steps
    cos_eval_d2_add.set_attributes(unbreakable = True, precision = ext_precision, tag = "cos_eval_d2_add", debug = {ML_DoubleDouble: debug_ddtolx, ML_Binary64: debug_lftolx}[ext_precision])
    #cos_eval_d2 = cos_eval_d2_add.hi.modify_attributes(precision = self.precision) + cos_eval_d2_add.lo.modify_attributes(precision = self.precision)
    cos_eval_d2 = cos_eval_d2_add.hi.modify_attributes(precision = self.precision) 
    cos_eval_d2.set_attributes(tag = "cos_eval_d2", debug = debug_precision, precision = self.precision)

    
    exact_sub = (tabulated_cos_hi - red_vx)
    exact_sub.set_attributes(tag = "exact_sub", debug = debug_precision, unbreakable = True, prevent_optimization = True, precision = self.precision)

    # cos_eval_2 and cos_eval_3 are to be used when tabulated_sin is positive and very close to 1.0
    cos_eval_2 = exact_sub + ((- red_vx * ((tabulated_sin - 1) + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))) + tabulated_cos_lo)
    cos_eval_2.set_attributes(tag = "cos_eval_2", precision = self.precision, debug = debug_precision)

    cos_eval_3 = (tabulated_cos_hi + (- red_vx - red_vx * ((tabulated_sin - 1) + tabulated_cos_hi * red_vx * 0.5 + tabulated_sin * poly_sin - tabulated_cos_hi * poly_cos))) + tabulated_cos_lo 
    cos_eval_3.set_attributes(tag = "cos_eval_3", precision = self.precision, debug = debug_precision)

    # computing evaluation error for cos_eval_d
    cos_eval_d_eerror = []
    for i in xrange(0, 2**(frac_pi_index-1)-3):
      copy_map = {
        tabulated_cos_hi : Variable("tabulated_cos_hi", interval = Interval(cos_table_hi[i][0]), precision = self.precision), 
        tabulated_cos_lo : Variable("tabulated_cos_lo", interval = Interval(cos_table_lo[i][0]), precision = self.precision), 
        tabulated_sin    : Variable("tabulated_sin", interval = Interval(sin_table[i][0]), precision = self.precision),
        red_vx           : Variable("red_vx", precision = self.precision, interval = approx_interval),
      }

      cos_eval_d_eerror_local = sup(abs(self.get_eval_error(cos_eval_d, variable_copy_map = copy_map, relative_error = True)))
      cos_eval_d_eerror.append(cos_eval_d_eerror_local)
      if cos_eval_d_eerror_local > S2**-52:
        print "cos_eval_d_eerror_local: ", i, cos_eval_d_eerror_local

    for i in xrange(2**(frac_pi_index-1)-3, 2**(frac_pi_index-1)-1):
      copy_map = {
        tabulated_cos_hi : Variable("tabulated_cos_hi", interval = Interval(cos_table_hi[i][0]), precision = self.precision), 
        tabulated_cos_lo : Variable("tabulated_cos_lo", interval = Interval(cos_table_lo[i][0]), precision = self.precision), 
        tabulated_sin    : Variable("tabulated_sin", interval = Interval(sin_table[i][0]), precision = self.precision),
        red_vx           : Variable("red_vx", precision = self.precision, interval = approx_interval),
      }

      cos_eval_d_eerror_local = sup(abs(self.get_eval_error(cos_eval_2, variable_copy_map = copy_map, relative_error = True)))
      cos_eval_d_eerror.append(cos_eval_d_eerror_local)
      if cos_eval_d_eerror_local > S2**-52:
        print "cos_eval_d_eerror_local_2: ", i, cos_eval_d_eerror_local

      cos_eval_d_eerror_local = sup(abs(self.get_eval_error(cos_eval_3, variable_copy_map = copy_map, relative_error = True)))
      cos_eval_d_eerror.append(cos_eval_d_eerror_local)
      if cos_eval_d_eerror_local > S2**-52:
        print "cos_eval_d_eerror_local_3: ", i, cos_eval_d_eerror_local

    print "max error: ", max(cos_eval_d_eerror)

    # sys.exit(1)

    # selecting int precision for cast corresponding to precision width
    cast_int_precision = self.precision.get_integer_format()

    cond_3 = LogicalAnd(
                Comparison(Abs(red_vx), Constant(cos_table_hi[2**(frac_pi_index-1)-1][0] / S2, precision = self.precision), specifier = Comparison.GreaterOrEqual, tag = "comp_bound", debug = True),
                # opposite sign
                Equal(
                  BitLogicRightShift(
                    BitLogicXor(
                      TypeCast(red_vx, precision = cast_int_precision),
                      TypeCast(tabulated_cos_hi, precision = cast_int_precision)
                    ),
                    Constant(cast_int_precision.get_bit_size() - 1)
                  ),
                  Constant(1, precision = cast_int_precision)
                ),
              tag = "cond3",
              debug = True
            )

    result_sel_c = ( Equal(modk, Constant(2**(frac_pi_index-1)-1), precision = ML_Int32) |
                     Equal(modk, Constant(2**(frac_pi_index-1)), precision = ML_Int32)   |
                     Equal(modk, Constant(2**(frac_pi_index-1)+1), precision = ML_Int32) 
                     #Equal(modk, Constant(3 * 2**(frac_pi_index-1)-1), precision = ML_Int32) |
                     #Equal(modk, Constant(3 * 2**(frac_pi_index-1)), precision = ML_Int32) |
                     #Equal(modk, Constant(3 * 2**(frac_pi_index-1)+1), precision = ML_Int32) 
                  ).modify_attributes(
                    tag = "result_sel_c",
                    debug = debugd
                  )

    result = Statement(
        ConditionBlock(
          result_sel_c,
          ConditionBlock(
            cond_3,
            Return(cos_eval_2),
            Return(cos_eval_3)
          ), 
          ConditionBlock(
            (
              Equal(modk, Constant(3* 2**(frac_pi_index-1)+1), precision = ML_Int32) | 
              Equal(modk, Constant(3* 2**(frac_pi_index-1)), precision = ML_Int32) | 
              Equal(modk, Constant(3* 2**(frac_pi_index-1)-1), precision = ML_Int32) 
            ).modify_attributes(tag = "result_sel_d2", debug = debugd),
            Return(cos_eval_d2),
            Return(cos_eval_d)
          )
        )
      )

    #######################################################################
    #                    LARGE ARGUMENT MANAGEMENT                        #
    #                 (lar: Large Argument Reduction)                     #
    #######################################################################
    # payne and hanek argument reduction for large arguments
    ph_k = 6
    ph_frac_pi     = S2**ph_k / pi
    ph_inv_frac_pi = pi / S2**ph_k 
    # ph_chunk_num = 
    ph_statement, ph_acc, ph_acc_int = generate_payne_hanek(vx, ph_frac_pi, self.precision, n = 100, k = ph_k)

    # assigning Large Argument Reduction reduced variable
    lar_vx = Variable("lar_vx", precision = self.precision, var_type = Variable.Local)
    lar_tab_index = Variable("lar_tab_index", precision = ML_Int32, var_type = Variable.Local)

    lar_red_vx = lar_vx * Constant(ph_inv_frac_pi, precision = self.precision)
    lar_red_vx.set_attributes(tag = "lar_red_vx", debug = debug_precision)

    red_bound     = S2**20
    lar_cond = Abs(vx) >= red_bound
    lar_cond.set_attributes(tag = "lar_cond", likely = False)

    
    lar_tabulated_cos_hi = TableLoad(cos_table_hi, lar_tab_index, 0, tag = "lar_tab_cos_hi", debug = debug_precision) 
    lar_tabulated_cos_lo = TableLoad(cos_table_lo, lar_tab_index, 0, tag = "lar_tab_cos_lo", debug = debug_precision) 
    lar_tabulated_sin    = TableLoad(sin_table,    lar_tab_index, 0, tag = "lar_tab_sin", debug = debug_precision) 

    lar_approx_interval = Interval(-1, 1)

    lar_poly_cos = polynomial_scheme_builder(poly_object_cos.sub_poly(start_index = 4, offset = 1), lar_red_vx, unified_precision = self.precision)
    lar_poly_sin = polynomial_scheme_builder(poly_object_sin.sub_poly(start_index = 2), lar_red_vx, unified_precision = self.precision)
    lar_poly_cos.set_attributes(tag = "lar_poly_cos", debug = debug_precision)
    lar_poly_sin.set_attributes(tag = "lar_poly_sin", debug = debug_precision)


    lar_cos_eval_d = lar_tabulated_cos_hi + (- lar_red_vx * (lar_tabulated_sin + (lar_tabulated_cos_hi * lar_red_vx * 0.5 + (lar_tabulated_sin * lar_poly_sin + (- lar_tabulated_cos_hi * lar_poly_cos)))) + lar_tabulated_cos_lo)

    lar_cos_eval_d.set_attributes(tag = "lar_cos_eval_d", debug = debug_precision, precision = self.precision)
    
    lar_exact_sub = (lar_tabulated_cos_hi - lar_red_vx)
    lar_exact_sub.set_attributes(tag = "lar_exact_sub", debug = debug_precision, unbreakable = True, prevent_optimization = True)

    lar_cos_eval_2 = lar_exact_sub + ((- lar_red_vx * ((lar_tabulated_sin - 1) + (lar_tabulated_cos_hi * lar_red_vx * 0.5 + (lar_tabulated_sin * lar_poly_sin + (- lar_tabulated_cos_hi * lar_poly_cos))))) + lar_tabulated_cos_lo)
    lar_cos_eval_2.set_attributes(tag = "lar_cos_eval_2", precision = self.precision, debug = debug_precision)

    #lar_cos_eval_4 = - lar_red_vx - lar_red_vx * (lar_poly_sin)
    lar_cos_eval_4 = lar_tabulated_sin * FusedMultiplyAdd(-lar_vx, Constant(ph_inv_frac_pi, precision = self.precision), -lar_red_vx * lar_poly_sin)
    lar_cos_eval_4.set_attributes(tag = "lar_cos_eval_4", debug = debug_precision)

    # to be used when lar_tabulated_sin is very close to 1 
    lar_cos_eval_3 = (lar_tabulated_cos_hi + (- lar_red_vx - lar_red_vx * ((lar_tabulated_sin - 1) + lar_tabulated_cos_hi * lar_red_vx * 0.5 + lar_tabulated_sin * lar_poly_sin - lar_tabulated_cos_hi * lar_poly_cos))) + lar_tabulated_cos_lo 
    lar_cos_eval_3.set_attributes(tag = "lar_cos_eval_3", precision = self.precision, debug = debug_precision)

    # selecting int precision for cast corresponding to precision width
    cast_int_precision = self.precision.get_integer_format()

    lar_cond_3 = LogicalAnd(
                Comparison(Abs(lar_red_vx), Constant(cos_table_hi[2**(frac_pi_index-1)-1][0] / S2, precision = self.precision), specifier = Comparison.GreaterOrEqual, tag = "lar_comp_bound", debug = True),
                Equal(
                  BitLogicRightShift(
                    BitLogicXor(
                      TypeCast(lar_red_vx, precision = cast_int_precision),
                      TypeCast(lar_tabulated_cos_hi, precision = cast_int_precision)
                    ),
                    Constant(cast_int_precision.get_bit_size() - 1)
                  ),
                  Constant(1, precision = cast_int_precision)
                ),
              tag = "lar_cond3",
              debug = True
            )

    lar_result_sel_c = (Equal(lar_tab_index, Constant(2**(frac_pi_index-1)-1), precision = ML_Int32) |
                        Equal(lar_tab_index, Constant(2**(frac_pi_index-1)+1), precision = ML_Int32) # |
                        # Equal(lar_tab_index, Constant(3 * 2**(frac_pi_index-1)+1), precision = ML_Int32) |
                        # Equal(lar_tab_index, Constant(3 * 2**(frac_pi_index-1)-1), precision = ML_Int32)
                        ).modify_attributes(
                            tag = "lar_result_sel_c",
                            debug = debugd
                        )

    lar_result_sel_mid_c = (Equal(lar_tab_index, Constant(2**(frac_pi_index-1)), precision = ML_Int32) | 
                            Equal(lar_tab_index, Constant(3 * 2**(frac_pi_index-1)), precision = ML_Int32) 
                          ).modify_attributes(tag = "lar_result_sel_mid_c", debug = debugd)
    #LogicalOr(
    #                LogicalOr(
    #                  Equal(lar_tab_index, Constant(2**(frac_pi_index-1)-1), precision = ML_Int32),
    #                  Equal(lar_tab_index, Constant(2**(frac_pi_index-1)), precision = ML_Int32)
    #                ),
    #                Equal(lar_tab_index, Constant(2**(frac_pi_index-1)+1), precision = ML_Int32),
    #                tag = "result_sel_c",
    #                debug = debugd
    #              )

    int_precision = {
      ML_Binary64: ML_Int64,
      ML_Binary32: ML_Int32
    }[self.precision]
    C32 = Constant(2**(ph_k+1), precision = int_precision)
    ph_acc_int_red = Conversion(Select(ph_acc_int < Constant(0, precision = int_precision), ph_acc_int + C32, ph_acc_int), precision = self.precision)

    lar_result = Statement(
        ph_statement,
        ReferenceAssign(lar_vx, ph_acc, debug = debug_precision),
        ReferenceAssign(lar_tab_index, ph_acc_int_red, debug = debugd),
        lar_cos_eval_4,
        lar_cos_eval_3,
        lar_cos_eval_2,
        lar_cos_eval_d,
        ConditionBlock(
          lar_result_sel_c,
          ConditionBlock(
            lar_cond_3,
            Return(lar_cos_eval_2),
            Return(lar_cos_eval_3)
          ), 
          ConditionBlock(
            lar_result_sel_mid_c,
            Return(lar_cos_eval_4),
            Return(lar_cos_eval_d)
          )
        ),
        prevent_optimization = True
      )

    scheme = Statement(
      ConditionBlock(
        lar_cond,
        lar_result,
        result
      )
    )


    return scheme

  def numeric_emulate(self, input_value):
    if self.sin_output:
      return sin(input_value)
    else:
      return cos(input_value)

  standard_test_cases =[sollya.parse(x) for x in  ["0x1.d5c0bcp-4", "-0x1.3e25bp+2"]]



if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_function_name = "new_sincos", default_output_file = "new_sincos.c" )
  # argument extraction 
  arg_template.get_parser().add_argument("--sin", dest = "sin_output", default = False, const = True, action = "store_const", help = "select sine output (default is cosine)")

  args = arg_template.arg_extraction()

  ml_sincos = ML_SinCos(args, sin_output = args.sin_output)
  ml_sincos.gen_implementation()
