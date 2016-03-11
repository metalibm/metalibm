# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed




    


class ML_SinCos(ML_Function("ml_cos")):
  """ Implementation of cosinus function """
  def __init__(self, 
               precision = ML_Binary32, 
               accuracy  = ML_Faithful,
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               fast_path_extract = True,
               target = GenericProcessor(), 
               output_file = "cosf.c", 
               function_name = "cosf", 
               cos_output = True):
    # initializing I/O precision
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

      debug_flag = debug_flag
    )
    self.precision = precision
    self.cos_output = cos_output



  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_SinCos functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cos" if self.cos_output else "mpfr_sin"
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
    frac_pi_index = {ML_Binary32: 5, ML_Binary64: 6}[self.precision]

    frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, RN)
    inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, RN)
    inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, RN)
    # computing k = E(x * frac_pi)
    vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
    k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = debugd)
    fk = Conversion(k, precision = self.precision, tag = "fk")

    inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
    inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

    red_vx_hi = (vx - inv_frac_pi_cst * fk)
    red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
    red_vx_lo_sub = inv_frac_pi_lo_cst * fk
    red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
    vx_d = Conversion(vx, precision = ML_Binary64, tag = "vx_d")
    pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d_hi = (vx_d - inv_frac_pi_cst * fk)
    pre_red_vx_d_hi.set_attributes(tag = "pre_red_vx_d_hi", precision = ML_Binary64, debug = debug_lftolx)
    pre_red_vx_d = pre_red_vx_d_hi - inv_frac_pi_lo_cst * fk
    pre_red_vx_d.set_attributes(tag = "pre_red_vx_d", debug = debug_lftolx, precision = ML_Binary64)


    modk = Modulo(k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "modk", debug = True)

    sel_c = Equal(BitLogicAnd(modk, 2**(frac_pi_index-1)), 2**(frac_pi_index-1))
    sel_c.set_attributes(tag = "sel_c", debug = debugd)
    red_vx = pre_red_vx # Select(sel_c, -pre_red_vx, pre_red_vx)
    red_vx.set_attributes(tag = "red_vx", debug = debug_precision, precision = self.precision)

    red_vx_d = Select(sel_c, -pre_red_vx_d, pre_red_vx_d)
    red_vx_d.set_attributes(tag = "red_vx_d", debug = debug_lftolx, precision = ML_Binary64)

    approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

    Log.report(Log.Info, "approx interval: %s\n" % approx_interval)

    error_goal_approx = S2**-self.precision.get_precision()


    Log.report(Log.Info, "building tabulated approximation for sin and cos")
    poly_degree_vector = [None] * 2**(frac_pi_index+1)


    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

    table_index_size = frac_pi_index+1
    cos_table_hi = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_hi"))
    cos_table_lo = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("cos_table_lo"))
    sin_table = ML_Table(dimensions = [2**table_index_size, 1], storage_precision = self.precision, tag = self.uniquify_name("sin_table"))

    #cos_hi_prec = self.precision.get_sollya_object() # int(self.precision.get_field_size() * 0.7)
    cos_hi_prec =  int(self.precision.get_field_size() - 2)

    for i in xrange(2**(frac_pi_index+1)):
      local_x = i*pi/S2**frac_pi_index
      cos_local_hi = round(cos(local_x), cos_hi_prec, RN)
      cos_table_hi[i][0] = cos_local_hi 
      cos_table_lo[i][0] = round(cos(local_x) - cos_local_hi, self.precision.get_sollya_object(), RN)
      #cos_table_d[i][0] = round(cos(local_x), ML_Binary64.get_sollya_object(), RN)

      sin_table[i][0] = round(sin(local_x), self.precision.get_sollya_object(), RN)


    tabulated_cos_hi = TableLoad(cos_table_hi, modk, 0, tag = "tab_cos_hi", debug = debug_precision) 
    tabulated_cos_lo = TableLoad(cos_table_lo, modk, 0, tag = "tab_cos_lo", debug = debug_precision) 
    tabulated_sin  = TableLoad(sin_table, modk, 0, tag = "tab_sin", debug = debug_precision) 

    Log.report(Log.Info, "building mathematical polynomials for sin and cos")
    poly_degree_cos   = sup(guessdegree(cos(x), approx_interval, S2**-(self.precision.get_field_size()+1)) ) + 2
    poly_degree_sin   = sup(guessdegree(sin(x)/x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 2

    print poly_degree_cos, poly_degree_sin

    poly_degree_cos_list = range(0, poly_degree_cos + 1, 2)
    poly_degree_sin_list = range(0, poly_degree_sin + 1, 2)

    if self.precision is ML_Binary32:
      poly_degree_cos_list = [0, 2, 4, 6]
      poly_degree_sin_list = [0, 2, 4, 6]

    else:
      poly_degree_cos_list = [0, 2, 4, 6]
      poly_degree_sin_list = [0, 2, 4, 6]

    poly_cos_prec_list = [1, 1] + [self.precision] * (len(poly_degree_cos_list) - 2)
    poly_sin_prec_list = [1] + [self.precision] * (len(poly_degree_sin_list) - 1)

    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    poly_object_cos, poly_error_cos = Polynomial.build_from_approximation_with_error(cos(x), poly_degree_cos_list, poly_cos_prec_list, approx_interval, absolute, error_function = error_function)
    poly_object_sin, poly_error_sin = Polynomial.build_from_approximation_with_error(sin(x)/x, poly_degree_sin_list, poly_sin_prec_list, approx_interval, absolute, error_function = error_function)
  
    print poly_object_cos.get_sollya_object()
    print poly_object_sin.get_sollya_object()
    print "poly_error: ", poly_error_cos, poly_error_sin

    poly_cos = polynomial_scheme_builder(poly_object_cos.sub_poly(start_index = 4, offset = 1), red_vx, unified_precision = self.precision)
    poly_sin = polynomial_scheme_builder(poly_object_sin.sub_poly(start_index = 2), red_vx, unified_precision = self.precision)
    poly_cos.set_attributes(tag = "poly_cos", debug = debug_precision)
    poly_sin.set_attributes(tag = "poly_sin", debug = debug_precision)


    cos_eval_d = (tabulated_cos_hi - red_vx * (tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))) + tabulated_cos_lo


    tab_cos_dd = (tabulated_cos_hi + tabulated_cos_lo).modify_attributes(precision = ML_DoubleDouble, tag = "tab_cos_dd")

    #cos_eval_lo_dd_op0 = tab_cos_dd
    cos_eval_lo_dd_op1 = ((-red_vx) * (tabulated_sin)).modify_attributes(precision = ML_DoubleDouble)
    cos_eval_lo_dd_op2 = ((-tabulated_cos_hi) * ((red_vx * red_vx) * 0.5)).modify_attributes(precision = self.precision)
    cos_eval_lo_dd_op3 = ((-tabulated_sin) * (poly_sin * red_vx)).modify_attributes(precision = self.precision)
    cos_eval_lo_dd_op4 = (tabulated_cos_hi * (red_vx * poly_cos)).modify_attributes(precision = self.precision)

    cos_eval_d2_add = AdditionN(
                        tabulated_cos_hi,
                        AdditionN(
                          cos_eval_lo_dd_op1, 
                          AdditionN(
                            cos_eval_lo_dd_op2,
                            cos_eval_lo_dd_op3,
                            cos_eval_lo_dd_op4,
                            tabulated_cos_lo,
                            unbreakable = True,
                            precision = self.precision
                          ),
                          precision = ML_DoubleDouble,
                          unbreakable = True
                        ),
                        precision = ML_DoubleDouble,
                        unbreakable = True
                      )


    cos_eval_d2_add.set_attributes(unbreakable = True, precision = ML_DoubleDouble, tag = "cos_eval_d2_add", debug = debug_ddtolx)
    cos_eval_d2 = cos_eval_d2_add.hi + cos_eval_d2_add.lo
    cos_eval_d2.set_attributes(tag = "cos_eval_d2", debug = debug_precision, precision = self.precision)


    # cos_eval_d = tabulated_cos_hi + FusedMultiplyAdd(- red_vx, (tabulated_sin + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos)))), tabulated_cos_lo)

    cos_eval_d.set_attributes(tag = "cos_eval_d", debug = debug_precision, precision = self.precision)
    
    exact_sub = (tabulated_cos_hi - red_vx)
    exact_sub.set_attributes(tag = "exact_sub", debug = debug_precision, unbreakable = True, prevent_optimization = True)

    # cos_eval_2 and cos_eval_3 are to be used when tabulated_sin is positive and very close to 1.0
    cos_eval_2 = exact_sub + ((- red_vx * ((tabulated_sin - 1) + (tabulated_cos_hi * red_vx * 0.5 + (tabulated_sin * poly_sin + (- tabulated_cos_hi * poly_cos))))) + tabulated_cos_lo)
    cos_eval_2.set_attributes(tag = "cos_eval_2", precision = self.precision, debug = debug_precision)

    cos_eval_3 = (tabulated_cos_hi + (- red_vx - red_vx * ((tabulated_sin - 1) + tabulated_cos_hi * red_vx * 0.5 + tabulated_sin * poly_sin - tabulated_cos_hi * poly_cos))) + tabulated_cos_lo 
    cos_eval_3.set_attributes(tag = "cos_eval_3", precision = self.precision, debug = debug_precision)

    # computing evaluation error for cos_eval_d
    cos_eval_d_eerror = []
    for i in xrange(0, 2**(frac_pi_index-1)-3):
      copy_map = {
        #tabulated_cos_hi : Constant(cos_table_hi[i][0], tag = "tabulated_cos_hi", precision = ML_Binary64), 
        #tabulated_sin    : Constant(sin_table[i][0], tag = "tabulated_sin", precision = ML_Binary64),
        tabulated_cos_hi : Variable("tabulated_cos_hi", interval = Interval(cos_table_hi[i][0]), precision = ML_Binary64), 
        tabulated_cos_lo : Variable("tabulated_cos_lo", interval = Interval(cos_table_lo[i][0]), precision = ML_Binary64), 
        tabulated_sin    : Variable("tabulated_sin", interval = Interval(sin_table[i][0]), precision = ML_Binary64),
        red_vx           : Variable("red_vx", precision = ML_Binary64, interval = approx_interval),
      }

      cos_eval_d_eerror_local = sup(abs(self.get_eval_error(cos_eval_d, variable_copy_map = copy_map, relative_error = True)))
      cos_eval_d_eerror.append(cos_eval_d_eerror_local)
      if cos_eval_d_eerror_local > S2**-52:
        print "cos_eval_d_eerror_local: ", i, cos_eval_d_eerror_local

    for i in xrange(2**(frac_pi_index-1)-3, 2**(frac_pi_index-1)-1):
      copy_map = {
        #tabulated_cos_hi : Constant(cos_table_hi[i][0], tag = "tabulated_cos_hi", precision = ML_Binary64), 
        #tabulated_sin    : Constant(sin_table[i][0], tag = "tabulated_sin", precision = ML_Binary64),
        tabulated_cos_hi : Variable("tabulated_cos_hi", interval = Interval(cos_table_hi[i][0]), precision = ML_Binary64), 
        tabulated_cos_lo : Variable("tabulated_cos_lo", interval = Interval(cos_table_lo[i][0]), precision = ML_Binary64), 
        tabulated_sin    : Variable("tabulated_sin", interval = Interval(sin_table[i][0]), precision = ML_Binary64),
        red_vx           : Variable("red_vx", precision = ML_Binary64, interval = approx_interval),
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
    cast_int_precision = {ML_Binary64: ML_Int64, ML_Binary32: ML_Int32}[self.precision]

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
    cast_int_precision = {ML_Binary64: ML_Int64, ML_Binary32: ML_Int32}[self.precision]

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

    int_precision = ML_Int64
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





if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_sincos", default_output_file = "new_sincos.c" )
  # argument extraction 
  cos_output = arg_template.test_flag_option("--cos", True, False, parse_arg = arg_template.parse_arg, help_str = "select cos output") 
  enable_subexpr_sharing = arg_template.test_flag_option("--enable-subexpr-sharing", True, False, parse_arg = arg_template.parse_arg, help_str = "force subexpression sharing")

  parse_arg_index_list = arg_template.sys_arg_extraction()
  arg_template.check_args(parse_arg_index_list)


  ml_sincos = ML_SinCos(arg_template.precision, 
                     libm_compliant            = arg_template.libm_compliant, 
                     debug_flag                = arg_template.debug_flag, 
                     target                    = arg_template.target, 
                     fuse_fma                  = arg_template.fuse_fma, 
                     fast_path_extract         = arg_template.fast_path,
                     function_name             = arg_template.function_name,
                     accuracy                  = arg_template.accuracy,
                     output_file               = arg_template.output_file,
                     cos_output                = cos_output)
  ml_sincos.gen_implementation(display_after_opt = arg_template.display_after_opt, enable_subexpr_sharing = enable_subexpr_sharing)
