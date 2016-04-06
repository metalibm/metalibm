# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, cosh, guessdegree, dirtyinfnorm, RN

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_HyperbolicCosine(ML_Function("ml_cosh")):
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = GenericProcessor(), 
             output_file = "my_cosh.c", 
             function_name = "my_cosh",
             language = C_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "cosh",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,
      libm_compliant = libm_compliant,

      processor = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      language = language,
      vector_size = vector_size,
      arg_template = arg_template
    )

    self.accuracy  = accuracy
    self.precision = precision

  def generate_scheme(self):
    # declaring target and instantiating optimization engine

    vx = self.implementation.add_input_variable("x", self.precision) 

    Log.set_dump_stdout(True)

    Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
    if self.debug_flag: 
        Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    index_size = 3


    # argument reduction
    arg_reg_value = log(2)/2**index_size
    inv_log2_value = round(1/arg_reg_value, self.precision.get_sollya_object(), RN)
    inv_log2_cst = Constant(inv_log2_value, precision = self.precision)

    # for r_hi to be accurate we ensure k * log2_hi_value_cst is exact
    # by limiting the number of non-zero bits in log2_hi_value_cst
    # cosh(x) ~ exp(abs(x))/2  for a big enough x
    # cosh(x) > 2^1023 <=> exp(x) > 2^1024 <=> x > log(2^21024)
    # k = inv_log2_value * x 
    # -1 for guard
    max_k_approx = inv_log2_value * log(sollya.SollyaObject(2)**1024)
    max_k_bitsize = int(ceil(log2(max_k_approx)))
    Log.report(Log.Info, "max_k_bitsize: %d" % max_k_bitsize)
    log2_hi_value_precision = self.precision.get_precision() - max_k_bitsize - 1 

    log2_hi_value = round(arg_reg_value, log2_hi_value_precision, RN)
    log2_lo_value = round(arg_reg_value - log2_hi_value, self.precision.get_sollya_object(), RN)
    log2_hi_value_cst = Constant(log2_hi_value, tag = "log2_hi_value", precision = self.precision)
    log2_lo_value_cst = Constant(log2_lo_value, tag = "log2_lo_value", precision = self.precision)

    k = NearestInteger(Multiplication(inv_log2_cst, vx), precision = self.precision)
    r_hi = vx - k * log2_hi_value_cst
    r_lo = -k * log2_lo_value_cst
    # reduced argument
    r = r_hi + r_lo
    r.set_attributes(tag = "r", debug = debug_multi)

    approx_interval = Interval(-arg_reg_value/2, arg_reg_value/2)
    error_goal_approx = 2**-(self.precision.get_precision())
    int_precision = {ML_Binary32: ML_Int32, ML_Binary64: ML_Int64}[self.precision]

    poly_degree = sup(guessdegree(exp(sollya.x), approx_interval, error_goal_approx)) 
    precision_list = [1] + [self.precision] * (poly_degree)

    k_integer = Conversion(k, precision = int_precision, tag = "k_integer", debug = debug_multi)
    k_hi = BitLogicRightShift(k_integer, Constant(index_size), tag = "k_int_hi", precision = int_precision, debug = debug_multi)
    k_lo = Modulo(k_integer, 2**index_size, tag = "k_int_lo", precision = int_precision, debug = debug_multi)
    pow_exp = ExponentInsertion(Conversion(k_hi, precision = int_precision), precision = self.precision, tag = "pow_exp", debug = debug_multi)

    exp_table = ML_Table(dimensions = [2 * 2**index_size, 4], storage_precision = self.precision, tag = self.uniquify_name("exp2_table"))
    for i in range(2 * 2**index_size):
      input_value = i - 2**index_size if i >= 2**index_size else i 
      # using SollyaObject wrapper to force evaluation by sollya
      # with higher precision
      exp_value  = sollya.SollyaObject(2)**((input_value)* 2**-index_size)
      mexp_value = sollya.SollyaObject(2)**((-input_value)* 2**-index_size)
      pos_value_hi = round(exp_value, self.precision.get_sollya_object(), RN)
      pos_value_lo = round(exp_value - pos_value_hi, self.precision.get_sollya_object(), RN)
      neg_value_hi = round(mexp_value, self.precision.get_sollya_object(), RN)
      neg_value_lo = round(mexp_value - neg_value_hi, self.precision.get_sollya_object(), RN)
      exp_table[i][0] = neg_value_hi
      exp_table[i][1] = neg_value_lo
      exp_table[i][2] = pos_value_hi
      exp_table[i][3] = pos_value_lo

    # log2_value = log(2) / 2^index_size
    # cosh(x) = 1/2 * (exp(x) + exp(-x))
    # exp(x) = exp(x - k * log2_value + k * log2_value
    #  
    # r = x - k * log2_value
    # exp(x) = exp(r) * 2 ^ (k / 2^index_size)
    #
    # k / 2^index_size = h + l * 2^-index_size
    # exp(x) = exp(r) * 2^h * 2^(l *2^-index_size)
    #
    # cosh(x) = exp(r) * 2^(h-1) 2^(l *2^-index_size) + exp(-r) * 2^(-h-1) * 2^(-l *2^-index_size)
    #
    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(exp(sollya.x), poly_degree, precision_list, approx_interval, sollya.absolute, error_function = error_function)

    print "poly_approx_error: ", poly_approx_error, float(log2(poly_approx_error))

    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme
    poly_pos = polynomial_scheme_builder(poly_object.sub_poly(start_index = 1), r, unified_precision = self.precision)
    poly_pos.set_attributes(tag = "poly_pos", debug = debug_multi)

    poly_neg = polynomial_scheme_builder(poly_object.sub_poly(start_index = 1), -r, unified_precision = self.precision)
    poly_neg.set_attributes(tag = "poly_neg", debug = debug_multi)

    table_index = Addition(k_lo, Constant(2**index_size, precision = int_precision), precision = int_precision, tag = "table_index", debug = debug_multi)

    neg_value_load_hi = TableLoad(exp_table, table_index, 0, tag = "neg_value_load_hi", debug = debug_multi)
    neg_value_load_lo = TableLoad(exp_table, table_index, 1, tag = "neg_value_load_lo", debug = debug_multi)
    pos_value_load_hi = TableLoad(exp_table, table_index, 2, tag = "pos_value_load_hi", debug = debug_multi)
    pos_value_load_lo = TableLoad(exp_table, table_index, 3, tag = "pos_value_load_lo", debug = debug_multi)

    k_plus = Subtraction(k_hi, Constant(1, precision = int_precision), precision = int_precision, tag = "k_plus", debug = debug_multi)
    k_neg = Subtraction(-k_hi, Constant(1, precision = int_precision), precision = int_precision, tag = "k_neg", debug = debug_multi)

    pow_exp_pos = ExponentInsertion(k_plus, precision = self.precision)
    pow_exp_neg = ExponentInsertion(k_neg, precision = self.precision)

    pos_exp = (pos_value_load_hi + (pos_value_load_hi * poly_pos + (pos_value_load_lo + pos_value_load_lo * poly_pos))) * pow_exp_pos 
    pos_exp.set_attributes(tag = "pos_exp", debug = debug_multi)

    neg_exp = (neg_value_load_hi + (neg_value_load_hi * poly_neg + (neg_value_load_lo + neg_value_load_lo * poly_neg))) * pow_exp_neg 
    neg_exp.set_attributes(tag = "neg_exp", debug = debug_multi)

    result = Addition(
                pos_exp,
                neg_exp,
                precision = self.precision,
                tag = "result",
                debug = debug_multi
              )

    # ov_flag = Comparison(vx_int_hi, Constant(self.precision.get_emax(), precision = self.precision), specifier = Comparison.Greater)

    # main scheme
    Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    scheme = Statement(
                Return(
                    result
                  ))

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_cosh"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call

  def numeric_emulate(self, input_value):
    return cosh(input_value)


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_cosh", default_output_file = "new_cosh.c" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_exp2          = ML_HyperbolicCosine(args)

    ml_exp2.gen_implementation()
