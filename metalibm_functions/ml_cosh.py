# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

import sollya

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, cosh,
    guessdegree, dirtyinfnorm, RN, acosh, RD
)
S2 = sollya.SollyaObject(2)
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Result, FO_Arg
)
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.core.special_values import FP_PlusInfty


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_HyperbolicCosine(ML_Function("ml_cosh")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args=args)

  @staticmethod
  def get_default_args(**args):
    """ Generate a default argument structure set specifically for
        the Hyperbolic Cosine """
    default_cosh_args = {
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor(),
        "output_file": "my_cosh.c",
        "function_name": "my_cosh",
        "language": C_Code,
        "vector_size": 1
    }
    default_cosh_args.update(args)
    return DefaultArgTemplate(**default_cosh_args)

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

    vx = Abs(vx)
    int_precision = self.precision.get_integer_format()

    # argument reduction
    arg_reg_value = log(2)/2**index_size
    inv_log2_value = round(1/arg_reg_value, self.precision.get_sollya_object(), RN)
    inv_log2_cst = Constant(inv_log2_value, precision = self.precision, tag = "inv_log2")

    # for r_hi to be accurate we ensure k * log2_hi_value_cst is exact
    # by limiting the number of non-zero bits in log2_hi_value_cst
    # cosh(x) ~ exp(abs(x))/2  for a big enough x
    # cosh(x) > 2^1023 <=> exp(x) > 2^1024 <=> x > log(2^1024)
    # k = inv_log2_value * x 
    # -1 for guard
    max_k_approx  = inv_log2_value * log(sollya.SollyaObject(2)**1024)
    max_k_bitsize = int(ceil(log2(max_k_approx)))
    Log.report(Log.Info, "max_k_bitsize: %d" % max_k_bitsize)
    log2_hi_value_precision = self.precision.get_precision() - max_k_bitsize - 1 

    log2_hi_value = round(arg_reg_value, log2_hi_value_precision, RN)
    log2_lo_value = round(arg_reg_value - log2_hi_value, self.precision.get_sollya_object(), RN)
    log2_hi_value_cst = Constant(log2_hi_value, tag = "log2_hi_value", precision = self.precision)
    log2_lo_value_cst = Constant(log2_lo_value, tag = "log2_lo_value", precision = self.precision)

    k = Trunc(Multiplication(inv_log2_cst, vx), precision = self.precision)
    k_log2 = Multiplication(k, log2_hi_value_cst, precision = self.precision, exact = True, tag = "k_log2", unbreakable = True)
    r_hi = vx - k_log2
    r_hi.set_attributes(tag = "r_hi", debug = debug_multi, unbreakable = True)
    r_lo = -k * log2_lo_value_cst
    # reduced argument
    r = r_hi + r_lo
    r.set_attributes(tag = "r", debug = debug_multi)

    r_eval_error = self.get_eval_error(r_hi, variable_copy_map = 
      {
        vx: Variable("vx", interval = Interval(0, 715), precision = self.precision),
        k: Variable("k", interval = Interval(0, 1024), precision = self.precision)
      })

    approx_interval = Interval(-arg_reg_value, arg_reg_value)
    error_goal_approx = 2**-(self.precision.get_precision())

    poly_degree = sup(guessdegree(exp(sollya.x), approx_interval, error_goal_approx)) 
    precision_list = [1] + [self.precision] * (poly_degree)

    k_integer = Conversion(k, precision = int_precision, tag = "k_integer", debug = debug_multi)
    k_hi = BitLogicRightShift(k_integer, Constant(index_size), tag = "k_int_hi", precision = int_precision, debug = debug_multi)
    k_lo = Modulo(k_integer, 2**index_size, tag = "k_int_lo", precision = int_precision, debug = debug_multi)
    pow_exp = ExponentInsertion(Conversion(k_hi, precision = int_precision), precision = self.precision, tag = "pow_exp", debug = debug_multi)

    exp_table = ML_NewTable(dimensions = [2 * 2**index_size, 4], storage_precision = self.precision, tag = self.uniquify_name("exp2_table"))
    for i in range(2 * 2**index_size):
      input_value = i - 2**index_size if i >= 2**index_size else i 

      reduced_hi_prec = int(self.precision.get_mantissa_size() * 2 / 3.0)
      # using SollyaObject wrapper to force evaluation by sollya
      # with higher precision
      exp_value  = sollya.SollyaObject(2)**((input_value)* 2**-index_size)
      mexp_value = sollya.SollyaObject(2)**((-input_value)* 2**-index_size)
      pos_value_hi = round(exp_value, reduced_hi_prec, RN)
      pos_value_lo = round(exp_value - pos_value_hi, self.precision.get_sollya_object(), RN)
      neg_value_hi = round(mexp_value, reduced_hi_prec, RN)
      neg_value_lo = round(mexp_value - neg_value_hi, self.precision.get_sollya_object(), RN)
      exp_table[i][0] = neg_value_hi
      exp_table[i][1] = neg_value_lo
      exp_table[i][2] = pos_value_hi
      exp_table[i][3] = pos_value_lo

    # log2_value = log(2) / 2^index_size
    # cosh(x) = 1/2 * (exp(x) + exp(-x))
    # exp(x) = exp(x - k * log2_value + k * log2_value)
    #  
    # r = x - k * log2_value
    # exp(x) = exp(r) * 2 ^ (k / 2^index_size)
    #
    # k / 2^index_size = h + l * 2^-index_size, with k, h, l integers
    # exp(x) = exp(r) * 2^h * 2^(l *2^-index_size)
    #
    # cosh(x) = exp(r) * 2^(h-1) 2^(l *2^-index_size) + exp(-r) * 2^(-h-1) * 2^(-l *2^-index_size)
    # S=2^(h-1), T = 2^(-h-1)
    # exp(r)  = 1 + poly_pos(r)
    # exp(-r) = 1 + poly_neg(r)
    # 2^(l / 2^index_size)  = pos_value_hi + pos_value_lo 
    # 2^(-l / 2^index_size) = neg_value_hi + neg_value_lo 
    #
    # cosh(x) = 
    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

    poly_object, poly_approx_error = Polynomial.build_from_approximation_with_error(exp(sollya.x), poly_degree, precision_list, approx_interval, sollya.absolute, error_function = error_function)


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

    k_plus = Max(Subtraction(k_hi, Constant(1, precision = int_precision), precision = int_precision, tag = "k_plus", debug = debug_multi), Constant(self.precision.get_emin_normal(), precision = int_precision))
    k_neg = Max(Subtraction(-k_hi, Constant(1, precision = int_precision), precision = int_precision, tag = "k_neg", debug = debug_multi), Constant(self.precision.get_emin_normal(), precision = int_precision))

    pow_exp_pos = ExponentInsertion(k_plus, precision=self.precision, tag="pow_exp_pos", debug=debug_multi)
    pow_exp_neg = ExponentInsertion(k_neg, precision=self.precision, tag="pow_exp_neg", debug=debug_multi)

    hi_terms = (pos_value_load_hi * pow_exp_pos + neg_value_load_hi * pow_exp_neg)
    hi_terms.set_attributes(tag = "hi_terms")


    pos_exp = (pos_value_load_hi * poly_pos + (pos_value_load_lo + pos_value_load_lo * poly_pos)) * pow_exp_pos 
    pos_exp.set_attributes(tag = "pos_exp", debug = debug_multi)

    neg_exp = (neg_value_load_hi * poly_neg + (neg_value_load_lo + neg_value_load_lo * poly_neg)) * pow_exp_neg 
    neg_exp.set_attributes(tag = "neg_exp", debug = debug_multi)

    result = Addition(
                Addition(
                  pos_exp,
                  neg_exp,
                  precision = self.precision,
                ),
                hi_terms,
                precision = self.precision,
                tag = "result",
                debug = debug_multi
              )

    # ov_value 
    ov_value = round(acosh(self.precision.get_max_value()), self.precision.get_sollya_object(), RD)
    ov_flag = Comparison(Abs(vx), Constant(ov_value, precision = self.precision), specifier = Comparison.Greater, tag="ov_flag")

    # main scheme
    Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
    scheme = Statement(
                Return(
                  Select(
                    ov_flag,
                    FP_PlusInfty(self.precision),
                    result
                  )))

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

  standard_test_cases =[(sollya_parse(x), None) for x in  ["1.705527","0.935715", "-0x1.e45322ap-1", "0x1.b8ef9f54p-1", "-0x1.b8ef9f54p-1", "0x1.b6fdb8a8p-1"]]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_arg=ML_HyperbolicCosine.get_default_args()
    )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_cosh          = ML_HyperbolicCosine(args)

    ml_cosh.gen_implementation()
