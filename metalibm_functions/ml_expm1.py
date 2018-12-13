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
###############################################################################
import sys

import sollya

from sollya import Interval, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm
S2 = sollya.SollyaObject(2)

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.core.special_values import FP_PlusInfty, FP_QNaN


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_ExponentialM1_Red(ML_FunctionBasis):
  function_name = "ml_expm1"
  def __init__(self, args):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_ExponentialM1_Red,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_expm1 = {
        "output_file": "my_expm1.c",
        "function_name": "my_expm1",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_expm1.update(kw)
    return DefaultArgTemplate(**default_args_expm1)

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
    
    C_m1 = Constant(-1, precision = self.precision)
    
    test_NaN_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debug_multi, tag = "NaN_or_inf", precision = ML_Bool)
    test_NaN = Test(vx, specifier = Test.IsNaN, likely = False, debug = debug_multi, tag = "is_NaN", precision = ML_Bool)
    test_inf = Comparison(vx, 0, specifier = Comparison.Greater, debug = debug_multi, tag = "sign", precision = ML_Bool, likely = False);
    
    #  Infnty input
    infty_return = Statement(ConditionBlock(test_inf, Return(FP_PlusInfty(self.precision)), Return(C_m1)))
    #  non-std input (inf/nan)
    specific_return = ConditionBlock(test_NaN, Return(FP_QNaN(self.precision)), infty_return)
    
    # Over/Underflow Tests
    
    precision_emax = self.precision.get_emax()
    precision_max_value = S2**(precision_emax + 1)
    expm1_overflow_bound = ceil(log(precision_max_value + 1))
    overflow_test = Comparison(vx, expm1_overflow_bound, likely = False, specifier = Comparison.Greater, precision = ML_Bool)
    overflow_return = Statement(Return(FP_PlusInfty(self.precision)))
    
    precision_emin = self.precision.get_emin_subnormal()
    precision_min_value = S2** precision_emin
    expm1_underflow_bound = floor(log(precision_min_value) + 1)
    underflow_test = Comparison(vx, expm1_underflow_bound, likely = False, specifier = Comparison.Less, precision = ML_Bool)
    underflow_return = Statement(Return(C_m1))
    
    sollya_precision = {ML_Binary32: sollya.binary32, ML_Binary64: sollya.binary64}[self.precision]
    int_precision = {ML_Binary32: ML_Int32, ML_Binary64: ML_Int64}[self.precision]
    
    # Constants
    
    log_2 = round(log(2), sollya_precision, sollya.RN)
    invlog2 = round(1/log(2), sollya_precision, sollya.RN)
    log_2_cst = Constant(log_2, precision = self.precision)
    
    interval_vx = Interval(expm1_underflow_bound, expm1_overflow_bound)
    interval_fk = interval_vx * invlog2
    interval_k = Interval(floor(inf(interval_fk)), ceil(sup(interval_fk)))
    
    log2_hi_precision = self.precision.get_field_size() - 6
    log2_hi = round(log(2), log2_hi_precision, sollya.RN)
    log2_lo = round(log(2) - log2_hi, sollya_precision, sollya.RN)


    # Reduction
    unround_k = vx * invlog2
    ik = NearestInteger(unround_k, precision = int_precision, debug = debug_multi, tag = "ik")
    k = Conversion(ik, precision = self.precision, tag = "k")
    
    red_coeff1 = Multiplication(k, log2_hi, precision = self.precision)
    red_coeff2 = Multiplication(Negation(k, precision = self.precision), log2_lo, precision = self.precision)
    
    pre_sub_mul = Subtraction(vx, red_coeff1, precision  = self.precision)
    
    s = Addition(pre_sub_mul, red_coeff2, precision = self.precision)
    z = Subtraction(s, pre_sub_mul, precision = self.precision)
    t = Subtraction(red_coeff2, z, precision = self.precision)
    
    r = Addition(s, t, precision = self.precision)
    
    r.set_attributes(tag = "r", debug = debug_multi)
    
    r_interval = Interval(-log_2/S2, log_2/S2)
    
    local_ulp = sup(ulp(exp(r_interval), self.precision))
    
    Log.report(Log.Info, "ulp: ", local_ulp)
    error_goal = S2**-1*local_ulp
    Log.report(Log.Info, "error goal: ", error_goal)
    
    
    # Polynomial Approx
    error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)
    Log.report(Log.Info, "\033[33;1m Building polynomial \033[0m\n")
    
    poly_degree = sup(guessdegree(expm1(sollya.x), r_interval, error_goal) + 1)
    
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme
    poly_degree_list = range(0, poly_degree)
    
    precision_list = [self.precision] *(len(poly_degree_list) + 1)
    poly_object, poly_error = Polynomial.build_from_approximation_with_error(expm1(sollya.x), poly_degree, precision_list, r_interval, sollya.absolute, error_function = error_function)
    sub_poly = poly_object.sub_poly(start_index = 2)
    Log.report(Log.Info, "Poly : %s" % sub_poly)
    Log.report(Log.Info, "poly error : {} / {:d}".format(poly_error, int(sollya.log2(poly_error))))
    pre_sub_poly = polynomial_scheme_builder(sub_poly, r, unified_precision = self.precision)
    poly = r + pre_sub_poly
    poly.set_attributes(tag = "poly", debug = debug_multi)
    
    exp_k = ExponentInsertion(ik, tag = "exp_k", debug = debug_multi, precision = self.precision)
    exp_mk = ExponentInsertion(-ik, tag = "exp_mk", debug = debug_multi, precision = self.precision)
    
    diff = 1 - exp_mk
    diff.set_attributes(tag = "diff", debug = debug_multi) 
    
    # Late Tests
    late_overflow_test = Comparison(ik, self.precision.get_emax(), specifier = Comparison.Greater, likely = False, debug = debug_multi, tag = "late_overflow_test")
    
    overflow_exp_offset = (self.precision.get_emax() - self.precision.get_field_size() / 2)
    diff_k = ik - overflow_exp_offset 
    
    exp_diff_k = ExponentInsertion(diff_k, precision = self.precision, tag = "exp_diff_k", debug = debug_multi)
    exp_oflow_offset = ExponentInsertion(overflow_exp_offset, precision = self.precision, tag = "exp_offset", debug = debug_multi)
    
    late_overflow_result = (exp_diff_k * (1 + poly)) * exp_oflow_offset - 1.0
    
    late_overflow_return = ConditionBlock(
        Test(late_overflow_result, specifier = Test.IsInfty, likely = False), 
        ExpRaiseReturn(ML_FPE_Overflow, return_value = FP_PlusInfty(self.precision)), 
        Return(late_overflow_result)
        )


    late_underflow_test = Comparison(k, self.precision.get_emin_normal(), specifier = Comparison.LessOrEqual, likely = False)
    
    underflow_exp_offset = 2 * self.precision.get_field_size()
    corrected_coeff = ik + underflow_exp_offset
    
    exp_corrected = ExponentInsertion(corrected_coeff, precision = self.precision)
    exp_uflow_offset = ExponentInsertion(-underflow_exp_offset, precision = self.precision)
    
    late_underflow_result = ( exp_corrected * (1 + poly)) * exp_uflow_offset - 1.0
    
    test_subnormal = Test(late_underflow_result, specifier = Test.IsSubnormal, likely = False)
    
    late_underflow_return = Statement(
        ConditionBlock(
            test_subnormal, 
            ExpRaiseReturn(ML_FPE_Underflow, return_value = late_underflow_result)), 
            Return(late_underflow_result)
            )
    
    # Reconstruction
    
    std_result = exp_k * ( poly + diff )
    std_result.set_attributes(tag = "result", debug = debug_multi)
    
    result_scheme = ConditionBlock(
        late_overflow_test, 
        late_overflow_return, 
        ConditionBlock(
            late_underflow_test, 
            late_underflow_return, 
            Return(std_result)
            )
        )
        
    std_return = ConditionBlock(
        overflow_test, 
        overflow_return, 
        ConditionBlock(
            underflow_test, 
            underflow_return, 
            result_scheme)
        )
        
    scheme = ConditionBlock(
        test_NaN_or_inf, 
        Statement(specific_return), 
        std_return
        )

    return scheme


  def numeric_emulate(self, input_value):
    return expm1(input_value)

  standard_test_cases = [[sollya.parse(x)] for x in ["0x1.9b3216p-2", "0x1.8c108p-2"]]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_ExponentialM1_Red.get_default_args())
    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()
 
    ml_expm1_red         = ML_ExponentialM1_Red(args)

    ml_expm1_red.gen_implementation()
