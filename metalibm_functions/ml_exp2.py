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

from sollya import Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN
S2 = sollya.SollyaObject(2)

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.core.special_values import FP_PlusInfty

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_Exp2(ML_FunctionBasis):
  function_name = "ml_exp2"
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self,
      args
    )


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Exponential,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_exp2 = {
        "output_file": "ml_exp2.c",
        "function_name": "ml_exp2",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_exp2.update(kw)
    return DefaultArgTemplate(**default_args_exp2)

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

    r_interval = Interval(-0.5, 0.5)

    local_ulp = sup(ulp(2**r_interval, self.precision))
    Log.report(Log.Info, "ulp: ", local_ulp)
    error_goal = S2**-1*local_ulp
    Log.report(Log.Info, "error goal: ", error_goal)

    sollya_precision = {ML_Binary32: sollya.binary32, ML_Binary64: sollya.binary64}[self.precision]
    int_precision = {ML_Binary32: ML_Int32, ML_Binary64: ML_Int64}[self.precision]

    #Argument Reduction
    vx_int = NearestInteger(vx, precision = int_precision, tag = 'vx_int', debug = debug_multi)
    vx_intf = Conversion(vx_int, precision = self.precision)
    vx_r = vx - vx_intf
    vx_r.set_attributes(tag = "vx_r", debug = debug_multi)
    degree = sup(guessdegree(2**(sollya.x), r_interval, error_goal)) + 2
    precision_list = [1] + [self.precision] * degree


    exp_X = ExponentInsertion(vx_int, tag = "exp_X", debug = debug_multi, precision = self.precision)

    #Polynomial Approx
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

    poly_object, poly_error = Polynomial.build_from_approximation_with_error(2**(sollya.x) - 1 , degree, precision_list, r_interval, sollya.absolute)
    Log.report(Log.Info, "Poly : %s" % poly_object)
    Log.report(Log.Info, "poly_error : ", poly_error)
    poly = polynomial_scheme_builder(poly_object.sub_poly(start_index = 1), vx_r, unified_precision = self.precision)
    poly.set_attributes(tag = "poly", debug = debug_multi)



    #Handling special cases
    oflow_bound = Constant(self.precision.get_emax() + 1, precision = self.precision)
    subnormal_bound = self.precision.get_emin_subnormal()
    uflow_bound = self.precision.get_emin_normal()
    Log.report(Log.Info, "oflow : ", oflow_bound)
    #print "uflow : ", uflow_bound
    #print "sub : ", subnormal_bound
    test_overflow = Comparison(vx, oflow_bound, specifier = Comparison.GreaterOrEqual)
    test_overflow.set_attributes(tag = "oflow_test", debug = debug_multi, likely = False, precision = ML_Bool)

    test_underflow = Comparison(vx, uflow_bound, specifier = Comparison.Less)
    test_underflow.set_attributes(tag = "uflow_test", debug = debug_multi, likely = False, precision = ML_Bool)

    test_subnormal = Comparison(vx, subnormal_bound, specifier = Comparison.Greater)
    test_subnormal.set_attributes(tag = "sub_test", debug = debug_multi, likely = False, precision = ML_Bool)

    subnormal_offset = - (uflow_bound - vx_int)
    subnormal_offset.set_attributes( tag = "offset", debug = debug_multi)
    exp_offset = ExponentInsertion(subnormal_offset, precision = self.precision, debug = debug_multi, tag = "exp_offset")
    exp_min = ExponentInsertion(uflow_bound, precision = self.precision, debug = debug_multi, tag = "exp_min")
    subnormal_result = exp_offset*exp_min*poly + exp_offset*exp_min
    
    test_std = LogicalOr(test_overflow, test_underflow, precision = ML_Bool, tag = "std_test", likely = False)
    
    #Reconstruction
    result = exp_X*poly + exp_X
    result.set_attributes(tag = "result", debug = debug_multi)
    
    C0 = Constant(0, precision = self.precision)
    
    return_inf = Return(FP_PlusInfty(self.precision))
    return_C0 = Return(C0)
    return_sub = Return(subnormal_result)
    return_std = Return(result)

    non_std_statement = Statement(
      ConditionBlock(
        test_overflow,
        return_inf,
        ConditionBlock(
          test_subnormal,
          return_sub,
          return_C0
          )
        )
      )

    scheme = Statement(
      ConditionBlock(
        test_std,
        non_std_statement,
        return_std
      )
    )

    return scheme

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_exp"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, require_header = ["mpfr.h"])
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Mpfr_t, ML_Int32], ML_Int32, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))

    return mpfr_call

  def numeric_emulate(self, input_value):
    return sollya.SollyaObject(2)**(input_value)

  standard_test_cases =[[sollya.parse(x)] for x in  ["0x1.ffead1bac7ad2p+9", "-0x1.ee9cb4p+1", "-0x1.db0928p+3"]]

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Exp2.get_default_args())
    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_exp2          = ML_Exp2(args)

    ml_exp2.gen_implementation()
