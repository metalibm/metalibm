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

from sollya import S2, SollyaObject, Interval, log2, log10, acos, sup

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_NewArgTemplate

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  
from metalibm_core.utility.debug_utils import *

class ML_Acos(ML_FunctionBasis):
  function_name = "ml_acos"
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Acos,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_acos = {
        "output_file": "my_exp.c",
        "function_name": "my_exp",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_acos.update(kw)
    return DefaultArgTemplate(**default_args_acos)

  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_acos"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call


  def generate_scheme(self):
    """ generate scheme """
    vx = self.implementation.add_input_variable("x", self.get_input_precision())

    # retrieving processor inverse approximation table
    lo_bound_global = SollyaObject(0.0)
    hi_bound_global = SollyaObject(0.75)
    approx_interval = Interval(lo_bound_global, hi_bound_global)
    approx_interval_size = hi_bound_global - lo_bound_global

    # table creation
    table_index_size = 7
    field_index_size = 2
    exp_index_size = table_index_size - field_index_size

    table_size = 2**table_index_size
    table_index_range = range(table_size)

    local_degree = 9
    coeff_table = ML_NewTable(
        dimensions=[table_size, local_degree],
        storage_precision=self.precision)

    exp_lo = 2**exp_index_size
    for i in table_index_range:
      lo_bound = (1.0 + (i % 2**field_index_size) * S2**-field_index_size) * S2**(i / 2**field_index_size - exp_lo)
      hi_bound = (1.0 + ((i % 2**field_index_size) + 1) * S2**-field_index_size) * S2**(i / 2**field_index_size - exp_lo)
      local_approx_interval = Interval(lo_bound, hi_bound)
      local_poly_object, local_error = Polynomial.build_from_approximation_with_error(
        acos(1 - sollya.x),
        local_degree,
        [self.precision] * (local_degree+1),
        local_approx_interval,
        sollya.absolute)
      local_error = int(log2(sup(abs(local_error / acos(1 - local_approx_interval)))))
      coeff_table
      for d in range(local_degree):
        coeff_table[i][d] = sollya.coeff(local_poly_object.get_sollya_object(), d) 

    table_index = BitLogicRightShift(
        vx, vx.get_precision().get_field_size() - field_index_size
    ) - (exp_lo << field_index_size)




    print "building mathematical polynomial"
    poly_degree = sup(sollya.guessdegree(acos(x), approx_interval, S2**-(self.precision.get_field_size()))) 
    print "guessed polynomial degree: ", int(poly_degree)
    #global_poly_object = Polynomial.build_from_approximation(log10(1+x)/x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, absolute)

    print "generating polynomial evaluation scheme"
    #_poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)

    # building eval error map
    #eval_error_map = {
    #  red_vx: Variable("red_vx", precision = self.precision, interval = red_vx.get_interval()),
    #  log_inv_hi: Variable("log_inv_hi", precision = self.precision, interval = table_high_interval),
    #  log_inv_lo: Variable("log_inv_lo", precision = self.precision, interval = table_low_interval),
    #}
    # computing gappa error
    #poly_eval_error = self.get_eval_error(result, eval_error_map)



    # main scheme
    print "MDL scheme"
    scheme = Statement(Return(vx))
    return scheme



if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
    default_arg=ML_Acos.get_default_args())
  args = arg_template.arg_extraction()


  ml_acos          = ML_Acos(args)
  ml_acos.gen_implementation()
