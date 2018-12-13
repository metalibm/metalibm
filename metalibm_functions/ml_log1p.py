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

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, log1p,
    guessdegree
)
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.special_values import (
    FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_function import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.code_generation.generator_utility import FunctionOperator
from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value  
from metalibm_core.utility.ml_template import ML_NewArgTemplate, ArgDefault
from metalibm_core.utility.debug_utils import * 

class ML_Log1p(ML_Function("ml_log1p")):
  def __init__(self, args):
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log1p,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log1p = {
        "output_file": "my_log1p.c",
        "function_name": "my_log1pf",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log1p.update(kw)
    return DefaultArgTemplate(**default_args_log1p)

  def generate_scheme(self):
    vx = self.implementation.add_input_variable("x", self.precision) 
    sollya_precision = self.get_input_precision().sollya_object

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = vx
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)


    log2_hi_value = round(log(2), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
    log2_lo_value = round(log(2) - log2_hi_value, self.precision.sollya_object, sollya.RN)

    log2_hi = Constant(log2_hi_value, precision = self.precision)
    log2_lo = Constant(log2_lo_value, precision = self.precision)

    vx_exp  = ExponentExtraction(vx, tag = "vx_exp", debug = debugd)

    int_precision = self.precision.get_integer_format()

    # retrieving processor inverse approximation table
    dummy_var = Variable("dummy", precision = self.precision)
    dummy_div_seed = ReciprocalSeed(dummy_var, precision = self.precision)
    inv_approx_table = self.processor.get_recursive_implementation(dummy_div_seed, language = None, table_getter = lambda self: self.approx_table_map)

    # table creation
    table_index_size = 7
    log_table = ML_NewTable(dimensions = [2**table_index_size, 2], storage_precision = self.precision)
    log_table[0][0] = 0.0
    log_table[0][1] = 0.0
    for i in range(1, 2**table_index_size):
        #inv_value = (1.0 + (self.processor.inv_approx_table[i] / S2**9) + S2**-52) * S2**-1
        inv_value = inv_approx_table[i] # (1.0 + (inv_approx_table[i] / S2**9) ) * S2**-1
        value_high = round(log(inv_value), self.precision.get_field_size() - (self.precision.get_exponent_size() + 1), sollya.RN)
        value_low = round(log(inv_value) - value_high, sollya_precision, sollya.RN)
        log_table[i][0] = value_high
        log_table[i][1] = value_low


    vx_exp = ExponentExtraction(vx, tag = "vx_exp", debug = debugd)

    # case close to 0: ctz
    ctz_exp_limit = -7
    ctz_cond = vx_exp < ctz_exp_limit
    ctz_interval = Interval(-S2**ctz_exp_limit, S2**ctz_exp_limit)

    ctz_poly_degree = sup(guessdegree(log1p(sollya.x)/sollya.x, ctz_interval, S2**-(self.precision.get_field_size()+1))) + 1
    ctz_poly_object = Polynomial.build_from_approximation(log1p(sollya.x)/sollya.x, ctz_poly_degree, [self.precision]*(ctz_poly_degree+1), ctz_interval, sollya.absolute)

    Log.report(Log.Info, "generating polynomial evaluation scheme")
    ctz_poly = PolynomialSchemeEvaluator.generate_horner_scheme(ctz_poly_object, vx, unified_precision = self.precision)
    ctz_poly.set_attributes(tag = "ctz_poly", debug = debug_lftolx)

    ctz_result = vx * ctz_poly

    neg_input = Comparison(vx, -1, likely = False, specifier = Comparison.Less, debug = debugd, tag = "neg_input")
    vx_nan_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debugd, tag = "nan_or_inf")
    vx_snan = Test(vx, specifier = Test.IsSignalingNaN, likely = False, debug = debugd, tag = "snan")
    vx_inf  = Test(vx, specifier = Test.IsInfty, likely = False, debug = debugd, tag = "inf")
    vx_subnormal = Test(vx, specifier = Test.IsSubnormal, likely = False, debug = debugd, tag = "vx_subnormal")
    
    log_function_code = CodeFunction("new_log", [Variable("x", precision = ML_Binary64)], output_format = ML_Binary64) 
    log_call_generator = FunctionOperator(log_function_code.get_name(), arity = 1, output_precision = ML_Binary64, declare_prototype = log_function_code)
    newlog_function = FunctionObject(log_function_code.get_name(), (ML_Binary64,), ML_Binary64, log_call_generator)


    # case away from 0.0
    pre_vxp1 = vx + 1.0
    pre_vxp1.set_attributes(tag = "pre_vxp1", debug = debug_lftolx)
    pre_vxp1_exp = ExponentExtraction(pre_vxp1, tag = "pre_vxp1_exp", debug = debugd)
    cm500 = Constant(-500, precision = ML_Int32)
    c0 = Constant(0, precision = ML_Int32)
    cond_scaling = pre_vxp1_exp > 2**(self.precision.get_exponent_size()-2)
    scaling_factor_exp = Select(cond_scaling, cm500, c0)
    scaling_factor = ExponentInsertion(scaling_factor_exp, precision = self.precision, tag = "scaling_factor")

    vxp1 = pre_vxp1 * scaling_factor
    vxp1.set_attributes(tag = "vxp1", debug = debug_lftolx)
    vxp1_exp = ExponentExtraction(vxp1, tag = "vxp1_exp", debug = debugd)

    vxp1_inv = ReciprocalSeed(vxp1, precision = self.precision, tag = "vxp1_inv", debug = debug_lftolx, silent = True)

    vxp1_dirty_inv = ExponentInsertion(-vxp1_exp, precision = self.precision, tag = "vxp1_dirty_inv", debug = debug_lftolx)

    table_index = BitLogicAnd(BitLogicRightShift(TypeCast(vxp1, precision = int_precision, debug = debuglx), self.precision.get_field_size() - 7, debug = debuglx), 0x7f, tag = "table_index", debug = debuglx) 

    # argument reduction
    # TODO: detect if single operand inverse seed is supported by the targeted architecture
    pre_arg_red_index = TypeCast(BitLogicAnd(TypeCast(vxp1_inv, precision = ML_UInt64), Constant(-2, precision = ML_UInt64), precision = ML_UInt64), precision = self.precision, tag = "pre_arg_red_index", debug = debug_lftolx)
    arg_red_index = Select(Equal(table_index, 0), vxp1_dirty_inv, pre_arg_red_index, tag = "arg_red_index", debug = debug_lftolx)

    red_vxp1 = Select(cond_scaling, arg_red_index * vxp1 - 1.0, (arg_red_index * vx - 1.0) + arg_red_index)
    #red_vxp1 = arg_red_index * vxp1 - 1.0
    red_vxp1.set_attributes(tag = "red_vxp1", debug = debug_lftolx)

    log_inv_lo = TableLoad(log_table, table_index, 1, tag = "log_inv_lo", debug = debug_lftolx) 
    log_inv_hi = TableLoad(log_table, table_index, 0, tag = "log_inv_hi", debug = debug_lftolx)

    inv_err = S2**-6 # TODO: link to target DivisionSeed precision

    Log.report(Log.Info, "building mathematical polynomial")
    approx_interval = Interval(-inv_err, inv_err)
    poly_degree = sup(guessdegree(log(1+sollya.x)/sollya.x, approx_interval, S2**-(self.precision.get_field_size()+1))) + 1
    global_poly_object = Polynomial.build_from_approximation(log(1+sollya.x)/sollya.x, poly_degree, [self.precision]*(poly_degree+1), approx_interval, sollya.absolute)
    poly_object = global_poly_object.sub_poly(start_index = 1)

    Log.report(Log.Info, "generating polynomial evaluation scheme")
    _poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, red_vxp1, unified_precision = self.precision)
    _poly.set_attributes(tag = "poly", debug = debug_lftolx)
    Log.report(Log.Info, global_poly_object.get_sollya_object())


    vxp1_inv_exp = ExponentExtraction(vxp1_inv, tag = "vxp1_inv_exp", debug = debugd)
    corr_exp = Conversion(-vxp1_exp + scaling_factor_exp, precision = self.precision)# vxp1_inv_exp

    #poly = (red_vxp1) * (1 +  _poly)
    #poly.set_attributes(tag = "poly", debug = debug_lftolx, prevent_optimization = True)

    pre_result = -log_inv_hi + (red_vxp1 + red_vxp1 * _poly + (-corr_exp * log2_lo - log_inv_lo))
    pre_result.set_attributes(tag = "pre_result", debug = debug_lftolx)
    exact_log2_hi_exp = - corr_exp * log2_hi
    exact_log2_hi_exp.set_attributes(tag = "exact_log2_hi_exp", debug = debug_lftolx, prevent_optimization = True)
    #std_result =  exact_log2_hi_exp + pre_result

    exact_log2_lo_exp = - corr_exp * log2_lo
    exact_log2_lo_exp.set_attributes(tag = "exact_log2_lo_exp", debug = debug_lftolx)#, prevent_optimization = True)
    
    init = exact_log2_lo_exp  - log_inv_lo
    init.set_attributes(tag = "init", debug = debug_lftolx, prevent_optimization = True)
    fma0 = (red_vxp1 * _poly + init) # - log_inv_lo)
    fma0.set_attributes(tag = "fma0", debug = debug_lftolx)
    step0 = fma0 
    step0.set_attributes(tag = "step0", debug = debug_lftolx) #, prevent_optimization = True)
    step1 = step0 + red_vxp1
    step1.set_attributes(tag = "step1", debug = debug_lftolx, prevent_optimization = True)
    step2 = -log_inv_hi + step1
    step2.set_attributes(tag = "step2", debug = debug_lftolx, prevent_optimization = True)
    std_result = exact_log2_hi_exp + step2
    std_result.set_attributes(tag = "std_result", debug = debug_lftolx, prevent_optimization = True)


    # main scheme
    Log.report(Log.Info, "MDL scheme")
    pre_scheme = ConditionBlock(neg_input,
        Statement(
            ClearException(),
            Raise(ML_FPE_Invalid),
            Return(FP_QNaN(self.precision))
        ),
        ConditionBlock(vx_nan_or_inf,
            ConditionBlock(vx_inf,
                Statement(
                    ClearException(),
                    Return(FP_PlusInfty(self.precision)),
                ),
                Statement(
                    ClearException(),
                    ConditionBlock(vx_snan,
                        Raise(ML_FPE_Invalid)
                    ),
                    Return(FP_QNaN(self.precision))
                )
            ),
            ConditionBlock(vx_subnormal,
                Return(vx),
                ConditionBlock(ctz_cond,
                    Statement(
                        Return(ctz_result),
                    ),
                    Statement(
                        Return(std_result)
                    )
                )
            )
        )
    )
    scheme = pre_scheme
    return scheme

  def numeric_emulate(self, input_value):
    return log1p(input_value)



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=ML_Log1p.get_default_args())
    args = arg_template.arg_extraction()


    ml_log1p = ML_Log1p(args) 
    ml_log1p.gen_implementation()
