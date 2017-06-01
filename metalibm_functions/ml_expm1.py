# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed



class ML_ExponentialM1_Red(ML_Function("ml_expm1")):
  def __init__(self, 
             arg_template = DefaultArgTemplate, 
             precision = ML_Binary32, 
             accuracy  = ML_Faithful,
             target = GenericProcessor(), 
             output_file = "my_expm1.c", 
             function_name = "my_expm1",
             ):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "expm1_red",
      function_name = function_name,
      output_file = output_file,

      io_precisions = io_precisions,

      processor = target,

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
    
    test_NaN_or_inf = Test(vx, specifier = Test.IsInfOrNaN, likely = False, debug = debug_multi, tag = "NaN_or_inf")
    test_NaN = Test(vx, specifier = Test.IsNaN, likely = False, debug = debug_multi, tag = "is_NaN")
    test_inf = Comparison(vx, 0, specifier = Comparison.Greater, debug = debug_multi, tag = "sign");
    
    #  Infnty input
    infty_return = Statement(ConditionBlock(test_inf, Return(FP_PlusInfty(self.precision)), Return(-1)))
    #  non-std input (inf/nan)
    specific_return = ConditionBlock(test_NaN, Return(FP_QNaN(self.precision)), infty_return)
    
    # Over/Underflow Tests
    
    precision_emax = self.precision.get_emax()
    precision_max_value = S2**(precision_emax + 1)
    expm1_overflow_bound = ceil(log(precision_max_value + 1))
    overflow_test = Comparison(vx, expm1_overflow_bound, likely = False, specifier = Comparison.Greater)
    overflow_return = Statement(Return(FP_PlusInfty(self.precision)))
    
    precision_emin = self.precision.get_emin_subnormal()
    precision_min_value = S2** precision_emin
    expm1_underflow_bound = floor(log(precision_min_value) + 1)
    underflow_test = Comparison(vx, expm1_underflow_bound, likely = False, specifier = Comparison.Less)
    underflow_return = Statement(Return(-1))
    
    sollya_prec_map = {ML_Binary32: sollya.binary32, ML_Binary64: sollya.binary64}
    
    # Constants

    log_2 = round(log(2), sollya_prec_map[self.precision], sollya.RN)
    invlog2 = round(1/log(2), sollya_prec_map[self.precision], sollya.RN)
    
    interval_vx = Interval(expm1_underflow_bound, expm1_overflow_bound)
    interval_fk = interval_vx * invlog2
    interval_k = Interval(floor(inf(interval_fk)), ceil(sup(interval_fk)))
    
    log2_hi_precision = self.precision.get_field_size() - (ceil(log2(sup(abs(interval_k)))) + 4)
    log2_hi = round(log(2), log2_hi_precision, sollya.RN)
    log2_lo = round(log(2) - log2_hi, sollya_prec_map[self.precision], sollya.RN)


    # Reduction
    
    unround_k = vx * invlog2
    k = NearestInteger(unround_k, precision = self.precision, tag = "k")
    ik = NearestInteger(unround_k, precision = ML_Int32, debug = debug_multi, tag = "ik")
    exact_pre_mul = (k * log2_hi)
    exact_pre_mul.set_attributes(exact = True)
    exact_hi_part = vx - exact_pre_mul
    exact_hi_part.set_attributes(exact = True, prevent_optimization = True)
    exact_lo_part = - k * log2_lo
    exact_lo_part.set_attributes(prevent_optimization = True)
    r = exact_hi_part + exact_lo_part
    r.set_attributes(tag = "r", debug = debug_multi)
    
    r_interval = Interval(-log_2/S2, log_2/S2)
    
    approx_interval = Interval(-log(2)/2, log(2)/2)
    
    opt_r = self.optimise_scheme(r, copy = {})
    #r = opt_r
    
    tag_map = {}
    self.opt_engine.register_nodes_by_tag(opt_r, tag_map)

    cg_eval_error_copy_map = {
        vx: Variable("x", precision = self.precision, interval = interval_vx),
        tag_map["k"]: Variable("k", interval = interval_k, precision = self.precision)
    }

    #try:
    if is_gappa_installed():
        #eval_error = gappacg.get_eval_error(opt_r, cg_eval_error_copy_map, gappa_filename = "red_arg.g")
        eval_error = self.gappa_engine.get_eval_error_v2(self.opt_engine, opt_r, cg_eval_error_copy_map, gappa_filename = "red_arg.g")
    else:
        eval_error = 0.0
        Log.report(Log.Warning, "gappa is not installed in this environnement")
    Log.report(Log.Info, "eval error: %s" % eval_error)
    #except:
    #    Log.report(Log.Info, "gappa error evaluation failed")
    
    
    local_ulp = sup(ulp(exp(r_interval), self.precision))
    
    print "ulp: ", local_ulp 
    error_goal = S2**-1*local_ulp
    print "error goal: ", error_goal 
    
    
    # Polynomial Approx
    
    Log.report(Log.Info, "\033[33;1m Building polynomial \033[0m\n")
    
    poly_degree = max(sup(guessdegree(expm1(sollya.x), r_interval, error_goal)), 2)
    
    polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
    
    Log.report(Log.Info, "Degree : %d" % poly_degree)
    precision_list = [self.precision] *(poly_degree + 1)
    poly_object = Polynomial.build_from_approximation(expm1(sollya.x), poly_degree, precision_list, r_interval, sollya.absolute)
    sub_poly = poly_object.sub_poly(start_index = 2)
    Log.report(Log.Info, "Poly : %s" % sub_poly)
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
    
    test_subnormal = Test(late_underflow_result, specifier = Test.IsSubnormal)
    
    late_underflow_return = Statement(
        ConditionBlock(
            test_subnormal, 
            ExpRaiseReturn(ML_FPE_Underflow, return_value = late_underflow_result)), 
            Return(late_underflow_result)
            )
    
    # Reconstruction
    
    std_result = exp_k * ( poly + diff )
    
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

  standard_test_cases = [(sollya.parse("-0x1.0783eep+6"),)]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_function_name = "new_expm1_red", default_output_file = "new_expm1_red.c" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()
 
    ml_expm1_red         = ML_ExponentialM1_Red(args)

    ml_expm1_red.gen_implementation()
