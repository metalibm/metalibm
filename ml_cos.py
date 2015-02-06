# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.generator_utility import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator


from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed


# lambda_msb is a function to determine the Median msb index, it takes an exponent as operand
# and returns a ML_Integer subtype value
def generate_payne_hanek(exact_constant, msb_index_dag, size, lsb, precision):
    cst_msb = int(ceil(log2(exact_constant)))
    p = precision.get_field_size()
    tmp_lsb = cst_msb - p + 1 

    array_size = int(ceil((lsb - cst_msb + 1) / p))
    cst_table = ML_Table(dimensions = [array_size, 0], storage_precision = precision)

    sollya_precision = precision.get_sollya_object()

    # building value table
    tmp_cst = exact_constant
    for i in xrange(0, array_size):
      new_cst = round(tmp_cst, sollya_precision, RN)
      tmp_cst = tmp_cst - new_cst
      cst_table[i][0] = tmp_cst

    init_diff = cst_msb - msb_index_dag

    msb_index = (init_diff) / p  
    nb_index = (size / p) + 1
    # number of valid bit in the first segment
    nvb = p - Modulo(init_diff, p)
    valid_mask = BitLogicRightShift(0xffffffff, 32 - nvb)
    Select(nvb == p, 0.0, TypeCast(BitLogic

    # building median retrieval
    med_msb = TableLoad(cst_table, msb_index, 0)
    BitLogicRightShift(0xffffff, p - Modulo(init_diff, p) 


def generate_payne_hanek(exact_cst_formula, precision, vx, n, k):
    cst_msb = int(ceil(log2(exact_constant)))
    p = precision.get_field_size()
    tmp_lsb = cst_msb - p + 1 

    array_size = int(ceil((lsb - cst_msb + 1) / p))
    cst_table = ML_Table(dimensions = [array_size, 0], storage_precision = precision)

    sollya_precision = precision.get_sollya_object()

    chunk_size = p

    # building value table
    tmp_cst = exact_constant
    for i in xrange(0, array_size):
      new_cst = round(tmp_cst, chunk_size, RN)
      tmp_cst = tmp_cst - new_cst
      cst_table[i][0] = tmp_cst

    e = ExponentExtraction(vx)
    msb_exp = - e + Constant(p - 1 - k)
    msb_index = (Constant(cst_msb) - msb_exp) / Constant(chunk_size)

    lsb_exp = - e + Constant(p - 1 - n)
    lsb_index = (Constant(cst_msb) - lsb_exp) / Constant(chunk_size)







class ML_Cosine:
    def __init__(self, 
                 precision = ML_Binary32, 
                 accuracy  = ML_Faithful,
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "cosf.c", 
                 function_name = "cosf"):

        # declaring target and instantiating optimization engine
        processor = target
        self.precision = precision
        opt_eng = OptimizationEngine(processor)
        gappacg = GappaCodeGenerator(processor, declare_cst = True, disable_debug = True)

        # declaring CodeFunction and retrieving input variable
        self.function_name = function_name
        exp_implementation = CodeFunction(self.function_name, output_format = self.precision)
        vx = Abs(exp_implementation.add_input_variable("x", self.precision), tag = "vx") 


        Log.report(Log.Info, "\033[33;1m generating implementation scheme \033[0m")
        if debug_flag: 
            Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

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

        sollya_precision = precision.get_sollya_object()
        hi_precision = precision.get_field_size() - 3

        # argument reduction
        frac_pi_index = 3
        frac_pi     = round(S2**frac_pi_index / pi, sollya_precision, RN)
        inv_frac_pi = round(pi / S2**frac_pi_index, hi_precision, RN)
        inv_frac_pi_lo = round(pi / S2**frac_pi_index - inv_frac_pi, sollya_precision, RN)
        # computing k = E(x * frac_pi)
        vx_pi = Multiplication(vx, frac_pi, precision = self.precision)
        k = NearestInteger(vx_pi, precision = ML_Int32, tag = "k", debug = True)
        fk = Conversion(k, precision = self.precision, tag = "fk")

        inv_frac_pi_cst    = Constant(inv_frac_pi, tag = "inv_frac_pi", precision = self.precision)
        inv_frac_pi_lo_cst = Constant(inv_frac_pi_lo, tag = "inv_frac_pi_lo", precision = self.precision)

        red_vx_hi = (vx - inv_frac_pi_cst * fk)
        red_vx_hi.set_attributes(tag = "red_vx_hi", debug = debug_precision, precision = self.precision)
        red_vx_lo_sub = inv_frac_pi_lo_cst * fk
        red_vx_lo_sub.set_attributes(tag = "red_vx_lo_sub", debug = debug_precision, unbreakable = True, precision = self.precision)
        pre_red_vx = red_vx_hi - inv_frac_pi_lo_cst * fk


        modk = Modulo(k, 2**(frac_pi_index+1), precision = ML_Int32, tag = "switch_value", debug = True)

        sel_c = Equal(BitLogicAnd(modk, 2**(frac_pi_index-1)), 2**(frac_pi_index-1))
        red_vx = Select(sel_c, -pre_red_vx, pre_red_vx)
        red_vx.set_attributes(tag = "red_vx", debug = debug_precision, precision = self.precision)

        approx_interval = Interval(-pi/(S2**(frac_pi_index+1)), pi / S2**(frac_pi_index+1))

        Log.report(Log.Info, "approx interval: %s\n" % approx_interval)

        error_goal_approx = S2**-self.precision.get_precision()


        Log.report(Log.Info, "\033[33;1m building mathematical polynomial \033[0m\n")
        poly_degree_vector = [None] * 2**(frac_pi_index+1)



        error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

        #polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_estrin_scheme
        polynomial_scheme_builder = PolynomialSchemeEvaluator.generate_horner_scheme

        index_relative = []

        poly_object_vector = [None] * 2**(frac_pi_index+1)
        for i in xrange(2**(frac_pi_index+1)):
          sub_func = cos(x+i*pi/S2**frac_pi_index)
          degree = int(sup(guessdegree(sub_func, approx_interval, error_goal_approx))) + 1

          degree_list = range(degree+1)
          a_interval = approx_interval
          if i == 0:
            # ad-hoc, TODO: to be cleaned
            degree = 6
            degree_list = range(0, degree+1, 2)
          elif i % 2**(frac_pi_index) == 2**(frac_pi_index-1):
            degree_list = range(1, degree+1, 2)

          poly_degree_vector[i] = degree 

          constraint = absolute
          delta = (2**(frac_pi_index - 3))
          centered_i = (i % 2**(frac_pi_index)) - 2**(frac_pi_index-1)
          if centered_i < delta and centered_i > -delta and centered_i != 0:
            constraint = relative
            index_relative.append(i)
          Log.report(Log.Info, "generating approximation for %d/%d" % (i, 2**(frac_pi_index+1)))
          poly_object_vector[i], _ = Polynomial.build_from_approximation_with_error(sub_func, degree_list, [binary32]*(degree+1), a_interval, constraint, error_function = error_function) 






        # unified power map for red_sx^n
        upm = {}
        rel_error_list = []

        poly_scheme_vector = [None] * (2**(frac_pi_index+1))

        for i in xrange(2**(frac_pi_index+1)):
          poly_object = poly_object_vector[i]
          poly_precision = self.precision
          if i == 3 or i == 5 or i == 7 or i == 9: 
              poly_precision = ML_Binary64
              c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = ML_Binary64)
              c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
              poly_hi = (c0 + c1 * red_vx)
              poly_hi.set_precision(ML_Binary64)
              poly_scheme = poly_hi + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)
          else:
            poly_scheme = polynomial_scheme_builder(poly_object, red_vx, unified_precision = poly_precision, power_map_ = upm)
          #if i == 3:
          #  c0 = Constant(coeff(poly_object.get_sollya_object(), 0), precision = self.precision)
          #  c1 = Constant(coeff(poly_object.get_sollya_object(), 1), precision = self.precision)
          #  poly_scheme = (c0 + c1 * red_vx) + polynomial_scheme_builder(poly_object.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)

          poly_scheme.set_attributes(tag = "poly_cos%dpi%d" % (i, 2**(frac_pi_index)), debug = debug_precision)
          poly_scheme_vector[i] = poly_scheme

          opt_scheme = opt_eng.optimization_process(poly_scheme, self.precision, copy = True, fuse_fma = fuse_fma)

          tag_map = {}
          opt_eng.register_nodes_by_tag(opt_scheme, tag_map)

          cg_eval_error_copy_map = {
              tag_map["red_vx"]: Variable("red_vx", precision = self.precision, interval = approx_interval),
          }


          #try:
          #if is_gappa_installed():
          #    eval_error = gappacg.get_eval_error_v2(opt_eng, opt_scheme, cg_eval_error_copy_map, gappa_filename = "red_arg_%d.g" % i)
          #    poly_range = cos(approx_interval+i*pi/S2**frac_pi_index)
          #    rel_error_list.append(eval_error / poly_range)


        #for rel_error in rel_error_list:
        #  print sup(abs(rel_error))

        #return 

        # case 17
        #poly17 = poly_object_vector[17]
        #c0 = Constant(coeff(poly17.get_sollya_object(), 0), precision = self.precision)
        #c1 = Constant(coeff(poly17.get_sollya_object(), 1), precision = self.precision)
        #poly_scheme_vector[17] = FusedMultiplyAdd(c1, red_vx, c0, specifier = FusedMultiplyAdd.Standard) + polynomial_scheme_builder(poly17.sub_poly(start_index = 2), red_vx, unified_precision = self.precision, power_map_ = upm)

        half = 2**frac_pi_index
        sub_half = 2**(frac_pi_index - 1)

        factor_cond = BitLogicXor(BitLogicRightShift(modk, frac_pi_index), BitLogicRightShift(modk, frac_pi_index-1))

        CM1 = Constant(-1, precision = self.precision)
        C1  = Constant(1, precision = self.precision)
        factor = Select(factor_cond, CM1, C1)
        factor2 = Select(Equal(modk, Constant(sub_half)), CM1, C1) 


        switch_map = {}
        if 0:
          for i in xrange(2**(frac_pi_index+1)):
            switch_map[i] = Return(poly_scheme_vector[i])
        else:
          for i in xrange(2**(frac_pi_index-1)):
            switch_case = (i, half - i)
            #switch_map[i]      = Return(poly_scheme_vector[i])
            #switch_map[half-i] = Return(-poly_scheme_vector[i])
            if i!= 0:
              switch_case = switch_case + (half+i, 2*half-i)
              #switch_map[half+i] = Return(-poly_scheme_vector[i])
              #switch_map[2*half-i] = Return(poly_scheme_vector[i])
            if poly_scheme_vector[i].get_precision() != self.precision:
              poly_result = Conversion(poly_scheme_vector[i], precision = self.precision)
            else:
              poly_result = poly_scheme_vector[i]
            switch_map[switch_case] = Return(factor*poly_result)
          #switch_map[sub_half] = Return(-poly_scheme_vector[sub_half])
          #switch_map[half + sub_half] = Return(poly_scheme_vector[sub_half])
          switch_map[(sub_half, half + sub_half)] = Return(factor2 * poly_scheme_vector[sub_half])



        

        result = SwitchBlock(modk, switch_map)


        # main scheme
        Log.report(Log.Info, "\033[33;1m MDL scheme \033[0m")
        scheme = Statement(result)


        # fusing FMA
        if fuse_fma: 
            Log.report(Log.Info, "\033[33;1m MDL fusing FMA \033[0m")
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        Log.report(Log.Info, "\033[33;1m MDL abstract scheme \033[0m")
        opt_eng.instantiate_abstract_precision(scheme, None)


        Log.report(Log.Info, "\033[33;1m MDL instantiated scheme \033[0m")
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)


        Log.report(Log.Info, "\033[33;1m subexpression sharing \033[0m")
        opt_eng.subexpression_sharing(scheme)

        Log.report(Log.Info, "\033[33;1m silencing operation \033[0m")
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(scheme)

        # check processor support
        Log.report(Log.Info, "\033[33;1m checking processor support \033[0m")
        opt_eng.check_processor_support(scheme)

        # factorizing fast path
        if fast_path_extract:
            Log.report(Log.Info, "\033[33;1m factorizing fast path\033[0m")
            opt_eng.factorize_fast_path(scheme)

        print "ml_cos DAG: "
        print scheme.get_str(depth = None, display_precision = True)
        
        Log.report(Log.Info, "\033[33;1m generating source code \033[0m")
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        #self.result.add_header("support_lib/ml_types.h")
        self.result.add_header("support_lib/ml_special_values.h")
        #display(decimal)
        #self.result.add_header_comment("polynomial degree  for  cos(x): %d" % poly_degree_cos)
        #self.result.add_header_comment("polynomial degree  for  sin(x): %d" % poly_degree_sin)
        #self.result.add_header_comment("sollya polynomial  for  cos(x): %s" % poly_object_cos.get_sollya_object())
        #self.result.add_header_comment("sollya polynomial  for  sin(x): %s" % poly_object_sin.get_sollya_object())
        #self.result.add_header_comment("polynomial approximation error cos: %s" % poly_approx_error_cos)
        #self.result.add_header_comment("polynomial approximation error sin: %s" % poly_approx_error_sin)
        #self.result.add_header_comment("polynomial evaluation    error: %s" % poly_eval_error)
        if debug_flag:
            self.result.add_header("stdio.h")
            self.result.add_header("inttypes.h")
        output_stream = open(output_file, "w")#"%s.c" % exp_implementation.get_name(), "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()



if __name__ == "__main__":
    # auto-test
    arg_template = ML_ArgTemplate(default_function_name = "new_cos", default_output_file = "new_cos.c" )
    # argument extraction 
    parse_arg_index_list = arg_template.sys_arg_extraction()
    arg_template.check_args(parse_arg_index_list)


    ml_exp          = ML_Cosine(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  function_name             = arg_template.function_name,
                                  accuracy                  = arg_template.accuracy,
                                  output_file               = arg_template.output_file)
