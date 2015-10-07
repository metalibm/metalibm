# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.debug_utils import * 

class ML_Log(ML_Function("ml_log")):
  def __init__(self, 
               precision = ML_Binary64, 
               abs_accuracy = S2**-24, 
               libm_compliant = True, 
               debug_flag = False, 
               fuse_fma = True, 
               fast_path_extract = True,
               target = GenericProcessor(), 
               output_file = "log_fixed.c", 
               function_name = "log_fixed"):
    # initializing I/O precision
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, 
      base_name = "log",
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

  def generate_emulate(self, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for ML_Log2 functions
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_log"
    emulate_func_op = FunctionOperator(emulate_func_name, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1)}, require_header = ["mpfr.h"]) 
    emulate_func   = FunctionObject(emulate_func_name, [ML_Mpfr_t, ML_Int32], ML_Mpfr_t, emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(result, emulate_func(mpfr_x, mpfr_rnd)))

    return mpfr_call



  # explore the parameters of the argument reduction
  # get the fastest code possible with some memory constraint :
  # for all possible parameters of the arg reg:
  # - get the final interval and the tables sizes proven by gappa
  # - eliminate the ones that desn't fits in the memory constraints
  # - get the smallest degree of the polynomial that achieve 2^-53 relative precision
  #   (or 2**-(self.precision.get_field_size()+1) depending on self.precision)
  # - get the smallest degree that achieve 2^-~128 absolute precision
  #   (TODO: get exact limit with worst cases. should be around 2^-114)
  # of all the parameters that achived thoses degrees, choose the one that have the smallest table size
  def generate_argument_reduction(self, vx, memory_bytes_limit):
    # TODO: make thoses parameters vary
    size1 = 7
    prec1 = 9
    size2 = 14
    prec2 = 15
    
    #vx = Variable("gappa_x", precision = ML_Custom_FixedPoint_Format(0, 52, False), interval = Interval(1,2-2**-52))
    vx.set_interval(Interval(1, 1.00001))
    
    #vx  = self.implementation.add_input_variable("gappa_x", ML_Custom_FixedPoint_Format(0,52,False))
    vx1 = Conversion(vx, precision = ML_Custom_FixedPoint_Format(0,size1,False))
    vinv_x1 = Division(Constant(1, precision = ML_Exact), vx1, precision = ML_Exact)
    vinv_x = Conversion(vinv_x1, precision = ML_Custom_FixedPoint_Format(0, prec1))
    vy = Multiplication(vx, vinv_x, precision = ML_Custom_FixedPoint_Format(0, 52+prec1))
    
    #self.precison = ML_Binary64
    #opt_expr = self.optimise_scheme(vy)
    
    annotation = Multiplication(vx1, vinv_x1, precision = ML_Exact)
    #print annotation.get_str(depth = True, display_precision = True)

    vx_me = Variable("me", interval = vx.get_interval(), precision = ML_Binary64)
    swap_map = {vx: vx_me }
    vy_goal = vy.copy(swap_map)

    annotation_hint = annotation.copy(swap_map)

    
    gappa_code = self.gappa_engine.get_interval_code_no_copy(vy_goal, bound_list = [vx_me])
    print "annotation: ", 
    print annotation.get_str(depth = None, display_precision = True)
    print "annotation_hint: ", 
    print annotation_hint.get_str(depth = None, display_precision = True)
    self.gappa_engine.add_hint(gappa_code, annotation_hint, Constant(1, precision = ML_Exact),
                               Comparison(swap_map[vinv_x1], Constant(0, precision = ML_Exact), specifier = Comparison.NotEqual, precision = ML_Bool))

    #print gappa_code.get_str()
    eval_error = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))
    print "eval error: ", eval_error
    return {
      'size1': 7, 'prec1': 9, 'size2': 14, 'prec2': 15,
      'prec_inv1': ML_Custom_FixedPoint_Format(1, 9, False),
      'prec_inv2': ML_Custom_FixedPoint_Format(1, 15, False),
      'tableSize1': 128,      'tableSize2': 192,
      'outinterval': Interval(0, 197375.0*2**-31)}


  def round_nearest_fixed(x, prec):
    return nearestint(x * 2**(-prec)) * 2**prec


  def generate_scheme(self):
    input_var = self.implementation.add_input_variable("input_var", self.precision) 

    sollya_precision = self.precision.sollya_object

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = input_var
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    # handling special cases
    test_nan_or_inf = Test(input_var, specifier = Test.IsInfOrNaN, likely = False, debug = True, tag = "nan_or_inf")
    test_nan =        Test(input_var, specifier = Test.IsNaN, debug = True, tag = "is_nan_test")
    test_positive   = Comparison(input_var, 0, specifier = Comparison.GreaterOrEqual, debug = True, tag = "inf_sign")
    test_signaling_nan = Test(input_var, specifier = Test.IsSignalingNaN, debug = True, tag = "is_signaling_nan")
    return_snan = Statement(ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

    # constants computation
    v_log2_hi = nearestint(log(2) * 2**-52) * 2**52
    v_log2_lo = round(log(2) - v_log2_hi, 64+53, RN)
    log2_hi = Constant(v_log2_hi, precision = self.precision, tag = "log2_hi")
    log2_lo = Constant(v_log2_lo, precision = self.precision, tag = "log2_lo")
   
    # compute the parameters for the argument reduction, given the specified constraints
    ve = ExponentExtraction(input_var, tag = "x_exponent", debug = debugd)
    vx = MantissaExtraction(input_var, tag = "x_mantissa", debug = debug_lftolx, precision = self.precision)
    arg_reduc = self.generate_argument_reduction(vx, 2560)

    # create the tables for the first argument reduction
    inv_table_1 = ML_Table(dimensions = [arg_reduc['tableSize1'], 1],
                           storage_precision = arg_reduc['prec_inv1'],
                           tag = self.uniquify_name("inv_table_1"))
    log_table_1 = ML_Table(dimensions = [arg_reduc['tableSize1'], 1],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("inv_table_1"))
    for i in xrange(0, arg_reduc['tableSize1']-1):
      x1 = 1 + i/2**arg_reduc['size1']
      inv_x1 = Division(Constant(1, precision = ML_Exact), x1,
                        precision = arg_reduc['prec_inv1'],
                        rounding_mode = ML_RoundTowardPlusInfty)
      log_x1 = Constant(log(x1), precision = ML_Custom_FixedPoint_Format(11, 128-11, False))
      inv_table_1[i][0] = inv_x1
      log_table_1[i][0] = log_x1

    # create the tables for the second argument reduction
    inv_table_2 = ML_Table(dimensions = [arg_reduc['tableSize2'], 1],
                           storage_precision = arg_reduc['prec_inv2'],
                           tag = self.uniquify_name("inv_table_1"))
    log_table_2 = ML_Table(dimensions = [arg_reduc['tableSize2'], 1],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("inv_table_2"))
    for i in xrange(0, arg_reduc['tableSize2']-1):
      x1 = 1 + i/2**arg_reduc['size2']
      inv_x1 = Division(Constant(1, precision = ML_Exact), x1,
                        precision = arg_reduc['prec_inv2'],
                        rounding_mode = ML_RoundTowardPlusInfty)
      log_x1 = Constant(log(x1), precision = ML_Custom_FixedPoint_Format(11, 128-11, False))
      inv_table_2[i][0] = inv_x1
      log_table_2[i][0] = log_x1

    # do the argument reduction
    
    _vx_mant = MantissaExtraction(input_var, tag = "_vx_mant", debug=debug_lftolx, precision = self.precision)
    _vx_exp =  ExponentExtraction(input_var, tag = "_vx_exp",  debug=debugd)

    print "First argument reduction: "
    _binary_mantissa = TypeCast(_vx_mant, precision = ML_UInt64, debug = debuglx)
    _vx_mantissa = TypeCast(BitLogicAnd(_binary_mantissa, Constant(1, precision = ML_UInt64)**52-1),
                            precision = ML_Custom_FixedPoint_Format(0,52,False))
    
    table_1_idx = TypeCast(Conversion(_vx_mantissa,
                                      precision = ML_Custom_FixedPoint_Format(0,arg_reduc['size1'],False)),
                           precision = ML_Integer)

    #table_1_idx = BitLogicRightShift(TypeCast(_vx_mantissa, precision = ML_UInt64), 52 - arg_reduc['size1'])
    print "index for first table: ", table_1_idx
    # TableLoad is of type FixedPoint(1,9,False)
    _red_vx = Multiplication(Addition(1,_vx_mantissa, precision = ML_Custom_FixedPoint_Format(1,52,False)),
                             TableLoad(inv_table_1, table_1_idx, 0),
                             tag = "_vy", debug=debug_lftolx, precision = ML_Custom_FixedPoint_Format(2, 52+9, False))
    _red_log = TableLoad(log_table_1, table_1_idx, 0, tag = "red_log_1");
    

    approx_interval = Interval(0, 27021597764222975*S2**-61)
    
    print "Second argument reduction: not yet implemented"
    print "Polynomials generation"
    poly_degree = 1+sup(guessdegree(log(1+x)/x, approx_interval, S2**-(self.precision.get_field_size())))
    print "degree required: ", poly_degree
    global_poly_object = Polynomial.build_from_approximation(log(1+x)/x, poly_degree, [1] + [self.precision]*(poly_degree), approx_interval, absolute)
    poly_object = global_poly_object.sub_poly(start_index = 1)
    print "generate polynomial evaluation scheme"
    _poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)
    _poly.set_attributes(tag = "poly", debug = debug_lftolx)
    print "generated polynomial: ", global_poly_object.get_sollya_object()

    print "pre result: "
    pre_result = _vx_exp * log(2) + _red_log + _poly * _red_vx
    vx_is_not_normal_positive = LogicalNot(Test(input_var, specifier = Test.IsIEEENormalPositive, likely = True, debug = debugd, tag = "is_special_cases"))
    return ConditionBlock(vx_is_not_normal_positive,
      Return(FP_QNaN(self.precision)),
      Return(pre_result)
    )
    """

    int_precision = ML_Int64
    integer_precision = ML_Int64
    def compute_log(_vx, exp_corr_factor = None):
        _vx_mant = MantissaExtraction(_vx, tag = "_vx_mant", debug = debug_lftolx, precision = self.precision)
        return exact_log2_hi_exp + pre_result, _poly, _log_inv_lo, _log_inv_hi, _red_vx, _result_one 

    result, poly, log_inv_lo, log_inv_hi, red_vx, new_result_one = compute_log(input_var)
    result.set_attributes(tag = "result", debug = debug_lftolx)
    new_result_one.set_attributes(tag = "new_result_one", debug = debug_lftolx)

    neg_input = Comparison(input_var, 0, likely = False, specifier = Comparison.Less, debug = debugd, tag = "neg_input")
    vx_nan_or_inf = Test(input_var, specifier = Test.IsInfOrNaN, likely = False, debug = debugd, tag = "nan_or_inf")
    vx_snan = Test(input_var, specifier = Test.IsSignalingNaN, likely = False, debug = debugd, tag = "snan")
    vx_inf  = Test(input_var, specifier = Test.IsInfty, likely = False, debug = debugd, tag = "inf")
    vx_subnormal = Test(input_var, specifier = Test.IsSubnormal, likely = False, debug = debugd, tag = "vx_subnormal")
    vx_zero = Test(input_var, specifier = Test.IsZero, likely = False, debug = debugd, tag = "vx_zero")

    exp_mone = Equal(vx_exp, -1, tag = "exp_minus_one", debug = debugd, likely = False)
    vx_one = Equal(vx, 1.0, tag = "vx_one", likely = False, debug = debugd)

    # exp=-1 case
    print "managing exp=-1 case"
    #red_vx_2 = arg_red_index * vx_mant * 0.5
    #approx_interval2 = Interval(0.5 - inv_err, 0.5 + inv_err)
    #poly_degree2 = sup(guessdegree(log(x), approx_interval2, S2**-(self.precision.get_field_size()+1))) + 1
    #poly_object2 = Polynomial.build_from_approximation(log(x), poly_degree, [self.precision]*(poly_degree+1), approx_interval2, absolute)
    #print "poly_object2: ", poly_object2.get_sollya_object()
    #poly2 = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object2, red_vx_2, unified_precision = self.precision)
    #poly2.set_attributes(tag = "poly2", debug = debug_lftolx)
    #result2 = (poly2 - log_inv_hi - log_inv_lo)

    result2 = (-log_inv_hi - log2_hi) + ((red_vx + poly * red_vx) - log2_lo - log_inv_lo)
    result2.set_attributes(tag = "result2", debug = debug_lftolx)

    m100 = -100
    S2100 = Constant(S2**100, precision = self.precision)
    result_subnormal, _, _, _, _, _ = compute_log(vx * S2100, exp_corr_factor = m100)

    print "managing close to 1.0 cases"
    one_err = S2**-7

    # main scheme
    print "MDL scheme"
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
                ConditionBlock(vx_zero, 
                    Statement(
                        ClearException(),
                        Raise(ML_FPE_DivideByZero),
                        Return(FP_MinusInfty(self.precision)),
                    ),
                    Return(result_subnormal)
                ),
                ConditionBlock(vx_one,
                    Statement(
                        ClearException(),
                        Return(FP_PlusZero(self.precision)),
                    ),
                    ConditionBlock(exp_mone,
                        Return(result2),
                        Return(result)
                    )
                    #ConditionBlock(cond_one,
                        #Return(new_result_one),
                        #ConditionBlock(exp_mone,
                            #Return(result2),
                            #Return(result)
                        #)
                    #)
                )
            )
        )
    )
    scheme = pre_scheme

    return scheme
    """

if __name__ == "__main__":
  # auto-test
  arg_template = ML_ArgTemplate(default_function_name = "new_log", default_output_file = "new_log.c" )
  arg_template.sys_arg_extraction()


  ml_log          = ML_Log(arg_template.precision, 
                                libm_compliant            = arg_template.libm_compliant, 
                                debug_flag                = arg_template.debug_flag, 
                                target                    = arg_template.target, 
                                fuse_fma                  = arg_template.fuse_fma, 
                                fast_path_extract         = arg_template.fast_path,
                                function_name             = arg_template.function_name,
                                output_file               = arg_template.output_file)

  ml_log.gen_implementation()
