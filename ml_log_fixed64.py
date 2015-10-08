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
from metalibm_core.code_generation.fixed_point_backend import FixedPointBackend

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.debug_utils import * 

from numpy import ndindex as ndrange # multidimentional range to iterate over

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
  """ return the size of the tables used by the argument reduction,
      and the interval of the output variable (and some other infos about the argument reduction= """ 
  def evaluate_argument_reduction(self, x, size1, prec1, size2, prec2):
    one = Constant(1, precision = ML_Exact, tag = "one")
    
    arg_reduc = {
      # parameters
      'size1': size1, 'prec1': prec1, 'size2': size2, 'prec2': prec2,
      # utilities
      'prec_inv1': ML_Custom_FixedPoint_Format(1, prec1, False),
      'prec_inv2': ML_Custom_FixedPoint_Format(1, prec2, False),
      # results
      'size_table1': None, 'size_table2': None, 'size_tables_byte': None,
      'midinterval': None, 'outinterval': None,
    }

    # do the argument reduction
    x1 =    Conversion(x, tag = "x1",
                       precision = ML_Custom_FixedPoint_Format(0,size1,False),
                       rounding_mode = ML_RoundTowardMinusInfty)
    s = Multiplication(Subtraction(x1, one, precision = ML_Exact),
                       Constant(S2**size1, precision = ML_Exact),
                       precision = ML_Exact,
                       tag = "indexTableX")
    inv_x1 =  Division(one, x1, tag = "ix1",
                       precision = ML_Exact)
    inv_x = Conversion(inv_x1,  tag = "ix",
                       precision = arg_reduc['prec_inv1'],
                       rounding_mode = ML_RoundTowardPlusInfty)
    y = Multiplication(x, inv_x, tag = "y",
                       precision = ML_Exact)
    dy =   Subtraction(y, one,  tag = "dy", 
                       precision = ML_Exact)
    y1 =    Conversion(y, tag = "y",
                       precision = ML_Custom_FixedPoint_Format(0,size2,False),
                       rounding_mode = ML_RoundTowardMinusInfty)
    t = Multiplication(Subtraction(y1, one, precision = ML_Exact),
                       Constant(S2**size2, precision = ML_Exact),
                       precision = ML_Exact,
                       tag = "indexTableY")
    inv_y1 =  Division(one, y1, tag = "iy1",
                       precision = ML_Exact)
    inv_y = Conversion(inv_y1, tag = "iy",
                       precision = arg_reduc['prec_inv2'],
                       rounding_mode = ML_RoundTowardPlusInfty)
    z = Multiplication(y, inv_y, tag = "z",
                       precision = ML_Exact)
    dz =   Subtraction(z, one, tag = "dz",
                       precision = ML_Exact)

    # add the necessary goals and hints
    x_gappa = Variable("x_gappa", interval = Interval(1, 2-S2**-52), precision = ML_Binary64)
    swap_map = {x: x_gappa}

    # goal: dz (result of the argument reduction)
    gappa_code = self.gappa_engine.get_interval_code_no_copy(dz.copy(swap_map), bound_list = [x_gappa])
    self.gappa_engine.add_goal(gappa_code, dy.copy(swap_map))
    self.gappa_engine.add_goal(gappa_code, s.copy(swap_map)) # range of index of table 1
    self.gappa_engine.add_goal(gappa_code, t.copy(swap_map)) # range of index of table 2
    # hints. are the ones with isAppox=True really necessary ?
    self.gappa_engine.add_hint(gappa_code, x.copy(swap_map), x1.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code, y.copy(swap_map), y1.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code, inv_x1.copy(swap_map), inv_x.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code, inv_y1.copy(swap_map), inv_y.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code,
                               Multiplication(x1, inv_x1, precision = ML_Exact).copy(swap_map), one,
                               Comparison(swap_map[inv_x1], Constant(0, precision = ML_Exact),
                                          specifier = Comparison.NotEqual, precision = ML_Bool))
    self.gappa_engine.add_hint(gappa_code,
                               Multiplication(y1, inv_y1, precision = ML_Exact).copy(swap_map), one,
                               Comparison(swap_map[inv_y1], Constant(0, precision = ML_Exact),
                                          specifier = Comparison.NotEqual, precision = ML_Bool))
    # execute and parse the result
    result = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))
    arg_reduc['size_table1'] = 1+sup(result['indexTableX']).getConstantAsInt()
    arg_reduc['size_table2'] = 1+sup(result['indexTableY']).getConstantAsInt()
    arg_reduc['midinterval'] = result['dy']
    arg_reduc['outinterval'] = result['goal']
    arg_reduc['size_tables_byte'] = arg_reduc['size_table1'] * (16 + arg_reduc['prec_inv1'].get_c_bit_size()/8) + arg_reduc['size_table2'] * (16 + arg_reduc['prec_inv2'].get_c_bit_size()/8)
    return arg_reduc

  def generate_argument_reduction(self, x, memory_limit):
    possible_arg_reducs = []
    best_arg_reduc = None
    # iterate through all possible values, and keep only the valid argument reduction
    # uses some usefull properties to reduce the space of parameters:
    #   size1 <= log2(memory_limit) - 4
    #   prec1 < -13 + 
    #   midinterval <= 2**-size1 + 2**(1-prec1)
    #memory_used >= 16 * size_table_1 == 16 * 2**size1. so size1 <= log2(memory_limit) - 4
    #  - prec2 don't influence the table sizes, and increasing it improve the argument reduction
    #    so it is the max such that sup(outinterval) <= 2**(64-52-prec1-prec2)
    #    but outinterval can't be mush smaller than 2**-(size2 + 1)
    #    so: 2**-(size2+2) <= 2**(64-52-prec1-prec2)

    max_size1 = floor(log(S2 * memory_limit)/log(S2) - 5).getConstantAsInt()
    print "===== size1 <=", max_size1, " =====", memory_limit, (log(S2 * memory_limit)/log(S2) - 5)
    for size1 in range(2,max_size1):
      max_prec1 = ceil(log((1 + S2**12) * (1 - S2**-(size1+1)))/log(S2)).getConstantAsInt() - 1
      print "===== prec1 <=", max_prec1, '=====', (1 - S2**-(size1+1)).getConstantAsDouble()
      for prec1 in range(size1,10):
        tmpres = self.evaluate_argument_reduction(x, size1, prec1, prec1, prec1)
        
        doBreak_size2 = False
        for size2 in range(prec1,15):
          if doBreak_size2:
            break
          for prec2 in range(size2,15):
            arg_reduc = self.evaluate_argument_reduction(x, size1, prec1, size2, prec2)

            # check the memory used
            if arg_reduc['size_tables_byte'] > memory_limit:
              doBreak_size2 = True
              break

            # check if the result of the first arg_red goes into a uint64_t
            midinterval = arg_reduc['midinterval']
            max_on_u64 = S2**-(52 - 64 + arg_reduc['prec_inv1'].get_frac_size())
            #print "Implicit bits 1:", (-1 - floor(log(sup(midinterval))/log(2)).getConstantAsInt())
            #print "Res Interval 1: {0} < {1} == 2^64*2^-(52+{2}+{3})".format(
            #  sup(midinterval), max_on_u64, arg_reduc['prec_inv1'].get_frac_size(), arg_reduc['prec_inv2'].get_frac_size())
            #assert(0 <= inf(midinterval) and sup(midinterval) < max_on_u64)
            if not(0 <= inf(midinterval) and sup(midinterval) <= max_on_u64):
              continue 

            # check if the result of the two arg_red goes into a uint64_t
            outinterval = arg_reduc['outinterval']
            max_on_u64 = S2**-(52 - 64 + arg_reduc['prec_inv1'].get_frac_size() + arg_reduc['prec_inv2'].get_frac_size())
            #print "Implicit bits 2:", (-1 - floor(log(sup(outinterval))/log(2)).getConstantAsInt())
            #print "Res Interval 2: {0} < {1} == 2^64*2^-(52+{2}+{3})".format(
            #  sup(outinterval), max_on_u64, arg_reduc['prec_inv1'].get_frac_size(), arg_reduc['prec_inv2'].get_frac_size())
            #assert(0 <= inf(outinterval) and sup(outinterval) < max_on_u64)
            if not(0 <= inf(outinterval) and sup(outinterval) < max_on_u64):
              continue
            
            print "Potentially valid argument reduction: ", size1, prec1, size2, prec2
            #print "  memory: ", size_in_bytes
            #print "  midinterval: ", sup(midinterval).getConstantAsDouble()
            #print "  outinterval: ", sup(outinterval).getConstantAsDouble()
            possible_arg_reducs.append(arg_reduc)
            if (best_arg_reduc is None) or sup(outinterval).getConstantAsDouble() > sup(best_arg_reduc['outinterval']).getConstantAsDouble():
              best_arg_reduc = arg_reduc
    print "\n\nBest arg reduc: \n", arg_reduc, "\n"
    
    return best_arg_reduc
    
  def round_nearest_fixed(x, prec):
    return nearestint(x * 2**(-prec)) * 2**prec


  def generate_scheme(self):
    input_var = self.implementation.add_input_variable("input_var", self.precision) 
    memory_limit = 4000
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
    arg_reduc = self.generate_argument_reduction(vx, memory_limit)

    # create the tables for the first argument reduction
    inv_table_1 = ML_Table(dimensions = [arg_reduc['size_table1'], 1],
                           storage_precision = arg_reduc['prec_inv1'],
                           tag = self.uniquify_name("inv_table_1"))
    log_table_1 = ML_Table(dimensions = [arg_reduc['size_table1'], 1],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("inv_table_1"))
    for i in xrange(0, arg_reduc['size_table1']-1):
      x1 = 1 + i/2**arg_reduc['size1']
      inv_x1 = Division(Constant(1, precision = ML_Exact), x1,
                        precision = arg_reduc['prec_inv1'],
                        rounding_mode = ML_RoundTowardPlusInfty)
      log_x1 = Constant(log(x1), precision = ML_Custom_FixedPoint_Format(11, 128-11, False))
      inv_table_1[i][0] = inv_x1
      log_table_1[i][0] = log_x1

    # create the tables for the second argument reduction
    inv_table_2 = ML_Table(dimensions = [arg_reduc['size_table2'], 1],
                           storage_precision = arg_reduc['prec_inv2'],
                           tag = self.uniquify_name("inv_table_1"))
    log_table_2 = ML_Table(dimensions = [arg_reduc['size_table2'], 1],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("inv_table_2"))
    for i in xrange(0, arg_reduc['size_table2']-1):
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
