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
import resource

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


  """ evaluate one argument reduction (Tang):
      given:
        an input variable of type Fixed(0,k,False), and with some input interval
        the number of bits to read from this variable for the argument reduction
        the precision of its inverse
      it returns:
        out_interval: the output interval of the variable
        length_table: the number of elements in the table
        sizeof_table: the size in byte of the table used
  """
  def evaluate_argument_reduction(self, in_interval, in_prec, inv_size, inv_prec):
    one = Constant(1, precision = ML_Exact, tag = "one")
    
    dx =     Variable("dx",
                      precision = ML_Custom_FixedPoint_Format(0, in_prec, False),
                      interval = in_interval)
    
    # do the argument reduction
    x =       Addition(dx, one, tag = "x",
                       precision = ML_Exact)
    x1 =    Conversion(x, tag = "x1",
                       precision = ML_Custom_FixedPoint_Format(0, inv_size, False),
                       rounding_mode = ML_RoundTowardMinusInfty)
    s = Multiplication(dx, Constant(S2**inv_size, precision = ML_Exact),
                       precision = ML_Exact,
                       tag = "interval_index_table")
    inv_x1 =  Division(one, x1, tag = "ix1",
                       precision = ML_Exact)
    inv_x = Conversion(inv_x1,  tag = "ix",
                       precision = ML_Custom_FixedPoint_Format(1, inv_prec, False),
                       rounding_mode = ML_RoundTowardPlusInfty)
    y = Multiplication(x, inv_x, tag = "y",
                       precision = ML_Exact)
    dy =   Subtraction(y, one,  tag = "dy", 
                       precision = ML_Exact)
    
    # add the necessary goals and hints
    dx_gappa = Variable("dx_gappa", interval = dx.get_interval(), precision = dx.get_precision())
    swap_map = {dx: dx_gappa}

    # goal: dz (result of the argument reduction)
    gappa_code = self.gappa_engine.get_interval_code_no_copy(dy.copy(swap_map), bound_list = [swap_map[dx]])
    #self.gappa_engine.add_goal(gappa_code, s.copy(swap_map)) # range of index of table
    # hints. are the ones with isAppox=True really necessary ?
    self.gappa_engine.add_hint(gappa_code, x.copy(swap_map), x1.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code, inv_x1.copy(swap_map), inv_x.copy(swap_map), isApprox = True)
    self.gappa_engine.add_hint(gappa_code,
                               Multiplication(x1, inv_x1, precision = ML_Exact).copy(swap_map), one,
                               Comparison(swap_map[inv_x1], Constant(0, precision = ML_Exact),
                                          specifier = Comparison.NotEqual, precision = ML_Bool))
    # execute and parse the result
    result = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))
    out_interval = result['goal']
    length_table = 1 + floor(sup(in_interval) * S2**inv_size).getConstantAsInt()
    sizeof_table = length_table * (16 + ML_Custom_FixedPoint_Format(1, inv_prec, False).get_c_bit_size()/8)
    return {
      'out_interval': out_interval,
      'length_table': length_table,
      'sizeof_table': sizeof_table,
    }

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
  def eval_argument_reduction(self, size1, prec1, size2, prec2):
    one = Constant(1, precision = ML_Exact, tag = "one")
    dx =     Variable("dx",
                      precision = ML_Custom_FixedPoint_Format(0, 52, False),
                      interval = Interval(0, 1 - S2**-52))

    # do the argument reduction
    x =       Addition(dx, one, tag = "x",
                       precision = ML_Exact)
    x1 =    Conversion(x, tag = "x1",
                       precision = ML_Custom_FixedPoint_Format(0, size1, False),
                       rounding_mode = ML_RoundTowardMinusInfty)
    s = Multiplication(Subtraction(x1, one, precision = ML_Exact),
                       Constant(S2**size1, precision = ML_Exact),
                       precision = ML_Exact,
                       tag = "indexTableX")
    inv_x1 =  Division(one, x1, tag = "ix1",
                       precision = ML_Exact)
    inv_x = Conversion(inv_x1,  tag = "ix",
                       precision = ML_Custom_FixedPoint_Format(1, prec1, False),
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
                       precision = ML_Custom_FixedPoint_Format(1,prec2,False),
                       rounding_mode = ML_RoundTowardPlusInfty)
    z = Multiplication(y, inv_y, tag = "z",
                       precision = ML_Exact)
    dz =   Subtraction(z, one, tag = "dz",
                       precision = ML_Exact)


    # add the necessary goals and hints
    dx_gappa = Variable("dx_gappa", interval = dx.get_interval(), precision = dx.get_precision())
    swap_map = {dx: dx_gappa}
    # goals (main goal: dz, the result of the argument reduction)
    gappa_code = self.gappa_engine.get_interval_code_no_copy(dz.copy(swap_map), bound_list = [dx_gappa])
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
    toto = Variable("toto", precision = ML_Binary64)
    self.gappa_engine.add_hypothesis(gappa_code, toto, Interval(0, S2**-52))
    
    # execute and parse the result
    result = execute_gappa_script_extract(gappa_code.get(self.gappa_engine))
    self.gappa_engine.clear_memoization_map() # avoid memory leak
    #print result['indexTableX'], result['indexTableY']
    length_table1 = 1 + floor(sup(result['indexTableX'])).getConstantAsInt()
    length_table2 = 1 + floor(sup(result['indexTableY'])).getConstantAsInt()
    if False and (length_table2 != 1 + floor(sup(result['dy']) * S2**size2).getConstantAsInt()):
      print "(dy*2**size2:", 1 + floor(sup(result['dy']*S2**size2)).getConstantAsInt(), ")"
      print "(indexTableY:", 1 + floor(sup(result['indexTableY'])).getConstantAsInt(), ")"
      print result['indexTableY'], result['dy']
      sys.exit(1)
    return {
      # arguments
      'size1': size1, 'prec1': prec1, 'size2': size2, 'prec2': prec2,
      # size of the tables
      'length_table1': length_table1,
      'length_table2': length_table2,
      'sizeof_table1': length_table1 * (16 + ML_Custom_FixedPoint_Format(1,prec1,False).get_c_bit_size()/8),
      'sizeof_table2': length_table2 * (16 + ML_Custom_FixedPoint_Format(1,prec2,False).get_c_bit_size()/8),
      # intervals
      'in_interval': dx.get_interval(),
      'mid_interval': result['dy'],
      'out_interval': result['goal'],
    }

  def generate_argument_reduction(self, memory_limit):
    best_arg_reduc = None

    best_arg_reduc = self.eval_argument_reduction(6,10,12,13)
    best_arg_reduc['sizeof_tables'] = best_arg_reduc['sizeof_table1'] + best_arg_reduc['sizeof_table2']
    best_arg_reduc['degree_poly1'] = 4
    best_arg_reduc['degree_poly2'] = 8
    return best_arg_reduc
    # iterate through all possible parameters, and return the best argument reduction
    # the order of importance of the caracteristics of a good argument reduction is:
    #   1- the argument reduction is valid
    #   2- the degree of the polynomials obtains are minimals
    #   3- the memory used is minimal
    # An arument reduction is valid iff:
    #   - the memory used is less than memory_limit
    #   - y-1 and z-1  fit into a uint64_t
    #   - the second argument reduction should usefull (ie: it should add at least 1 bit to the argument reduction)
    # From thoses validity constraint we deduce some bound on the parameters to reduce the space of value searched:
    # (note that thoses bound are implied by, but not equivalents to the constraints)
    #   size1 <= log2(memory_limit/17)                                       (memory_limit on the first table)
    #   prec1 < 13 + size1                                                   (y-1 fits into a uint64_t)
    #   size2 <= log2((memory_limit - sizeof_table1)/17/midinterval)          (memory_limit on both tables)
    #   size2 >= 1 - log2(midinterval)                                       (second arg red should be usefull)
    #   prec2 < 12 - prec1 - log2((y-y1)/y1),  for all possible y            (z-1 fits into a uint64_t)
    # note: it is hard to deduce a tight bound on prec2 from the last inequality
    # a good approximation is  size2 ~= max[for y]( - log2((y-y1)/y1)), but using it may eliminate valid arg reduc

    #self.eval_argument_reduction(12, 20, 22, 14)

    min_size1 = 1
    max_size1 = floor(log(memory_limit/17)/log(2)).getConstantAsInt()
    for size1 in xrange(max_size1, min_size1-1, -1):
      
      min_prec1 = size1
      max_prec1 = 12 + size1
      for prec1 in xrange(min_prec1,max_prec1+1):
        
        # we need sizeof_table1 and mid_interval for the bound on size2 and prec2
        first_arg_reduc = self.eval_argument_reduction(size1, prec1, prec1, prec1)
        mid_interval = first_arg_reduc['mid_interval']
        sizeof_table1 = first_arg_reduc['sizeof_table1']

        if not(0 <= inf(mid_interval) and sup(mid_interval) < S2**(64 - 52 - prec1)):
          continue
        if not(first_arg_reduc['sizeof_table1'] < memory_limit):
          continue
        
        min_size2 = 1 - ceil(log(sup(mid_interval))/log(2)).getConstantAsInt()
        max_size2 = floor(log((memory_limit - sizeof_table1)/(17 * sup(mid_interval)))/log(2)).getConstantAsInt()
        # during execution of the prec2 loop, it can reduces the interval of valid values for prec2
        # so min_prec2 and max_prec2 are setted here and not before the the prec2 loop
        # (because they are modified inside the body of the loop, for the next iteration of size2)
        min_prec2 = 0
        max_prec2 = 12 + max_size2 - prec1
        for size2 in xrange(max_size2,min_size2-1,-1):
          
          max_prec2 = min(max_prec2, 12 + size2 - prec1)
          for prec2 in xrange(max_prec2,min_prec2-1,-1):
            
            #print '=====\t\033[1m{}\033[0m({}/{}),\t\033[1m{}\033[0m({}/{}),\t\033[1m{}\033[0m({}/{}),\t\033[1m{}\033[0m({}/{})\t====='.format(size1,min_size1,max_size1,prec1,min_prec1,max_prec1,size2,min_size2,max_size2,prec2,min_prec2,max_prec2)
            #print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss #memory used by the programm

            arg_reduc = self.eval_argument_reduction(size1, prec1, size2, prec2)
            mid_interval = arg_reduc['mid_interval']
            out_interval = arg_reduc['out_interval']
            sizeof_tables = arg_reduc['sizeof_table1'] + arg_reduc['sizeof_table2']
            if not(0 <= inf(out_interval) and sup(out_interval) < S2**(64-52-prec1-prec2)):
              max_prec2 = prec2 - 1
              continue
            if memory_limit < sizeof_tables:
              continue
            #assert(prec2 < 12 + size2 - prec1) # test the approximation size2 ~= max[for y]( - log2((y-y1)/y1))

            # guess the degree of the two polynomials (relative error <= 2^-52 and absolute error <= 2^-120)
            # note: we exclude zero from out_interval to not perturb sollya (log(1+x)/x is not well defined on 0)
            sollya_out_interval = Interval(S2**(-52-prec1-prec2), sup(out_interval))
            guess_degree_poly1 = guessdegree(log(1+x)/x, sollya_out_interval, S2**-52)
            guess_degree_poly2 = guessdegree(log(1+x), sollya_out_interval, S2**-120)
            # TODO: detect when guessdegree return multiple possible degree, and find the right one
            if False and inf(guess_degree_poly1) <> sup(guess_degree_poly1):
              print "improvable guess_degree_poly1:", guess_degree_poly1
            if False and inf(guess_degree_poly2) <> sup(guess_degree_poly2):
              print "improvable guess_degree_poly2:", guess_degree_poly2
            degree_poly1 = sup(guess_degree_poly1).getConstantAsInt() + 1
            degree_poly2 = sup(guess_degree_poly2).getConstantAsInt()
            
            if ((best_arg_reduc is not None)
            and (best_arg_reduc['degree_poly1'] < degree_poly1 or best_arg_reduc['degree_poly2'] < degree_poly2)):
              min_prec2 = prec2 + 1
              break

            if ((best_arg_reduc is None)
             or (best_arg_reduc['degree_poly1'] > degree_poly1)
             or (best_arg_reduc['degree_poly1'] == degree_poly1 and best_arg_reduc['degree_poly2'] > degree_poly2)
             or (best_arg_reduc['degree_poly1'] == degree_poly1 and best_arg_reduc['degree_poly2'] == degree_poly2 and best_arg_reduc['sizeof_tables'] > sizeof_tables)):
              arg_reduc['degree_poly1'] = degree_poly1
              arg_reduc['degree_poly2'] = degree_poly2
              arg_reduc['sizeof_tables'] = sizeof_tables
              best_arg_reduc = arg_reduc
              #print "\n   --new best--  \n", arg_reduc, "\n"
    #print "\nBest arg reduc: \n", best_arg_reduc, "\n"
    return best_arg_reduc
    

  def generate_scheme(self):
    memory_limit = 2500

    # local overloading of RaiseReturn operation
    def ExpRaiseReturn(*args, **kwords):
        kwords["arg_value"] = input_var
        kwords["function_name"] = self.function_name
        return RaiseReturn(*args, **kwords)

    ### Constants computations ###

    v_log2_hi = nearestint(log(2) * 2**-52) * 2**52
    v_log2_lo = round(log(2) - v_log2_hi, 64+53, RN)
    log2_hi = Constant(v_log2_hi, precision = self.precision, tag = "log2_hi")
    log2_lo = Constant(v_log2_lo, precision = self.precision, tag = "log2_lo")
   
    print "\n\033[1mSearch parameters for the argument reduction:\033[0m (this can take a while)"
    arg_reduc = self.generate_argument_reduction(memory_limit)

    print "\n\033[1mArgument reduction found:\033[0m [({},{}),({},{})] -> polynomials of degree {},{}, using {} bytes of memory".format(arg_reduc['size1'],arg_reduc['prec1'],arg_reduc['size2'],arg_reduc['prec2'],arg_reduc['degree_poly1'],arg_reduc['degree_poly2'],arg_reduc['sizeof_tables']) 
    
    print "\n\033[1mGenerate the first logarithm table:\033[0m containing {} elements, using {} bytes of memory".format(arg_reduc['length_table1'], arg_reduc['sizeof_table1'])
    inv_table_1 = ML_Table(dimensions = [arg_reduc['length_table1']],
                           storage_precision = ML_Custom_FixedPoint_Format(1, arg_reduc['prec1'], False),
                           tag = self.uniquify_name("inv_table_1"))
    log_table_1 = ML_Table(dimensions = [arg_reduc['length_table1']],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("log_table_1"))
    for i in xrange(0, arg_reduc['length_table1']-1):
      x1 = 1 + i/S2*arg_reduc['size1']
      inv_x1 = ceil(S2**arg_reduc['prec1']/x1)*S2**arg_reduc['prec1']
      log_x1 = floor(log(x1) * S2**(128-11))*S2**(11-128)
      inv_table_1[i] = inv_x1 #Constant(inv_x1, precision = ML_Custom_FixedPoint_Format(1, arg_reduc['prec1'], False))
      log_table_1[i] = log_x1 #Constant(log_x1, precision = ML_Custom_FixedPoint_Format(11, 128-11, False))

    print "\n\033[1mGenerate the second logarithm table:\033[0m containing {} elements, using {} bytes of memory".format(arg_reduc['length_table2'], arg_reduc['sizeof_table2'])
    inv_table_2 = ML_Table(dimensions = [arg_reduc['length_table2']],
                           storage_precision = ML_Custom_FixedPoint_Format(1, arg_reduc['prec2'], False),
                           tag = self.uniquify_name("inv_table_2"))
    log_table_2 = ML_Table(dimensions = [arg_reduc['length_table2']],
                           storage_precision = ML_Custom_FixedPoint_Format(11, 128-11, False),
                           tag = self.uniquify_name("log_table_2"))
    for i in xrange(0, arg_reduc['length_table2']-1):
      y1 = 1 + i/S2**arg_reduc['size2']
      inv_y1 = ceil(S2**arg_reduc['prec2']/x1) * S2**arg_reduc['prec2']
      log_y1 = floor(log(inv_y1) * S2**(128-11))*S2**(11-128)
      inv_table_2[i] = inv_y1 #Constant(inv_y1, precision = ML_Custom_FixedPoint_Format(1, arg_reduc['prec2'], False))
      log_table_2[i] = log_y1 #Constant(log_y1, precision = ML_Custom_FixedPoint_Format(11, 128-11, False))
    
    ### Evaluation Scheme ###
    
    print "\n\033[1mGenerate the evaluation scheme:\033[0m"
    input_var = self.implementation.add_input_variable("input_var", self.precision) 
    ve = ExponentExtraction(input_var, tag = "x_exponent", debug = debugd)
    vx = MantissaExtraction(input_var, tag = "x_mantissa", precision = ML_Custom_FixedPoint_Format(0,52,False), debug = debug_lftolx)
    #vx = MantissaExtraction(input_var, tag = "x_mantissa", precision = self.precision, debug = debug_lftolx)

    print "filtering and handling special cases"
    test_is_special_cases = LogicalNot(Test(input_var, specifier = Test.IsIEEENormalPositive, likely = True, debug = debugd, tag = "is_special_cases"))
    handling_special_cases = Statement(
      ConditionBlock(
        Test(input_var, specifier = Test.IsSignalingNaN, debug = True),
        ExpRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision))
      ),
      ConditionBlock(
        Test(input_var, specifier = Test.IsNaN, debug = True),
        Return(input_var)
      )#,
      # TODO: add tests for x == 0 (raise DivideByZero, return -Inf), x < 0 (raise InvalidOperation, return qNaN)
      # all that remains is x is a subnormal positive
      #Statement(
      #  ReferenceAssign(Dereference(ve), Subtraction(ve, Subtraction(CountLeadingZeros(input_var, tag = 'subnormal_clz', precision = ve.get_precision()), Constant(12, precision = ve.get_precision())))),
      #  ReferenceAssign(Dereference(vx), BitLogicLeftShift(vx, Addition(CountLeadingZeros(input_var, tag = 'subnormal_clz', precision = ve.get_precision()), Constant(1, precision = ve.get_precision()))))
      #)
    )
    
    print "doing the argument reduction"
    v_dx = vx
    v_x1 = Conversion(v_dx, tag = 'x1',
                      precision = ML_Custom_FixedPoint_Format(0,arg_reduc['size1'],False),
                      rounding_mode = ML_RoundTowardMinusInfty)
    v_index_x = TypeCast(v_x1, tag = 'index_x',
                        precision = ML_Int32) #ML_Custom_FixedPoint_Format(v_x1.get_precision().get_c_bit_size(), 0, False))
    v_inv_x = TableLoad(inv_table_1, v_index_x, tag = 'inv_x')
    v_x = Addition(v_dx, 1, tag = 'x',
                   precision = ML_Custom_FixedPoint_Format(1,52,False))
    v_dy = Multiplication(v_x, v_inv_x, tag = 'dy',
                          precision = ML_Custom_FixedPoint_Format(0,52+arg_reduc['prec1'],False))
    v_y1 = Conversion(v_dy, tag = 'y1',
                      precision = ML_Custom_FixedPoint_Format(0,arg_reduc['size2'],False),
                      rounding_mode = ML_RoundTowardMinusInfty)
    v_index_y = TypeCast(v_y1, tag = 'index_y',
                        precision = ML_Int32) #ML_Custom_FixedPoint_Format(v_y1.get_precision().get_c_bit_size(), 0, False))
    v_inv_y = TableLoad(inv_table_2, v_index_y, tag = 'inv_y')
    v_y = Addition(v_dy, 1, tag = 'y',
                   precision = ML_Custom_FixedPoint_Format(1,52+arg_reduc['prec2'],False))
    # note that we limit the number of bits used to represent dz to 64.
    # we proved during the arg reduction that we can do that (sup(out_interval) < 2^(64-52-prec1-prec2))
    v_dz = Multiplication(v_y, v_inv_y, tag = 'z',
                          precision = ML_Custom_FixedPoint_Format(64-52-arg_reduc['prec1']-arg_reduc['prec2'],52+arg_reduc['prec1']+arg_reduc['prec2'],False))
    # reduce the number of bits used to represent dz. we can do that
    
    print "doing the first polynomial evaluation"
    global_poly1_object = Polynomial.build_from_approximation(log(1+x)/x, arg_reduc['degree_poly1']-1, [64] * (arg_reduc['degree_poly1']), arg_reduc['out_interval'], fixed, absolute)
    poly1_object = global_poly1_object.sub_poly(start_index = 1)
    print global_poly1_object
    print poly1_object
    poly1 = PolynomialSchemeEvaluator.generate_horner_scheme(poly1_object, v_dz, unified_precision = v_dz.get_precision())
    return ConditionBlock(test_is_special_cases, handling_special_cases, Return(poly1))

    #approx_interval = Interval(0, 27021597764222975*S2**-61)
    
    #poly_degree = 1+sup(guessdegree(log(1+x)/x, approx_interval, S2**-(self.precision.get_field_size())))
    #global_poly_object = Polynomial.build_from_approximation(log(1+x)/x, poly_degree, [1] + [self.precision]*(poly_degree), approx_interval, absolute)
    #poly_object = global_poly_object.sub_poly(start_index = 1)
    #_poly = PolynomialSchemeEvaluator.generate_horner_scheme(poly_object, _red_vx, unified_precision = self.precision)
    #_poly.set_attributes(tag = "poly", debug = debug_lftolx)

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
