# -*- coding: utf-8 -*-

# Dynamic implementation of Payne and Hanek argument reduction
# created:        Augest 24th, 2015
# last modified:  August 26th, 2015


from pythonsollya import *

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.utility.ml_template import ML_ArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

def get_remainder(vx, precision, k, debug = None, tag = ""):
  """ get in floating-point format <precision>
      the integer part of vx with the k least
      significant bits zeroed """
  int_precision = {
    ML_Binary64: ML_Int64,
    ML_Binary32: ML_Int32
  }[precision]
  result  = Conversion(
                BitLogicAnd(
                    NearestInteger(
                      vx, precision = int_precision), 
                    Constant(~(2**(k+1)-1), precision = int_precision),
                    tag = tag, 
                    debug = debug
                  ), 
                precision = precision
              )
  return result

def generate_payne_hanek(vx, frac_pi, precision, n = 100, k = 4, chunk_num = None, debug = False ):
  """ generate payne and hanek argument reduction for frac_pi * variable """
  # determining integer format corresponding to 
  # floating point precision argument
  int_precision = {
    ML_Binary64: ML_Int64,
    ML_Binary32: ML_Int32
  }[precision]

  cst_msb = floor(log2(abs(frac_pi)))
  cst_exp_range = cst_msb - precision.get_emin_subnormal() + 1

  # chunk size has to be so than multiplication by a splitted <v> (vx_hi or vx_lo)
  # is exact
  chunk_size = 20 # precision.get_field_size() / 2 - 2
  chunk_number = int(ceil((cst_exp_range + chunk_size - 1) / chunk_size)) 
  scaling_factor = S2**-(chunk_size/2)

  chunk_size_cst = Constant(chunk_size, precision = ML_Int32)
  cst_msb_node   = Constant(cst_msb, precision = ML_Int32)

  p = precision.get_field_size()

  # adapting debug format to precision argument
  debug_precision = {ML_Binary32: debug_ftox, ML_Binary64: debug_lftolx}[precision] if debug else None

  # saving sollya's global precision
  old_global_prec = get_prec()
  prec(cst_exp_range + 100)

  # table to store chunk of constant multiplicand
  cst_table = ML_Table(dimensions = [chunk_number, 1], storage_precision = precision, tag = "PH_cst_table")
  # table to store sqrt(scaling_factor) corresponding to the cst multiplicand chunks
  scale_table =  ML_Table(dimensions = [chunk_number, 1], storage_precision = precision, tag = "PH_scale_table")
  tmp_cst = frac_pi
  
  # this loop divide the digits of frac_pi into chunks 
  # the chunk lsb weight is given by a shift from 
  # cst_msb, multiple of the chunk index
  for i in xrange(chunk_number):
    value_div_factor = S2**(chunk_size * (i+1) - cst_msb)
    local_cst = int(tmp_cst * value_div_factor) / value_div_factor 
    local_scale = (scaling_factor**i)
    # storing scaled constant chunks
    cst_table[i][0] = local_cst / (local_scale**2)
    scale_table[i][0] = local_scale
    tmp_cst = tmp_cst - local_cst

  vx_exp = ExponentExtraction(vx)
  msb_exp = -vx_exp + p - 1 + k
  msb_exp.set_attributes(tag = "msb_exp", debug = (debugd if debug else None))

  msb_index = Select(cst_msb_node < msb_exp, 0, (cst_msb_node - msb_exp) / chunk_size_cst)
  msb_index.set_attributes(tag = "msb_index", debug = (debugd if debug else None))

  lsb_exp = -vx_exp + p - 1 -n
  lsb_exp.set_attributes(tag = "lsb_exp", debug = (debugd if debug else None))

  lsb_index = (cst_msb_node - lsb_exp) / chunk_size_cst
  lsb_index.set_attributes(tag = "lsb_index", debug = (debugd if debug else None))


  half_size = precision.get_field_size() / 2 + 1

  vx_hi = TypeCast(BitLogicAnd(TypeCast(vx, precision = ML_Int64), Constant(~(2**half_size-1), precision = ML_Int64)), precision = precision) 
  vx_hi.set_attributes(tag = "vx_hi", debug = debug_precision)

  vx_lo = vx - vx_hi
  vx_lo.set_attributes(tag = "vx_lo", debug = debug_precision)

  vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)

  half_scaling = Constant(S2**(-chunk_size/2), precision = precision)


  i1 = Constant(1, precision = ML_Int32)

  acc     = Variable("acc", precision = precision, var_type = Variable.Local)
  acc_int = Variable("acc_int", precision = int_precision, var_type = Variable.Local)

  init_loop = Statement(
    vx_hi,
    vx_lo, 
  
    ReferenceAssign(vi, msb_index), 
    ReferenceAssign(acc, Constant(0, precision = precision)),
    ReferenceAssign(acc_int, Constant(0, precision = precision)),
  )
  
  cst_load = TableLoad(cst_table, vi, 0, tag = "cst_load", debug = debug_precision)
  sca_load = TableLoad(scale_table, vi, 0, tag = "sca_load", debug = debug_precision)

  hi_mult = (vx_hi * sca_load) * (cst_load * sca_load)
  hi_mult.set_attributes(tag = "hi_mult", debug = debug_precision)
  pre_hi_mult_int   = NearestInteger(hi_mult, precision = int_precision, tag = "hi_mult_int", debug = (debuglld if debug else None))
  hi_mult_int_f = Conversion(pre_hi_mult_int, precision = precision, tag = "hi_mult_int_f", debug = debug_precision)
  pre_hi_mult_red = (hi_mult - hi_mult_int_f).modify_attributes(tag = "hi_mult_red", debug = debug_precision)

  # for the first chunks (vx_hi * <constant chunk>) exceeds 2**k+1 and may be 
  # discard (whereas it may lead to overflow during integer conversion
  pre_exclude_hi = ((cst_msb_node - (vi + i1) * chunk_size + i1) + (vx_exp + Constant(- half_size + 1, precision = ML_Int32))).modify_attributes(tag = "pre_exclude_hi", debug = (debugd if debug else None)) 
  pre_exclude_hi.propagate_precision(ML_Int32, [cst_msb_node, vi, vx_exp, i1])
  Ck = Constant(k, precision = ML_Int32)
  exclude_hi = pre_exclude_hi <= Ck
  exclude_hi.set_attributes(tag = "exclude_hi", debug = (debugd if debug else None))

  hi_mult_red = Select(exclude_hi, pre_hi_mult_red, Constant(0, precision = precision))
  hi_mult_int = Select(exclude_hi, pre_hi_mult_int, Constant(0, precision = int_precision))

  lo_mult = (vx_lo * sca_load) * (cst_load * sca_load)
  lo_mult.set_attributes(tag = "lo_mult", debug = debug_precision)
  lo_mult_int   = NearestInteger(lo_mult, precision = int_precision, tag = "lo_mult_int", debug = (debuglld if debug else None))
  lo_mult_int_f = Conversion(lo_mult_int, precision = precision, tag = "lo_mult_int_f", debug = debug_precision)
  lo_mult_red = (lo_mult - lo_mult_int_f).modify_attributes(tag = "lo_mult_red", debug = debug_precision)

  acc_expr = (acc + hi_mult_red) + lo_mult_red
  int_expr = ((acc_int + hi_mult_int) + lo_mult_int) % 2**(k+1) 

  CF1 = Constant(1, precision = precision)
  CI1 = Constant(1, precision = int_precision)

  acc_expr_int = NearestInteger(acc_expr, precision = int_precision)

  normalization = Statement(
      ReferenceAssign(acc, acc_expr - Conversion(acc_expr_int, precision = precision)),
      ReferenceAssign(acc_int, int_expr + acc_expr_int),
  )


  acc_expr.set_attributes(tag = "acc_expr", debug = debug_precision)
  int_expr.set_attributes(tag = "int_expr", debug = (debuglld if debug else None))

  red_loop = Loop(init_loop,
      vi <= lsb_index,
       Statement(
          acc_expr, 
          int_expr,
          normalization,
          #ReferenceAssign(acc, acc_expr), 
          #ReferenceAssign(acc_int, int_expr),
          ReferenceAssign(vi, vi + 1)
        )
      )
  result = Statement(lsb_index, msb_index, red_loop) 

  # restoring sollya's global precision
  prec(old_global_prec)

  return result, acc, acc_int



    

