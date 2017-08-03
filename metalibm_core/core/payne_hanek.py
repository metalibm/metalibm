# -*- coding: utf-8 -*-

# Dynamic implementation of Payne and Hanek argument reduction
# created:        Augest 24th, 2015
# last modified:  August 26th, 2015


from sollya import floor, ceil, log2, S2, settings

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
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
  int_precision = precision.get_integer_format()
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

def generate_payne_hanek(vx, frac_pi, precision, n = 100, k = 4, chunk_num = None, debug = False):
  """ generate payne and hanek argument reduction for frac_pi * variable """
  
  sollya.roundingwarnings = sollya.off
  
  int_precision = {
    ML_Binary32 : ML_Int32,
    ML_Binary64 : ML_Int64
    }[precision]
  
  p = precision.get_field_size()

  ## Constant msb
  cst_msb = floor(log2(abs(frac_pi)))
  cst_exp_range = cst_msb + 500 + 1

  # Chunk size has to be so that multiplication by a splitted <v> (vx_hi or vx_lo)
  # is exact, <vx_hi> or <vx_lo> has at most p/2 significant bits
  chunk_size = p / 2 - 2
  
  chunk_number = int(ceil((cst_exp_range + chunk_size - 1) / chunk_size)) 
  scaling_factor = S2**-(chunk_size/2)

  chunk_size_cst = Constant(chunk_size, precision = int_precision)
  cst_msb_node   = Constant(cst_msb, precision = int_precision)

  # Saving sollya's global precision
  old_global_prec = sollya.settings.prec
  sollya.settings.prec (cst_exp_range + n)

  # Table to store chunks of constant multiplicand
  cst_table = ML_NewTable(dimensions = [chunk_number, 1], storage_precision = precision, tag = "PH_cst_table")
  # Table to store sqrt(scaling_factor) corresponding to the constant multiplicand chunks
  scale_table =  ML_NewTable(dimensions = [chunk_number, 1], storage_precision = precision, tag = "PH_scale_table")
  
  # Constant value
  tmp_cst = frac_pi
  
  # Dividing frac_pi into <chunk_number> chunks
  for i in xrange(chunk_number):
    # Shift from constant msb to get current chunk's lsb
    chunk_lsb = ((i+1)*chunk_size - cst_msb)
    chunk_value = int(tmp_cst * S2**chunk_lsb) / S2**chunk_lsb 
    local_scale = (scaling_factor**i)
    
    # Storing scaled constant chunks
    cst_table[i][0] = chunk_value / (local_scale**2)
    scale_table[i][0] = local_scale
    # Updating constant value
    tmp_cst = tmp_cst - chunk_value

  # Computing which part of the constant we do not need to multiply
  # In the following comments, vi represents the bit of frac_pi of weight 2**-i
  
  # Bits vi so that i <= (vx_exp - p + 1 -k)  are not needed, because they result
  # in a multiple of 2pi and do not contribute to trig functions.    

  vx_exp = ExponentExtraction(vx)
  
  msb_exp = -(vx_exp - p + 1 - k)
  msb_exp.set_attributes(precision = int_precision, tag = "msb_exp", debug = debug_multi)

  # Index of the corresponding chunk
  msb_index = Select(cst_msb_node < msb_exp, 0, (cst_msb_node - msb_exp) / chunk_size_cst)
  msb_index.set_attributes(tag = "msb_index", debug = debug_multi)

  # For a desired accuracy of 2**-n, bits vi so that i >= (vx_exp + n + 4)  are not needed, because they contribute less than
  # 2**-n to the result
  
  lsb_exp = -(vx_exp + n + 4)
  lsb_exp.set_attributes(tag = "lsb_exp", debug = debug_multi)

  # Index of the corresponding chunk
  lsb_index = (cst_msb_node - lsb_exp) / chunk_size_cst
  lsb_index.set_attributes(tag = "lsb_index", debug = debug_multi)

  # Splitting vx
  half_size = precision.get_field_size() / 2 + 1
  half_scaling = Constant(S2**(-chunk_size/2), precision = precision)

  vx_hi = TypeCast(BitLogicAnd(TypeCast(vx, precision = int_precision), Constant(~(2**half_size-1), precision = int_precision)), precision = precision) 
  vx_hi.set_attributes(tag = "vx_hi_ph")#, debug = debug_multi)

  vx_lo = vx - vx_hi
  vx_lo.set_attributes(tag = "vx_lo_ph")#, debug = debug_multi)

  i1 = Constant(1, precision = ML_Int32)
  vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
  acc     = Variable("acc", precision = precision, var_type = Variable.Local)
  acc_int = Variable("acc_int", precision = int_precision, var_type = Variable.Local)

  init_loop = Statement(
    vx_hi,
    vx_lo, 
    ReferenceAssign(vi, msb_index), 
    ReferenceAssign(acc, Constant(0, precision = precision)),
    ReferenceAssign(acc_int, Constant(0, precision = precision)),
  )
  
  cst_load = TableLoad(cst_table, vi, 0, tag = "cst_load")#, debug = debug_multi)
  sca_load = TableLoad(scale_table, vi, 0, tag = "sca_load")#, debug = debug_multi)

  hi_mult = (vx_hi * sca_load) * (cst_load * sca_load)
  hi_mult.set_attributes(tag = "hi_mult")#, debug = debug_multi)
  
  pre_hi_mult_int   = NearestInteger(hi_mult, precision = int_precision, tag = "hi_mult_int")#, debug = debug_multi)
  hi_mult_int_f = Conversion(pre_hi_mult_int, precision = precision, tag = "hi_mult_int_f")#, debug = debug_multi)
  pre_hi_mult_red = (hi_mult - hi_mult_int_f).modify_attributes(tag = "hi_mult_red")#, debug = debug_multi)

  pre_exclude_hi = ((cst_msb_node - (vi + i1) * chunk_size + i1) + (vx_exp + Constant(- half_size + 1, precision = int_precision))).modify_attributes(tag = "pre_exclude_hi", debug = debug_multi) 
  pre_exclude_hi.propagate_precision(ML_Int32, [cst_msb_node, vi, vx_exp, i1])
  Ck = Constant(k, precision = ML_Int32)
  exclude_hi = pre_exclude_hi <= Ck
  exclude_hi.set_attributes(tag = "exclude_hi", debug = debug_multi)

  hi_mult_red = Select(exclude_hi, pre_hi_mult_red, Constant(0, precision = precision))
  hi_mult_int = Select(exclude_hi, pre_hi_mult_int, Constant(0, precision = int_precision))

  lo_mult = (vx_lo * sca_load) * (cst_load * sca_load)
  lo_mult.set_attributes(tag = "lo_mult")#, debug = debug_multi)
  lo_mult_int   = NearestInteger(lo_mult, precision = int_precision, tag = "lo_mult_int")#, debug = debug_multi
  lo_mult_int_f = Conversion(lo_mult_int, precision = precision, tag = "lo_mult_int_f")#, debug = debug_multi)
  lo_mult_red = (lo_mult - lo_mult_int_f).modify_attributes(tag = "lo_mult_red")#, debug = debug_multi)

  acc_expr = (acc + hi_mult_red) + lo_mult_red
  int_expr = ((acc_int + hi_mult_int) + lo_mult_int) % 2**(k+1)

  CF1 = Constant(1, precision = precision)
  CI1 = Constant(1, precision = int_precision)

  acc_expr_int = NearestInteger(acc_expr, precision = int_precision, tag = "acc_expr_int")#, debug = debug_multi)

  normalization = Statement(
      ReferenceAssign(acc, acc_expr - Conversion(acc_expr_int, precision = precision)),
      ReferenceAssign(acc_int, int_expr + acc_expr_int),
  )

  acc_expr.set_attributes(tag = "acc_expr")#, debug = debug_multi)
  int_expr.set_attributes(tag = "int_expr")#, debug = debug_multi)

  red_loop = Loop(init_loop,
      vi <= lsb_index,
       Statement(
          acc_expr, 
          int_expr,
          normalization,
          ReferenceAssign(vi, vi + 1)
        )
      )
      
  result = Statement(lsb_index, msb_index, red_loop) 

  # restoring sollya's global precision
  sollya.settings.prec = old_global_prec

  return result, acc, acc_int



    

