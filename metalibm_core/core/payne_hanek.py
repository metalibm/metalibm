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
# Dynamic implementation of Payne and Hanek argument reduction
# created:        Aug 24th, 2015
# last modified:  Mar  7th, 2018
###############################################################################


import sollya

from sollya import floor, ceil, log2
S2 = sollya.SollyaObject(2)

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

## Generate a partial integer remainder
#  result = precision((int(vx) >> k) << k)
#
#  @param vx input node
#  @param precision output precision
#  @param k number of LSB bits to be zeroed
#  @param debug debug attributes
#  @param tag node name
#  @return masked remainder converted to @p precision 
def get_remainder(vx, precision, k, debug = None, tag = ""):
    """ get in floating-point format <precision>
        the integer part of vx with the k least
        significant bits zeroed """
    int_precision = precision.get_integer_format()
    result  = Conversion(
        BitLogicAnd(
            NearestInteger(vx, precision = int_precision),
            Constant(~(2**(k+1)-1), precision = int_precision),
            tag = tag,
            debug = debug
        ),
        precision = precision
    )
    return result

## Generate a Payne&Hanek reduction node graph
#
#  @param vx input node
#  @param frac_pi constant considered during reduction
#  @param precision output format
#  @param n number of representative bits to be considered
#  @param k
#  @param chunk_num
#  @param debug debug enabling flag / attribute
#
#  @return
def generate_payne_hanek(
    vx, frac_pi, precision, n = 100, k = 4, chunk_num = None, debug = False
  ):
  """ generate payne and hanek argument reduction for frac_pi * variable """
  
  sollya.roundingwarnings = sollya.off
  debug_precision = debug_multi
  int_precision = {
    ML_Binary32 : ML_Int32,
    ML_Binary64 : ML_Int64
    }[precision]
  
  p = precision.get_field_size()

  # weight of the most significant digit of the constant
  cst_msb = floor(log2(abs(frac_pi)))
  # length of exponent range which must be covered by the approximation
  # of the constant
  cst_exp_range = cst_msb - precision.get_emin_subnormal() + 1

  # chunk size has to be so than multiplication by a splitted <v>
  # (vx_hi or vx_lo) is exact
  chunk_size = precision.get_field_size() / 2 - 2
  chunk_number = int(ceil((cst_exp_range + chunk_size - 1) / chunk_size)) 
  scaling_factor = S2**-(chunk_size/2)

  chunk_size_cst = Constant(chunk_size, precision = ML_Int32)
  cst_msb_node   = Constant(cst_msb, precision = ML_Int32)

  # Saving sollya's global precision
  old_global_prec = sollya.settings.prec
  sollya.settings.prec (cst_exp_range + n)

  # table to store chunk of constant multiplicand
  cst_table = ML_NewTable(
    dimensions = [chunk_number, 1],
    storage_precision = precision, tag = "PH_cst_table"
  )
  # table to store sqrt(scaling_factor) corresponding to the
  # cst multiplicand chunks
  scale_table =  ML_NewTable(
    dimensions = [chunk_number, 1],
    storage_precision = precision, tag = "PH_scale_table"
  )
  tmp_cst = frac_pi

  # cst_table stores normalized constant chunks (they have been
  # scale back to close to 1.0 interval)
  #
  # scale_table stores the scaling factors corresponding to the
  # denormalization of cst_table coefficients

  # this loop divide the digits of frac_pi into chunks 
  # the chunk lsb weight is given by a shift from 
  # cst_msb, multiple of the chunk index
  for i in range(chunk_number):
    value_div_factor = S2**(chunk_size * (i+1) - cst_msb)
    local_cst = int(tmp_cst * value_div_factor) / value_div_factor 
    local_scale = (scaling_factor**i)
    # storing scaled constant chunks
    cst_table[i][0] = local_cst / (local_scale**2)
    scale_table[i][0] = local_scale
    # Updating constant value
    tmp_cst = tmp_cst - local_cst

  # Computing which part of the constant we do not need to multiply
  # In the following comments, vi represents the bit of frac_pi of weight 2**-i
  
  # Bits vi so that i <= (vx_exp - p + 1 -k)  are not needed, because they result
  # in a multiple of 2pi and do not contribute to trig functions.    

  vx_exp = ExponentExtraction(vx)
  
  msb_exp = -(vx_exp - p + 1 - k)
  msb_exp.set_attributes(tag = "msb_exp", debug = debug_multi)

  # Select the highest index where the reduction should start
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

  # hi part (most significant digit) of vx input
  vx_hi = TypeCast(BitLogicAnd(TypeCast(vx, precision = int_precision), Constant(~int(2**half_size-1), precision = int_precision)), precision = precision) 
  vx_hi.set_attributes(tag = "vx_hi_ph")#, debug = debug_multi)

  vx_lo = vx - vx_hi
  vx_lo.set_attributes(tag = "vx_lo_ph")#, debug = debug_multi)
  
# loop iterator variable	
  vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
  # step scaling factor
  half_scaling = Constant(S2**(-chunk_size/2), precision = precision)


  i1 = Constant(1, precision = ML_Int32)

  # accumulator to the output precision
  acc     = Variable("acc", precision = precision, var_type = Variable.Local)
  # integer accumulator
  acc_int = Variable("acc_int", precision = int_precision, var_type = Variable.Local)

  init_loop = Statement(
    vx_hi,
    vx_lo, 
  
    ReferenceAssign(vi, msb_index), 
    ReferenceAssign(acc, Constant(0, precision = precision)),
    ReferenceAssign(acc_int, Constant(0, precision = int_precision)),
  )
  
  cst_load = TableLoad(cst_table, vi, 0, tag = "cst_load", debug = debug_precision)
  sca_load = TableLoad(scale_table, vi, 0, tag = "sca_load", debug = debug_precision)
  # loop body
  # hi_mult = vx_hi * <scale_factor> * <cst>
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
  exclude_hi.set_attributes(tag = "exclude_hi", debug = debug_multi)

  hi_mult_red = Select(exclude_hi, pre_hi_mult_red, Constant(0, precision = precision))
  hi_mult_int = Select(exclude_hi, pre_hi_mult_int, Constant(0, precision = int_precision))

  # lo part of the chunk reduction
  lo_mult = (vx_lo * sca_load) * (cst_load * sca_load)
  lo_mult.set_attributes(tag = "lo_mult")#, debug = debug_multi)
  lo_mult_int   = NearestInteger(lo_mult, precision = int_precision, tag = "lo_mult_int")#, debug = debug_multi
  lo_mult_int_f = Conversion(lo_mult_int, precision = precision, tag = "lo_mult_int_f")#, debug = debug_multi)
  lo_mult_red = (lo_mult - lo_mult_int_f).modify_attributes(tag = "lo_mult_red")#, debug = debug_multi)

  # accumulating fractional part
  acc_expr = (acc + hi_mult_red) + lo_mult_red
  # accumulating integer part
  int_expr = ((acc_int + hi_mult_int) + lo_mult_int) % 2**(k+1) 

  CF1 = Constant(1, precision = precision)
  CI1 = Constant(1, precision = int_precision)

  # extracting exceeding integer part in fractionnal accumulator
  acc_expr_int = NearestInteger(acc_expr, precision = int_precision)
  # normalizing integer and fractionnal accumulator by subtracting then
  # adding exceeding integer part
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
