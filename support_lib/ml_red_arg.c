#include "ml_red_arg.h"

float payne_hanek(ph_data_t* ph_data, unsigned cst_msb, float x, unsigned n) {
  // float mantissa size
  const unsigned p = 24;
  const unsigned k = ph_data.k;

  unsigned e, msb_exp, msb_index, lsb_exp, lsb_index;

  e = ml_exp_extraction_dirty_fp32(x);

  msb_exp   = -e + p - 1 - k;
  msb_index = (ph_data.cst_msb - msb_exp) / ph_data.chunk_size; 

  lsb_exp = -e + p - 1 - n;
  lsb_index = (ph_data.cst_msb - lsb_exp) / ph_data.chunk_size;

  float red_arg = 0.0f;
  unsigned i;
  for (i = lsb_index; i != msb_index; i++) {
    red_arg += x * ph_data.cst_data[i];
  }
  // msb chunk
  float msb_chunk  = ph_data.cst_data[msb_index];
  unsigned exp_msb = ml_exp_extraction_dirty_fp32(msb_chunk)
  unsigned sgn_msb = float_to_32b_encoding(x) & 0x80000000; 
  unsigned delta   = exp_msb - k;

  unsigned new_exp = exp_msb - (delta + 1); 
  unsigned mant_msb = ((float_to_32b_encoding(x) & 0x7fffff) | 0x800000);
  mant_msb = (mant_msb << (delta +1)) & 0xffffff;
  unsigned deltap = __builtin_clz(mant_msb) - 8;
  new_exp = new_exp - deltap;
  mant_msb = mant_msb << deltap;

  float corrected_msb_chunk = float_from_32b_encoding(sgn_msb | ((new_exp + 127) << 23) | mant_msb);

  red_arg += x * corrected_msb_chunk;


}
  
