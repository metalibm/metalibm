#include "support_lib/ml_utils.h"
#include "support_lib/ml_red_arg.h"
#include <stdio.h>
#include "math.h"
#include <inttypes.h>

float half_scaling_table[12] = {
  0x1p0f,
  0x1p-12f,
  0x1p-24f,
  0x1p-36f,
  0x1p-48f,
  0x1p-60f,
  0x1p-72f,
  0x1p-84f,
  0x1p-96f,
  0x1p-108f,
  0x1p-120f,
  0x1p-132f
};

ph_data_t pio8 = {                                                       
  .data_size = 12,                                                       
  .cst_msb = 1,  
  .k = 3, 
  .chunk_size = 24, 
  .exp_offset = 124,                                                     
  .cst_data = {
    0x5.17cc18p-1,
    0x3.72722p-1,
    0xa.94fe1p-5,                                                        
    0x3.abe8fcp-5,
    -0x1.65912p-5,                                                       
    0x6.db14bp-9,
    -0x3.361de4p-9,
    0x8.20ff3p-13,
    -0x7.4e2a1p-13,
    -0xa.21d4fp-17,
    -0x2.46ep-17,
    0,
  }
};


float payne_hanek_cosfp32(float x) {
  return payne_hanek(&pio8, x, 50);
}

float payne_hanek(ph_data_t* ph_data, float x, unsigned n) {
  // float mantissa size
  const int p = 24;
  const int k = ph_data->k;
  const int cst_msb = ph_data->cst_msb;
  const int chunk_size = ph_data->chunk_size;

  int e, msb_exp, msb_index, lsb_exp, lsb_index;

  e = ml_exp_extraction_dirty_fp32(x);
  printf("e(x)=%d\n", e);

  msb_exp   = -e + p - 1 + k;
  msb_index = cst_msb < msb_exp ? 0 : ((cst_msb - msb_exp) / chunk_size); 

  lsb_exp = -e + p - 1 - n;
  lsb_index = (cst_msb - lsb_exp) / chunk_size;


  printf("msb_exp  =%d\n", msb_exp);
  printf("msb_index=%d\n", msb_index);
  printf("lsb_exp  =%d\n", lsb_exp);
  printf("lsb_index=%d\n", lsb_index);

  double red_arg = 0.0;

  unsigned i;
  /*for (i = lsb_index; i != msb_index; i++) {
    red_arg += x * ph_data.cst_data[i];
  }*/

  float corrected_msb_chunk;
  float msb_chunk  = ph_data->cst_data[msb_index];
  printf("\nmsb_chunk: %x\n", float_to_32b_encoding(msb_chunk));
  // msb chunk
  /*if (cst_msb < msb_exp) {
    corrected_msb_chunk = msb_chunk; 
  } else {
    int exp_msb = ml_exp_extraction_dirty_fp32(msb_chunk) - msb_index * 24;
    int sgn_msb = float_to_32b_encoding(x) & 0x80000000u; 
    int delta   = exp_msb - msb_exp;

    printf("exp_msb=%d, delta = %d\n", exp_msb, delta);

    int new_exp = exp_msb - (delta + 1); 
    int mant_msb = ((float_to_32b_encoding(x) & 0x7fffff) | 0x800000);
    mant_msb = (mant_msb << (delta +1)) & 0xffffff;
    int deltap = __builtin_clz(mant_msb) - 8;
    new_exp = new_exp - deltap;
    mant_msb = mant_msb << deltap;

    corrected_msb_chunk = float_from_32b_encoding(sgn_msb | ((new_exp + 127 + ph_data->exp_offset) << 23) | mant_msb);
  }*/
  corrected_msb_chunk = msb_chunk; 

  printf("\ncorrected msb_chunk: %x\n", float_to_32b_encoding(corrected_msb_chunk));

  /*double red_arg_d = 0.0;
  double dx = x;
  red_arg_d += ((dx * half_scaling_table[msb_index]) * corrected_msb_chunk) * half_scaling_table[msb_index];
  printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  if (ml_exp_extraction_dirty_fp64(red_arg_d) > 52 + k) {
    // multiple of 8
    red_arg_d = 0.0;
  } else {
    red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> 3) << 3;
  }
  //printf("red_arg %f/%x\n", red_arg, float_to_32b_encoding(red_arg));
  printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  for (i = msb_index + 1; i <= lsb_index && i < ph_data->data_size; i++) {
    float scale_factor = half_scaling_table[i];
    red_arg_d += ((dx * scale_factor) * ph_data->cst_data[i]) * scale_factor;
    printf("i : %d\n", i);
    //printf("   red_arg %f/%x\n", red_arg, float_to_32b_encoding(red_arg));
    printf("    red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  }*/
  double red_arg_d = 0.0, dx = x;
  for (i = msb_index; i <= lsb_index && i < ph_data->data_size; i++) {
      red_arg_d += ((dx * half_scaling_table[i]) * ph_data->cst_data[i]) * half_scaling_table[i];
      printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
      int exp_red_arg_d = ml_exp_extraction_dirty_fp64(red_arg_d);
      if (exp_red_arg_d > 52 + k) {
        // multiple of 8
        red_arg_d = 0.0;
      } else if (exp_red_arg_d < 3) {
      } else {
        red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> 3) << 3;
      }
      printf("new red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));

  };



  return (float) red_arg_d;
}
  
