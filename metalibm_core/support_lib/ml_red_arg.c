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
  .k = 4, 
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

double ph_data_d[] = { 
  0x2.8be60db8p0,
  0x1.391054a8p-32,
  -0xf.62a0b83p-72,
  0x4.d377037p-104,
  -0x2.75a99b1p-136,
  0x1.0e4107fap-168,
  -0xb.a715085p-204,
  -0x1.0ea79236p-236,
  0,
  0,
  0,
  0
};


float ph_data_hf[] = {
  0x2.8be8p120,
  -0x1.f246p104,
  -0xc.6fp84,
  0x5.4a8p68,
  -0xf.62ap48,
  -0xb.82bp28,
  -0x2.c89p12,
  0x3.6d8cp-8,
  -0x1.a99cp-24,
  0xf.10ep-44,
  0x4.108p-60,
  -0x6.ba7p-80,
  -0x1.5086p-96,
  0xe.f16p-116,
  -0x7.9238p-132,
  0x1.1b8ep-148
};

#ifdef ML_DEBUG
#define ml_printf printf
#else
#define ml_printf 
#endif


/*double payne_hanek_cosfp32(float x) {
  return payne_hanek_fp32_ext(&pio8, x, 60);
}*/

float payne_hanek_fp32(ph_data_t* ph_data, float x, unsigned n) {
  // float mantissa size
  const int p = 24;
  const int k = ph_data->k;
  const int cst_msb = ph_data->cst_msb;
  const int chunk_size = ph_data->chunk_size;

  int e, msb_exp, msb_index, lsb_exp, lsb_index;

  e = ml_exp_extraction_dirty_fp32(x);
  ml_printf("e(x)=%d\n", e);

  msb_exp   = -e + p - 1 + k;
  msb_index = cst_msb < msb_exp ? 0 : ((cst_msb - msb_exp) / chunk_size); 

  lsb_exp = -e + p - 1 - n;
  lsb_index = (cst_msb - lsb_exp) / chunk_size;


  ml_printf("msb_exp  =%d\n", msb_exp);
  ml_printf("msb_index=%d\n", msb_index);
  ml_printf("lsb_exp  =%d\n", lsb_exp);
  ml_printf("lsb_index=%d\n", lsb_index);

  double red_arg = 0.0;

  unsigned i;
  /*for (i = lsb_index; i != msb_index; i++) {
    red_arg += x * ph_data.cst_data[i];
  }*/

  float corrected_msb_chunk;
  float msb_chunk  = ph_data->cst_data[msb_index];
  ml_printf("\nmsb_chunk: %x\n", float_to_32b_encoding(msb_chunk));
  // msb chunk
  /*if (cst_msb < msb_exp) {
    corrected_msb_chunk = msb_chunk; 
  } else {
    int exp_msb = ml_exp_extraction_dirty_fp32(msb_chunk) - msb_index * 24;
    int sgn_msb = float_to_32b_encoding(x) & 0x80000000u; 
    int delta   = exp_msb - msb_exp;

    ml_printf("exp_msb=%d, delta = %d\n", exp_msb, delta);

    int new_exp = exp_msb - (delta + 1); 
    int mant_msb = ((float_to_32b_encoding(x) & 0x7fffff) | 0x800000);
    mant_msb = (mant_msb << (delta +1)) & 0xffffff;
    int deltap = __builtin_clz(mant_msb) - 8;
    new_exp = new_exp - deltap;
    mant_msb = mant_msb << deltap;

    corrected_msb_chunk = float_from_32b_encoding(sgn_msb | ((new_exp + 127 + ph_data->exp_offset) << 23) | mant_msb);
  }*/
  corrected_msb_chunk = msb_chunk; 

  ml_printf("\ncorrected msb_chunk: %x\n", float_to_32b_encoding(corrected_msb_chunk));

  /*double red_arg_d = 0.0;
  double dx = x;
  red_arg_d += ((dx * half_scaling_table[msb_index]) * corrected_msb_chunk) * half_scaling_table[msb_index];
  ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  if (ml_exp_extraction_dirty_fp64(red_arg_d) > 52 + k) {
    // multiple of 8
    red_arg_d = 0.0;
  } else {
    red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> 3) << 3;
  }
  //ml_printf("red_arg %f/%x\n", red_arg, float_to_32b_encoding(red_arg));
  ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  for (i = msb_index + 1; i <= lsb_index && i < ph_data->data_size; i++) {
    float scale_factor = half_scaling_table[i];
    red_arg_d += ((dx * scale_factor) * ph_data->cst_data[i]) * scale_factor;
    ml_printf("i : %d\n", i);
    //ml_printf("   red_arg %f/%x\n", red_arg, float_to_32b_encoding(red_arg));
    ml_printf("    red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  }*/
  double red_arg_d = 0.0, dx = x;
  for (i = msb_index; i <= lsb_index && i < ph_data->data_size; i++) {
      red_arg_d += ((dx * half_scaling_table[i]) * ph_data->cst_data[i]) * half_scaling_table[i];
      ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
      int exp_red_arg_d = ml_exp_extraction_dirty_fp64(red_arg_d);
      if (exp_red_arg_d > 52 + k) {
        // multiple of 2^k
        red_arg_d = 0.0;
      } else if (exp_red_arg_d < k) {
      } else {
        red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> k) << k;
      }
      ml_printf("new red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));

  };



  return (float) red_arg_d;
}
  
double payne_hanek_fp32_ext(ph_data_t* ph_data, float x, unsigned n) {
  // float mantissa size
  const int p = 24;
  const int k = ph_data->k;
  const int cst_msb = ph_data->cst_msb;
  const int chunk_size = ph_data->chunk_size;

  int e, msb_exp, msb_index, lsb_exp, lsb_index;

  e = ml_exp_extraction_dirty_fp32(x);
  ml_printf("e(x)=%d\n", e);

  msb_exp   = -e + p - 1 + k;
  msb_index = cst_msb < msb_exp ? 0 : ((cst_msb - msb_exp) / chunk_size); 

  lsb_exp = -e + p - 1 - n;
  lsb_index = (cst_msb - lsb_exp) / chunk_size;


  ml_printf("msb_exp  =%d\n", msb_exp);
  ml_printf("msb_index=%d\n", msb_index);
  ml_printf("lsb_exp  =%d\n", lsb_exp);
  ml_printf("lsb_index=%d\n", lsb_index);

  double red_arg = 0.0;

  unsigned i;
  /*for (i = lsb_index; i != msb_index; i++) {
    red_arg += x * ph_data.cst_data[i];
  }*/

  float corrected_msb_chunk;
  float msb_chunk  = ph_data->cst_data[msb_index];
  ml_printf("\nmsb_chunk: %x\n", float_to_32b_encoding(msb_chunk));
  corrected_msb_chunk = msb_chunk; 

  ml_printf("\ncorrected msb_chunk: %x\n", float_to_32b_encoding(corrected_msb_chunk));

  double red_arg_d = 0.0, dx = x;
  for (i = msb_index; i <= lsb_index && i < ph_data->data_size; i++) {
      float local_chunk = ph_data->cst_data[i];
      ml_printf("local_chunk: %f/%"PRIx32"\n", local_chunk, float_to_32b_encoding(local_chunk));
      //red_arg_d += ((dx * half_scaling_table[i]) * local_chunk) * half_scaling_table[i];
      red_arg_d += ((dx * half_scaling_table[i]) * local_chunk) * half_scaling_table[i];
      ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
      int exp_red_arg_d = ml_exp_extraction_dirty_fp64(red_arg_d);
      if (exp_red_arg_d > 52 + k) {
        // multiple of 2^k
        red_arg_d = 0.0;
      } else if (exp_red_arg_d < k) {
      } else {
        red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> k) << k;
      }
      ml_printf("new red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));

  };



  return red_arg_d;
}

static inline double red_to_exp(double x, uint64_t e) {
  int old_e = ml_exp_extraction_dirty_fp64(x);
  if (old_e <= e) return x;
  ml_printf("old_e=%"PRIx64", e=%"PRIx64"\n", old_e, e);
  uint64_t mask = (0xffffffffffffffffull << (53 - (old_e - e)));
  ml_printf("mask=%"PRIx64"\n", mask);
  uint64_t delta = double_to_64b_encoding(x) & mask;
  return x - double_from_64b_encoding(delta);
}

double payne_hanek_cosfp32(float x) {
  ph_data_t* ph_data = &pio8;
  const int n = 65;
  // float mantissa size
  const int p = 24;
  const int k = ph_data->k;
  const int cst_msb = ph_data->cst_msb;
  const int chunk_size = 16;//ph_data->chunk_size;

  int e, msb_exp, msb_index, lsb_exp, lsb_index;

  e = ml_exp_extraction_dirty_fp32(x);
  ml_printf("e(x)=%d\n", e);

  msb_exp   = -e + p - 1 + k;
  msb_index = cst_msb < msb_exp ? 0 : ((cst_msb - msb_exp) / chunk_size); 

  lsb_exp = -e + p - 1 - n;
  lsb_index = (cst_msb - lsb_exp) / chunk_size;


  ml_printf("msb_exp  =%d\n", msb_exp);
  ml_printf("msb_index=%d\n", msb_index);
  ml_printf("lsb_exp  =%d\n", lsb_exp);
  ml_printf("lsb_index=%d\n", lsb_index);

  double red_arg = 0.0;

  unsigned i;
  /*for (i = lsb_index; i != msb_index; i++) {
    red_arg += x * ph_data.cst_data[i];
  }*/

  //float corrected_msb_chunk;
  //float msb_chunk  = ph_data->cst_data[msb_index];
  //ml_printf("\nmsb_chunk: %x\n", float_to_32b_encoding(msb_chunk));
  //corrected_msb_chunk = msb_chunk; 

  //ml_printf("\ncorrected msb_chunk: %x\n", float_to_32b_encoding(corrected_msb_chunk));

  //double scaling = half_scaling_table[msb_index];

  float rx = ml_mantissa_extraction_fp32(x); 

  float scalingf = 0x1p-120f * ml_exp_insertion_fp32(e);
  ml_printf("scalingf = %f/%"PRIx32"\n", scalingf, float_to_32b_encoding(scalingf));

  double red_arg_d = 0.0, dx = x;
  unsigned j;
  for (i = msb_index, j = 0; j < 3 && i <= lsb_index && i < 8; i++, j++) {
      ml_printf("========\n");
      ml_printf("i: %d\n", i);
      ml_printf("chunk: %f/%"PRIx64"\n", ph_data_d[i], double_to_64b_encoding(ph_data_d[i]));
      red_arg_d += rx * (ph_data_hf[i] * scalingf);
      ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
      red_arg_d = red_to_exp(red_arg_d, 3);
      ml_printf("new red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  };
  for (; i <= lsb_index && i < 8; i++) {
      ml_printf("========\n");
      ml_printf("i: %d\n", i);
      ml_printf("chunk: %f/%"PRIx64"\n", ph_data_d[i], double_to_64b_encoding(ph_data_d[i]));
      red_arg_d += rx * (ph_data_hf[i] * scalingf);
      ml_printf("red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
      ml_printf("new red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  };

  //red_arg_d *= 0x1p-120 * ml_exp_insertion_fp64(e);
  ml_printf("f0 red_arg_d %f/%"PRIx64"\n", red_arg_d, double_to_64b_encoding(red_arg_d));
  //red_arg_d -= (__builtin_k1_fixedd(_K1_FPU_NEAREST_EVEN, red_arg_d, 0) >> k) << k;


  double result;
  asm __volatile__ (
    "fabsd %d0 = %d1\n"
    ";;\n"
    : "=r"(result)
    : "r"(red_arg_d)
    :
  );
  ml_printf("result %f/%"PRIx64"\n", result, double_to_64b_encoding(result));
  return result;
}
