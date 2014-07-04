#ifndef __K1A_UTILS_H__
#define __K1A_UTILS_H__

#include <stdint.h>
#include <support_lib/ml_types.h>
#include <HAL/hal/hal.h>
#include <cpu.h>

double k1_fma_d4_rn(double x, double y, double z); 

ml_dd_t k1_fma_dd_d3_rn(double x, double y, double z); 

double k1_subnormalize_d_dd_i(ml_dd_t x, int scale_factor);

double k1_round_signed_overflow_fp64(double sign);
float  k1_round_signed_overflow_fp32(float  sign);

void k1_save_fp_context();
void k1_restore_fp_context();

typedef enum __k1_fpu_rounding_mode ml_rnd_mode_t;

#endif
