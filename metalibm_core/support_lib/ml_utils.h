/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2013)
* All rights reserved
* created:          Dec 23rd, 2013
* last-modified:    Dec 23rd, 2013
*
* author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
*******************************************************************************/
#include <stdint.h>
#include <support_lib/ml_types.h>

#ifndef __UTILS_H__
#define __UTILS_H__


int ml_isnanf(float); 


/** cast from a binary32 float to its uint32_t binary encoding */
//uint32_t float_to_32b_encoding(float v);
static inline uint32_t float_to_32b_encoding(float v) {
    uif_conv_t conv_tmp;
    conv_tmp.f = v;
    return conv_tmp.u;
}


/** cast from a uin32_t binary encoding to its binary32 float value */
//float float_from_32b_encoding(uint32_t v);
static inline float float_from_32b_encoding(uint32_t v) {
    uif_conv_t conv_tmp;
    conv_tmp.u = v;
    return conv_tmp.f;
}


static inline uint64_t double_to_64b_encoding(double v) {
    uid_conv_t conv_tmp;
    conv_tmp.d = v;
    return conv_tmp.u;
}

static inline double double_from_64b_encoding(uint64_t v) {
    uid_conv_t conv_tmp;
    conv_tmp.u = v;
    return conv_tmp.d;
}

/** Metalibm implementation of x * y + z with single rounding 
 * rely on HW FMA (dot not use on architecture not providing one)
 *  */
static inline float ml_fmaf(float x, float y, float z) {
    double dx = x;
    double dy = y;
    double dz = z;

    float result = (dx * dy) + dz;
    return result;
}

/** count leading zeroes */
#if defined(__GNUC__)
static inline int ml_count_leading_zeros_32b (uint32_t x) {
    return (x == 0) ? 0 : __builtin_clzl (x);
}
static inline int ml_count_leading_zeros_64b (uint64_t x) {
	return (x == 0) ? 0 : __builtin_clzll(x);
}
#else
static const uint8_t ml_clz_lkup[256] = {
    64, 63, 62, 62, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60,
    59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56
};
static inline int ml_count_leading_zeros_32b (uint32_t x) {
    int n = ((x)   >= (UINT32_C(1) << 16)) * 16;
    n += ((x >> n) >= (UINT32_C(1) <<  8)) *  8;
    return clz_lkup[x >> n] - n;
}
static inline int ml_count_leading_zeros_64b (uint64_t x) {
    int n = ((x)   >= (UINT64_C(1) << 32)) * 32;
    n += ((x >> n) >= (UINT64_C(1) << 16)) * 16;
    n += ((x >> n) >= (UINT64_C(1) <<  8)) *  8;
    return clz_lkup[x >> n] - n;
}
#endif



#define ml_exp_insertion_fp32(k) (float_from_32b_encoding(((k + 127) << 23) & 0x7f800000))
#define ml_exp_insertion_fp64(k) (double_from_64b_encoding(((uint64_t) (k + 1023) << 52) & 0x7ff0000000000000ull))

#define ml_exp_insertion_no_offset_fp32(k) (float_from_32b_encoding(((k) << 23)))
#define ml_exp_insertion_no_offset_fp64(k) (double_from_64b_encoding(((uint64_t) (k) << 52)))

#define ml_sign_exp_insertion_fp64(aligned_sign, k) (double_from_64b_encoding((((uint64_t) (k + 1023) << 52) & 0x7ff0000000000000ull)| (aligned_sign)))

#define ml_aligned_sign_extraction_fp64(x) (double_to_64b_encoding(x) & 0x8000000000000000ull)

#define ml_exp_extraction_dirty_fp32(x) (((float_to_32b_encoding(x) >> 23) & 0xff) - 127) 
#define ml_exp_extraction_dirty_fp64(x) (((double_to_64b_encoding(x) >> 52) & 0x7ff) - 1023)

#define ml_mantissa_extraction_fp32(x)  (float_from_32b_encoding((float_to_32b_encoding(x) & 0x807fffff) | 0x3f800000))
#define ml_mantissa_extraction_fp64(x)  (double_from_64b_encoding((double_to_64b_encoding(x) & 0x800fffffffffffffull) | 0x3ff0000000000000ull))

#define ml_raw_sign_exp_extraction_fp32(x) ((int32_t)float_to_32b_encoding(x) >> 9) 
#define ml_raw_sign_exp_extraction_fp64(x) ((int64_t)double_to_64b_encoding(x) >> 12) 

#define ml_raw_mantissa_extraction_fp32(x) (float_to_32b_encoding(x) & 0x007FFFFF) 
#define ml_raw_mantissa_extraction_fp64(x) (double_to_64b_encoding(x) & 0x000fffffffffffffull) 

#define ml_is_normal_positive_fp64(x) ((uint64_t)(double_to_64b_encoding(x) >> 52) - 1u < 0x7FEu)
#define ml_is_normal_positive_fp32(x) ((uint64_t)(double_to_32b_encoding(x) >> 23) - 1u < 0x0FEu)

#define ml_is_nan_or_inff(x) ((float_to_32b_encoding(x) & 0x7f800000u) == 0x7f800000)
#define ml_is_nan_or_inf(x) ((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull)

#define ml_is_nanf(x) (((float_to_32b_encoding(x) & 0x7f800000u) == 0x7f800000) && (float_to_32b_encoding(x) & 0x007fffffu) != 0)
#define ml_is_nan(x) (((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) && ((double_to_64b_encoding(x) & 0x000fffffffffffffull) != 0))

#define ml_is_signaling_nanf(x) (ml_is_nanf(x) && ((float_to_32b_encoding(x) & 0x00400000) == 0))
#define ml_is_signaling_nan(x) (ml_is_nan(x) && ((double_to_64b_encoding(x) & 0x0008000000000000ull) == 0))

#define ml_is_quiet_nanf(x) (ml_is_nanf(x) && ((float_to_32b_encoding(x) & 0x00400000) != 0))
#define ml_is_quiet_nan(x) (ml_is_nan(x) && ((double_to_64b_encoding(x) & 0x0008000000000000ull) != 0))

#define ml_is_inff(x) ((float_to_32b_encoding(x) & 0x7fffffffu) == 0x7f800000)
#define ml_is_inf(x) ((double_to_64b_encoding(x) & 0x7fffffffffffffffull) == 0x7ff0000000000000ull)

#define ml_is_plus_inff(x) ((float_to_32b_encoding(x)) == 0x7f800000)
#define ml_is_plus_inf(x) ((double_to_64b_encoding(x)) == 0x7ff0000000000000ull)

#define ml_is_minus_inff(x) ((float_to_32b_encoding(x)) == 0xff800000)
#define ml_is_minus_inf(x) ((double_to_64b_encoding(x)) == 0xfff0000000000000ull)

#define ml_is_zerof(x) ((float_to_32b_encoding(x) & 0x7fffffffu) == 0)
#define ml_is_zero(x)  ((double_to_64b_encoding(x) & 0x7fffffffffffffffull) == 0)

#define ml_is_positivezerof(x) ((float_to_32b_encoding(x)) == 0)
#define ml_is_positivezero(x)  ((double_to_64b_encoding(x)) == 0)

#define ml_is_negativezerof(x) (float_to_32b_encoding(x) == 0x80000000u)
#define ml_is_negativezero(x)  (double_to_64b_encoding(x) == 0x8000000000000000ull)

#define ml_is_normalf(x) (float_to_32b_encoding(x) & 0x7f800000u)
#define ml_is_normal(x) (double_to_64b_encoding(x) & 0x7ff0000000000000ull)

#define ml_is_subnormalf(x) ((float_to_32b_encoding(x) & 0x7f800000u) == 0)
#define ml_is_subnormal(x) ((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0)


#define ml_comp_signf(x, y) ((float_to_32b_encoding(x) ^ float_to_32b_encoding(y)) & 0x80000000u)
#define ml_comp_sign(x, y)  ((double_to_64b_encoding(x) ^ double_to_64b_encoding(y)) >> 63)

#define ml_copy_signf(x, y) (float_from_32b_encoding((float_to_32b_encoding(x) & 0x80000000u) | float_to_32b_encoding(y)))
#define ml_copy_sign(x, y) (double_from_64b_encoding((double_to_64b_encoding(x) & 0x8000000000000000ull) | double_to_64b_encoding(y)))

#endif
